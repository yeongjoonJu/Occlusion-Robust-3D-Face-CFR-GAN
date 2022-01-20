#!/usr/bin/python
# -*- encoding: utf-8 -*-
from ctypes import ArgumentError
import os ; import sys 
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

from mmRegressor.network.resnet50_task import *
from mmRegressor.preprocess_img import Preprocess
from mmRegressor.load_data import *
from mmRegressor.reconstruct_mesh import Reconstruction, Compute_rotation_matrix, _need_const

import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
# from torchsummary import summary
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    cameras, lighting,
    PointLights, HardPhongShader,
    RasterizationSettings,
    BlendParams,
    MeshRenderer, MeshRasterizer
)

# Retina Face
if os.path.exists('Pytorch_Retinaface'):
    from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
    from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
    from Pytorch_Retinaface.utils.box_utils import decode, decode_landm

class Estimator3D(object):
    def __init__(self, is_cuda=True, batch_size=1, render_size=224, test=True, model_path=None, back_white=False, cuda_id=0, det_net=None):
        self.is_cuda = is_cuda
        self.render_size = render_size
        self.cuda_id = cuda_id
        # Network, cfg
        if det_net is not None:
            self.det_net = det_net[0]
            self.det_cfg = det_net[1]

        # load models
        if model_path is None:
            print('Load pretrained weights')
        else:
            print('Load {}'.format(model_path))
        self.load_3dmm_models(model_path, test)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
                
        self.argmax = lambda i, c: c[i]
        self.thresholding = torch.nn.Threshold(0.3, 0.0)

        tri = self.face_model.tri
        tri = np.expand_dims(tri, 0)
        self.tri = torch.FloatTensor(tri).repeat(batch_size, 1, 1)

        self.skin_mask = -1*self.face_model.skin_mask.unsqueeze(-1)
        
        if is_cuda:
            device = torch.device('cuda:'+str(cuda_id))
            self.tri = self.tri.cuda(cuda_id)
        else:
            device = torch.device('cpu')

        # Camera and renderer settings
        blend_params = BlendParams(background_color=(0.0,0.0,0.0))
        if back_white:
            blend_params= BlendParams(background_color=(1.0,1.0,1.0))

        self.R, self.T = look_at_view_transform(eye=[[0,0,10]], at=[[0,0,0]], up=[[0,1,0]], device=device)
        camera = cameras.FoVPerspectiveCameras(znear=0.01, zfar=50.0, aspect_ratio=1.0, fov=12.5936, R=self.R, T=self.T, device=device)
        lights = PointLights(ambient_color=[[1.0,1.0,1.0]], device=device, location=[[0.0,0.0,1e-5]])
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=render_size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    cull_backfaces=True
                )
            ),
            shader=HardPhongShader(cameras=camera, device=device, lights=lights, blend_params=blend_params)
        )

    def load_3dmm_models(self, model_path, test=True):
        # read face model
        face_model = BFM('mmRegressor/BFM/BFM_model_80.mat', self.cuda_id)

        self.face_model = face_model

        # read standard landmarks for preprocessing images
        self.lm3D = face_model.load_lm3d("mmRegressor/BFM/similarity_Lm3D_all.mat")
        
        regressor = resnet50_use()
        if model_path is None:
            regressor.load_state_dict(torch.load("mmRegressor/network/th_model_params.pth"))
        else:
            regressor.load_state_dict(torch.load(model_path, map_location='cuda:'+str(self.cuda_id)))

        if test:
            regressor.eval()
        if self.is_cuda:
            regressor = regressor.cuda(self.cuda_id)
        if test:
            for param in regressor.parameters():
                param.requires_grad = False

        self.regressor = regressor

    def regress_3dmm(self, img):
        arr_coef = self.regressor(img)
        coef = torch.cat(arr_coef, 1)

        return coef


    def reconstruct(self, coef, test=False):
        # reconstruct 3D face with output coefficients and face model
        face_shape, _, face_color, _,face_projection,_,gamma = Reconstruction(coef,self.face_model)
        verts_rgb = face_color[...,[2,1,0]]
        mesh = Meshes(verts=face_shape, faces=self.tri[:face_shape.shape[0],...], textures=Textures(verts_rgb=verts_rgb))

        rendered = self.phong_renderer(meshes_world=mesh, R=self.R, T=self.T)
        rendered = torch.clamp(rendered, 0.0, 1.0)

        landmarks_2d = torch.zeros_like(face_projection).cuda(self.cuda_id)
        landmarks_2d[...,0] = torch.clamp(face_projection[...,0].clone(), 0, self.render_size-1)
        landmarks_2d[...,1] = torch.clamp(face_projection[...,1].clone(), 0, self.render_size-1)
        landmarks_2d[...,1] = self.render_size - landmarks_2d[...,1].clone() - 1
        landmarks_2d = landmarks_2d[:,self.face_model.keypoints,:]

        if test:
            return rendered, landmarks_2d

        tex_mean = torch.sum(face_color*self.skin_mask) / torch.sum(self.skin_mask)
        ref_loss = torch.sum(torch.square((face_color - tex_mean)*self.skin_mask)) / (face_color.shape[0]*torch.sum(self.skin_mask))

        gamma = gamma.view(-1,3,9)    
        gamma_mean = torch.mean(gamma, dim=1, keepdim=True)
        gamma_loss = torch.mean(torch.square(gamma - gamma_mean))

        return rendered, landmarks_2d, ref_loss, gamma_loss


    def estimate_and_reconstruct(self, img):
        coef = self.regress_3dmm(img)
        return self.reconstruct(coef, test=True)

    
    def estimate_five_landmarks(self, img):
        # Detect and align
        img_raw = np.array(img)
        ori_h, ori_w, _ = img_raw.shape
        img = np.float32(cv2.resize(img_raw, (320, 320), cv2.INTER_CUBIC))
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.cuda_id)
        scale = scale.to(self.cuda_id)
        
        loc, conf, landms = self.det_net(img)
        priorbox = PriorBox(self.det_cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.cuda_id)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.det_cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.det_cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.cuda_id)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.1)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:args.top_k]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        landms = landms[keep]
        landm = np.array(landms[0])
        
        landm = np.reshape(landm, (-1, 2))
        landm[:,0] *= ori_w/320
        landm[:,1] *= ori_h/320

        return landm


    def align_convert2tensor(self, img_list, aligned=False):
        if not aligned and self.det_net is None:
            raise ArgumentError('Detection network is None!')

        input_img = []
        for filename in img_list:
            if aligned:
                img = cv2.imread(filename)
                if img.shape[0]!= self.render_size:
                    img = cv2.resize(img, (self.render_size,self.render_size), cv2.INTER_AREA)
                if img.shape[2]==4:
                    img = img[...,:3]
            else:
                img = Image.open(filename)
                lm = self.estimate_five_landmarks(img)
                img, _ = Preprocess(img, lm, self.lm3D, render_size=self.render_size)
                img = img[0].copy()
            
            img = self.to_tensor(img)
            input_img.append(img.unsqueeze(0))

        input_img = torch.cat(input_img)
        if self.is_cuda:
            input_img = input_img.type(torch.FloatTensor).cuda(self.cuda_id)
        
        return input_img


    def reconstruct2obj(self, img_list, save_path):
        input_imgs = []
        for filename in img_list:
            img = Image.open(filename)
            lm = self.estimate_five_landmarks(img)
            img, _ = Preprocess(img, lm, self.lm3D, render_size=self.render_size)
            img = img[0].copy()
            
            img = self.to_tensor(img)
            input_imgs.append(img.unsqueeze(0))
            
        input_imgs = torch.cat(input_imgs)
        if self.is_cuda:
            input_imgs = input_imgs.type(torch.FloatTensor).cuda(self.cuda_id)
        
        coef = self.regress_3dmm(input_imgs)

        # reconstruct 3D face with output coefficients and face model
        face_shape,_,face_color,tri,_,_,_ = Reconstruction(coef, self.face_model)
        
        for i, filename in enumerate(img_list):
            shape = face_shape.cpu()[i]
            color = face_color.cpu()[i]
            save_obj(os.path.join(save_path,os.path.basename(filename)[:-4]+'_mesh.obj'),shape,tri+1,np.clip(color,0,1))

    
    def render_and_estimate_landmarks(self, coef):
        face_shape, _, face_color, _,face_projection,_,_ = Reconstruction(coef,self.face_model)
        verts_rgb = face_color[...,[2,1,0]]
        mesh = Meshes(verts=face_shape, faces=self.tri[:face_shape.shape[0],...], textures=Textures(verts_rgb=verts_rgb))

        rendered = self.phong_renderer(meshes_world=mesh, R=self.R, T=self.T)
        rendered = torch.clamp(rendered, 0.0, 1.0)
        landmarks_2d = torch.zeros_like(face_projection).cuda(self.cuda_id)
        landmarks_2d[...,0] = torch.clamp(face_projection[...,0].clone(), 0, self.render_size-1)
        landmarks_2d[...,1] = torch.clamp(face_projection[...,1].clone(), 0, self.render_size-1)
        landmarks_2d[...,1] = self.render_size - landmarks_2d[...,1].clone() - 1
        landmarks_2d = landmarks_2d[:,self.face_model.keypoints,:]
        return rendered, landmarks_2d