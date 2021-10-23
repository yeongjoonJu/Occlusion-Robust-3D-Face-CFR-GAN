#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os ; import sys 
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

from faceParsing.model import BiSeNet

from mmRegressor.network.resnet50_task import *
from mmRegressor.preprocess_img import Preprocess
from mmRegressor.load_data import *
from mmRegressor.reconstruct_mesh import Reconstruction, Compute_rotation_matrix, _need_const
import time

import torch
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
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
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes

class Estimator3D(object):
    def __init__(self, is_cuda=True, batch_size=1, render_size=224, test=True, model_path=None, back_white=False, cuda_id=0):
        self.is_cuda = is_cuda
        self.render_size = render_size
        self.cuda_id = cuda_id

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
            self.tri = self.tri.cuda()
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
        face_model = BFM('mmRegressor/BFM/BFM_model_80.mat')

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
            regressor = regressor.cuda()
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

        landmarks_2d = torch.zeros_like(face_projection).cuda()
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