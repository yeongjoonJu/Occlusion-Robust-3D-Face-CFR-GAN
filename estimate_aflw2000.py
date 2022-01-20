import glob
import numpy as np
import torch
from scipy.io import loadmat
from mmRegressor.preprocess_img import Preprocess
from mmRegressor.load_data import BFM
from mmRegressor.network.resnet50_task import *
from PIL import Image
import cv2
from renderer import Estimator3D
import argparse


def get_5_points(landmarks):
    left_eye = [(landmarks[36][0]+landmarks[39][0])//2, (landmarks[36][1]+landmarks[39][1])//2]
    right_eye = [(landmarks[42][0]+landmarks[45][0])//2, (landmarks[42][1]+landmarks[45][1])//2]
    return [left_eye, right_eye, [landmarks[30][0],landmarks[30][1]], [landmarks[48][0],landmarks[48][1]], [landmarks[54][0],landmarks[54][1]]]

def align(img_list):
    face_model = BFM('mmRegressor/BFM/BFM_model_80.mat')
    lm3D = face_model.load_lm3d("mmRegressor/BFM/similarity_Lm3D_all.mat")

    boxes = None
    for img_name in img_list:
        mat = loadmat(img_name[:-3]+'mat')
        lmk = mat['pt3d_68'].transpose()
        lmk5 = np.array(get_5_points(lmk))

        _, _, box = Preprocess(Image.open(img_name), lmk5, lm3D, render_size=224, box=True)
        
        box = np.expand_dims(np.array(box), axis=0)
        if boxes is None:
            boxes = box
        else:
            boxes = np.concatenate((boxes, box), axis=0)

    print(boxes.shape)
    np.save('evaluation/aligned_params.npy', boxes)


if __name__=='__main__':
    ## First, execute align function for original AFLW2000!
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, help='path of !aligned! AFLW2000')
    parser.add_argument('--checkpoint', default=None, type=str, help='path of a checkpoint to evaluate')
    parser.add_argument('--save_path', type=str, default='evaluation/AFLW2000_pts68.npy')
    parser.add_argument('--gpu_id', default=0, type=int)
    args = parser.parse_args()

    model = Estimator3D(is_cuda=True, batch_size=1, model_path=args.checkpoint, test=True, back_white=False, cuda_id=args.gpu_id)
    img_list = glob.glob(args.img_path+'/*.jpg')
    img_list = sorted(img_list)

    aligned_params = np.load('evaluation/aligned_params.npy', allow_pickle=True)

    estimated_lmks = None
    n_imgs = len(img_list)
    for i, img_name in enumerate(img_list):
        print('%d / %d' % (i+1, n_imgs))
        img_ori = cv2.imread(img_name)
        img = torch.from_numpy(img_ori).permute(2,0,1).unsqueeze(0)

        if args.checkpoint is not None:
            # Our model
            img = img / 255.0
        else:
            # Previous model
            img = img / 1.0
        img = img.cuda(args.gpu_id)

        coef = model.regress_3dmm(img)
        _, lmk = model.render_and_estimate_landmarks(coef)

        lmk = lmk.cpu().numpy()[0]
        box = aligned_params[i]
        lmk[:,0] *= (box[1]-box[0]) / 224
        lmk[:,1] *= (box[3]-box[2]) / 224
        lmk[:,0] += (box[0] + box[5])
        lmk[:,1] += (box[2] + box[6])

        lmk = np.expand_dims(lmk, 0)
        if estimated_lmks is None:
            estimated_lmks = lmk
        else:
            estimated_lmks = np.concatenate((estimated_lmks, lmk), axis=0)

    print(estimated_lmks.shape)
    np.save(args.save_path, estimated_lmks)