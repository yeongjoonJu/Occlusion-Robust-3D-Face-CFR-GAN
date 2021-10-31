#!/usr/bin/python
# -*- encoding: utf-8 -*-
import argparse, sys
sys.path.append('Pytorch_Retinaface')

from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.data import cfg_re50, cfg_mnet

from renderer import Estimator3D
import torch
from tqdm import tqdm
from glob import glob
import os, cv2

import torch.backends.cudnn as cudnn

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser('Inference')
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=None, help='Trained model path. If value is None, baseline model is used')
    parser.add_argument('--det_base', type=str, default='re50', help='RetinaFace base network (re50 or mnet)')
    parser.add_argument('--aligned', type=bool, default=False, help='If images were already aligned, True')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    det_net = None
    if not args.aligned:
        # Load detection model
        if args.det_base=='re50':
            det_cfg = cfg_re50
            det_weights = 'Pytorch_Retinaface/weights/Resnet50_Final.pth'
        elif args.det_base=='mnet':
            det_cfg = cfg_mnet
            det_weights = 'Pytorch_Retinaface/weights/mobilenet0.25_Final.pth'
        else:
            raise NotImplementedError

        det_net = RetinaFace(cfg=det_cfg, phase='test')
        det_net = load_model(det_net, det_weights, False)
        det_net.eval()
        print('Finished loading detection model!')
        cudnn.benchmark = True
        det_net = det_net.cuda()
        det_net = (det_net, det_cfg)

    # Load 3D estimator with renderer
    estimator = Estimator3D(render_size=224, batch_size=args.batch_size, model_path=args.model_path, det_net=det_net, cuda_id=args.cuda_id)

    img_list = glob(args.img_path+'/*.jpg')
    print('The number of images:', len(img_list))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Estimate, render and save
    for k in tqdm(range(0, len(img_list), args.batch_size)):
        until = k+args.batch_size
        if until  > len(img_list):
            until = len(img_list)

        input_img = estimator.align_convert2tensor(img_list[k:until], aligned=args.aligned)
        if args.model_path is None:
            input_img = input_img*255.0
        rendered, landmarks = estimator.estimate_and_reconstruct(input_img)
        rendered = (rendered[...,:3]*255.0).cpu().detach().numpy().astype('uint8')


        for i in range(rendered.shape[0]):
            cv2.imwrite(os.path.join(args.save_path, os.path.basename(img_list[k+i])), rendered[i])
