#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os ; import sys
import glob
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

import torch.nn.functional as F
import math
import torch
import numpy as np
import torch, pickle

def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if 'fc' in name:
            continue
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def erosion(image, filter_size=5):
    """
    Args : Tensor N x H x W or N x C x H x W
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(1)
        
    pad_total = filter_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    image = F.pad(image, (pad_beg, pad_end, pad_beg, pad_end))
    kernel = torch.zeros(1, 1, filter_size, filter_size).to(image.device)
    image = F.unfold(image, filter_size, dilation=1, padding=0, stride=1)
    image = image.unsqueeze(1)
    L = image.size(-1)
    L_sqrt = int(math.sqrt(L))

    kernel = kernel.view(1, -1)
    kernel = kernel.unsqueeze(0).unsqueeze(-1)
    image = kernel - image
    image, _ = torch.max(image, dim=2, keepdim=False)
    image = -1 * image
    image = image.view(-1, 1, L_sqrt, L_sqrt)

    return image