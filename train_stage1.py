#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os, sys, random, warnings
import torch
import argparse
from renderer import Estimator3D
from datasets import LP_Dataset, FirstStageDataset
from logger import TrainStage1Logger

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.utils import make_grid
import torch.nn.functional as F
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel

from tqdm import tqdm
from faceParsing.model import BiSeNet
from face_backbone import IR_SE_50


logger = TrainStage1Logger('./logs_stage1')

def train(models, criterions, optimizer, scheduler, train_loader, val_loader, epoch, args):
    landmark_weight = torch.cat([torch.ones((1,28)),20*torch.ones((1,3)),torch.ones((1,6)),torch.ones((1,12))*5, torch.ones((1,11)), 20*torch.ones((1,8))], dim = 1).cuda(args.gpu)

    mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda(args.gpu).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda(args.gpu).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mean_f = torch.FloatTensor([0.5, 0.5, 0.5]).cuda(args.gpu).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std_f = torch.FloatTensor([0.5, 0.5, 0.5]).cuda(args.gpu).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    for i, (occluded, img, lmk, flag) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        
        # Configure model input
        occluded = Variable(occluded.type(torch.cuda.FloatTensor), requires_grad=False).cuda(args.gpu)
        img = Variable(img.type(torch.cuda.FloatTensor), requires_grad=False).cuda(args.gpu)
        lmk = Variable(lmk.type(torch.cuda.FloatTensor), requires_grad=False).cuda(args.gpu)
        flag = Variable(flag.type(torch.cuda.FloatTensor), requires_grad=False).cuda(args.gpu)

        coef = models['3D'].regress_3dmm(occluded[:,[2,1,0],...])
        rendered, landmark, reg_loss, gamma_loss = models['3D'].reconstruct(coef)
        rendered = rendered.permute(0,3,1,2).contiguous()[:,[2,1,0],:,:]

        # pose_loss = criterions['L1'](angles, ang)
        align_loss = torch.sum(torch.sum(torch.square(landmark-lmk), dim=2)*flag*landmark_weight, dim=1) / 68.0
        align_loss = torch.sum(align_loss) / lmk.shape[0]

        # coefficients regularization 1.7e-3
        coef_loss = torch.norm(coef[...,:80]) + 0.1*torch.norm(coef[...,80:144]) + 1.7e-3*torch.norm(coef[...,144:224])
            
        # For skin
        parsing_input = F.interpolate((img-mean)/std, (512,512))
        parsed = models['Seg'](parsing_input)
        parsed = F.interpolate(parsed, (224,224))
        parsed = torch.argmax(parsed, dim=1, keepdim=True)

        mask = torch.zeros_like(parsed, dtype=torch.float32).cuda(args.gpu)
        # skin 1, nose 2, eye_glass 3, r_eye 4, l_eye 5, r_brow 6, l_brow 7, r_ear 8, l_ear 9,
        # inner_mouth 10, u_lip 11, l_lip 12, hair 13
        indices = ((parsed>=1).type(torch.BoolTensor) & (parsed<=7).type(torch.BoolTensor) & (parsed!=3).type(torch.BoolTensor)) \
                | ((parsed>=11).type(torch.BoolTensor) & (parsed<=12).type(torch.BoolTensor))
        mask[indices] = 1.0

        # Get vector mask
        rendered_noise = torch.mean(rendered, dim=1, keepdim=True) > 0.0
        vector = torch.zeros_like(rendered_noise, dtype=img.dtype).cuda(args.gpu)
        vector[rendered_noise] = 1.0

        # Synthesize background
        rendered = img*(1.-vector) + rendered*vector

        # Perceptual loss
        affined_r = F.interpolate(rendered[:,:,15:-40,15:-15], (112,112), mode='bilinear', align_corners=True)
        affined_i = F.interpolate(img[:,:,15:-40,15:-15], (112,112), mode='bilinear', align_corners=True)
        emb_r = models['face']((affined_r-mean_f)/std_f)
        emb_i = models['face']((affined_i-mean_f)/std_f)
        id_loss = torch.mean(1. - criterions['Cos'](emb_r, emb_i))

        # Reconstruction loss
        rec_loss = torch.sum(torch.abs(img - rendered), dim=1)*mask
        rec_loss = torch.sum(rec_loss) / torch.sum(mask)   

        total_loss = coef_loss*1e-4 + rec_loss*0.01 + reg_loss*0.25 + align_loss*0.007 + gamma_loss*10.0 + id_loss*0.15
        total_loss.backward()
        optimizer.step()
        
        # logging
        if torch.distributed.get_rank() == 0:
            scheduler.step()
            total_iteration = len(train_loader) * epoch + i
            logger.log_training(coef_loss.item(), rec_loss.item(), reg_loss.item(), align_loss.item(), id_loss.item(), total_iteration)
            if total_iteration % 250 == 0:
                rendered_grid = make_grid(rendered, nrow=args.batch_size//2, normalize=True)
                lmk = lmk.type(torch.LongTensor)
                landmark = landmark.type(torch.LongTensor)
                
                color1 = torch.FloatTensor([1.0,0.0,0.0]).unsqueeze(-1).unsqueeze(-1)
                color2 = torch.FloatTensor([0.0,0.0,1.0]).unsqueeze(-1).unsqueeze(-1)
                
                for b in range(img.size(0)):
                    for l in range(68):
                        occluded[b, :, lmk[b,l,1]-2:lmk[b,l,1]+2, lmk[b,l,0]-2:lmk[b,l,0]+2] = color1
                        occluded[b, :, landmark[b,l,1]-2:landmark[b,l,1]+2, landmark[b,l,0]-2:landmark[b,l,0]+2] = color2
                
                input_grid = make_grid(occluded, nrow=args.batch_size//2, normalize=False)
                
                logger.log_train_image(input_grid, rendered_grid, total_iteration)
            
            sys.stdout.write('\r[Epoch %d/%d][Iter %d/%d][Total_iter %d]' % (epoch, args.epochs, i, len(train_loader), total_iteration))

            if i!=0 and total_iteration % args.val_iters == 0:
                error = validate(models, val_loader, epoch, args)
                logger.log_validation(error, epoch)
                torch.save(models['3D'].regressor.module.state_dict(), args.save_path+"/reg_it%d_%.4f_stage1.pth" % (total_iteration, error))
        

def validate(models, val_loader, epoch, args):
    with torch.no_grad():
        align_error = 0.0
        
        for i, (occluded, lmk) in enumerate(val_loader):
            print('\rval %d...' % (i+1), end='')
            occluded = Variable(occluded.type(torch.cuda.FloatTensor)).cuda(args.gpu)
            lmk = Variable(lmk.type(torch.cuda.FloatTensor)).cuda(args.gpu)
            coef = models['3D'].regress_3dmm(occluded[:,[2,1,0],...])
            _, landmark = models['3D'].reconstruct(coef, test=True)
            align_error += torch.mean(torch.abs(landmark - lmk))

        align_error /= len(val_loader)

    return align_error


def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print('rank', args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)

    # Load models
    estimator3d = Estimator3D(is_cuda=True, batch_size=args.batch_size, model_path=args.checkpoint, test=False, back_white=False, cuda_id=args.gpu)
    # estimator3d.regressor.cuda(args.gpu)
    parsing_net = BiSeNet(n_classes=19)
    parsing_net.cuda(args.gpu)
    parsing_net.load_state_dict(torch.load('faceParsing/model_final_diss.pth', map_location='cuda:'+str(args.gpu)))
    parsing_net.eval()
    face_encoder = IR_SE_50([112,112])
    face_encoder.load_state_dict(torch.load('saved_models/face_res_50.pth', map_location='cuda:'+str(args.gpu)))
    face_encoder.cuda(args.gpu)
    face_encoder.eval()
    print('All models were loaded')

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int(args.workers / ngpus_per_node)

    if False:
        estimator3d.regressor = DDP(estimator3d.regressor, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        parsing_net = DDP(parsing_net, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        face_encoder = DDP(face_encoder, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
    
    models = {}
    models['3D'] = estimator3d
    models['Seg'] = parsing_net
    models['face'] = face_encoder

    # Losses
    criterions = {}
    criterions['L2'] = torch.nn.MSELoss().cuda(args.gpu)
    criterions['L1'] = torch.nn.L1Loss().cuda(args.gpu)
    criterions['Cos'] = torch.nn.CosineSimilarity().cuda(args.gpu)

    cudnn.benchmark = True

    dataset = FirstStageDataset(occ_path=args.train_data_path + '/occluded', \
                                img_path=args.train_data_path + '/ori_img', \
                                lmk_path=args.train_data_path + '/landmarks')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        dataset, batch_size = args.batch_size,
        shuffle = (train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )

    val_dataset = LP_Dataset(args.val_data_path+'/occluded', args.val_data_path+'/landmarks')
    val_loader = DataLoader(
        val_dataset, batch_size = args.batch_size, shuffle = False,
        drop_last=True, num_workers=args.workers, pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(estimator3d.regressor.parameters(), lr=args.lr, betas=(0.5,0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*(len(train_loader)))

    print(len(train_loader))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(models, criterions, optimizer, scheduler, train_loader, val_loader, epoch, args)
        
        if torch.distributed.get_rank() == 0:
            error = validate(models, val_loader, epoch, args)
            logger.log_validation(error, epoch)
            torch.save(estimator3d.regressor.module.state_dict(), args.save_path+"/reg_ep%d_%.4f_stage1.pth" % (epoch+1, error))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Occlusion robust 3D face reconstruction')
    parser.add_argument('--train_data_path', required=True, help='path containing training data folders')
    parser.add_argument('--val_data_path', required=True, type=str, help='path containing validation data folders')
    parser.add_argument('--flag', default=None, type=str, help='flag prepended to filenames')
    parser.add_argument('--save_path', default='saved_models', help='path to save checkpoints')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to resume checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--val_iters', default=5000, type=int, metavar='N')
                        
    parser.add_argument('-b', '--batch-size', default=80, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--world-size', default=2, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')


    args = parser.parse_args()

    main(args)