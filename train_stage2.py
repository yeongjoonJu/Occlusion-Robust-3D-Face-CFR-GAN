import os, glob, sys, cv2
import torch
import argparse
from renderer import Estimator3D
from datasets import MaskedFaceDataset, LP_Dataset
from logger import TrainStage2Logger

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.utils import save_image, make_grid

from senet import FeatureMatchingLoss
from faceParsing.model import BiSeNet
from mmRegressor.network.resnet50_task import resnet50_use


logger = TrainStage2Logger('./logs_stage2')

def train(models, criterions, optimizer, scheduler, train_loader, epoch, args):
    landmark_weight = torch.cat([torch.ones((1,28)),5*torch.ones((1,3)),torch.ones((1,6)),torch.ones((1,12))*3, torch.ones((1,11)), 5*torch.ones((1,8))], dim = 1).cuda()
    fm_weights = [1/8, 1/4, 1/2, 1.0]

    for i, (masked, img) in enumerate(train_loader):
        # Train stage
        optimizer.zero_grad()
        
        # Configure model input
        img = Variable(img.type(torch.cuda.FloatTensor), requires_grad=False).cuda()
        masked = Variable(masked.type(torch.cuda.FloatTensor), requires_grad=False).cuda()

        feats = models['3D'].regressor(masked[:,[2,1,0],...], fm=True)
        coef = feats[-1]
        rendered, landmark = models['3D'].reconstruct_and_render(coef)
        rendered = rendered.permute(0,3,1,2).contiguous()[:,[2,1,0],:,:]
        
        feats_t = models['teacher'](img[:,[2,1,0],...], fm=True)
        coef_t = feats_t[-1]
        rendered_t, landmark_t = models['3D'].reconstruct_and_render(feats_t[-1])
        rendered_t = rendered_t.permute(0,3,1,2).contiguous()[:,[2,1,0],:,:]

        # Feature matching loss
        fm_loss = 0.0
        for k in range(4):
            fm_loss += criterions['L2'](feats[k], feats_t[k].detach()) * fm_weights[k]

        # Align loss
        align_loss = torch.mean(torch.sum(torch.abs(landmark - landmark_t.detach()), dim=2)*landmark_weight)

        # coefficients loss
        coef_loss = criterions['L2'](coef[...,:144], coef_t[...,:144]) + criterions['L2'](coef[...,144:224], coef_t[...,144:224]) \
                    + criterions['L2'](coef[...,224:], coef_t[...,224:])

        total_loss = coef_loss*0.1 + fm_loss + align_loss*0.01

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # logging
        total_iteration = len(train_loader) * epoch + i
        logger.log_training(coef_loss.item(), fm_loss.item(), align_loss.item(), total_loss.item(), total_iteration)
        if total_iteration % 250 == 0:
            landmark_t = landmark_t.type(torch.LongTensor)
            landmark = landmark.type(torch.LongTensor)
                
            color1 = torch.FloatTensor([1.0,0.0,0.0]).unsqueeze(-1).unsqueeze(-1)
            color2 = torch.FloatTensor([0.0,0.0,1.0]).unsqueeze(-1).unsqueeze(-1)
                            
            for b in range(img.size(0)):
                for l in range(68):
                    masked[b, :, landmark_t[b,l,1]-2:landmark_t[b,l,1]+2, landmark_t[b,l,0]-2:landmark_t[b,l,0]+2] = color1
                    masked[b, :, landmark[b,l,1]-2:landmark[b,l,1]+2, landmark[b,l,0]-2:landmark[b,l,0]+2] = color2

            masked_grid = make_grid(masked, nrow=args.batch_size//2, normalize=False)
            rendered_grid = make_grid(rendered, nrow=args.batch_size//2, normalize=True)
            img_grid = make_grid(img, nrow=args.batch_size//2, normalize=False)
            rendered_t_grid = make_grid(rendered_t, nrow=args.batch_size//2, normalize=True)
                
            logger.log_train_image(masked_grid, rendered_grid, img_grid, rendered_t_grid, total_iteration)

        sys.stdout.write('\r[Epoch %d/%d][Batch %d/%d]' % (epoch, args.epochs, i, len(train_loader)))
            

def validate(models, val_loader, epoch, args):
    with torch.no_grad():
        align_error = 0.0
        for j, (occluded, lmk) in enumerate(val_loader):
            print('\rval %d...' % (j+1), end='')
            occluded = Variable(occluded.type(torch.cuda.FloatTensor)).cuda()
            # img = Variable(img.type(torch.cuda.FloatTensor)).cuda()
            lmk = Variable(lmk.type(torch.cuda.FloatTensor)).cuda()
            coef = models['3D'].regress_3dmm(occluded[:,[2,1,0],...])
            _, landmark = models['3D'].reconstruct(coef, test=True)
            align_error += torch.mean(torch.abs(landmark - lmk))

        align_error /= len(val_loader)

    return align_error


def main(args):
    # Losses
    criterions = {}
    criterions['L2'] = torch.nn.MSELoss().cuda()
    criterions['L1'] = torch.nn.L1Loss().cuda()

    estimator3d = Estimator3D(is_cuda=True, batch_size=args.batch_size, model_path=args.checkpoint, test=False, back_white=False, device_id=args.cuda_id)
    estimator3d.regressor.cuda()

    models = {}
    models['3D'] = estimator3d
    models['teacher'] = resnet50_use()
    models['teacher'].load_state_dict(torch.load(args.checkpoint, map_location='cuda:'+str(args.cuda_id)))
    models['teacher'].cuda(args.cuda_id)
    models['teacher'].eval()
    
    train_dataset = MaskedFaceDataset(args.masked_data_path, args.origin_data_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers=args.workers
    )
    val_dataset = LP_Dataset(args.val_data_path+'/occluded', args.val_data_path+'/landmarks')
    val_loader = DataLoader(
        val_dataset, batch_size = args.batch_size, shuffle = False,
        drop_last=True, num_workers=args.workers, pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(estimator3d.regressor.parameters(), lr=args.lr, betas=(0.5,0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*(len(train_loader)))

    for epoch in range(args.start_epoch, args.epochs):
        train(models, criterions, optimizer, scheduler, train_loader, epoch, args)
        error = validate(models, val_loader, epoch, args)
        logger.log_validation(error, epoch)
        torch.save(estimator3d.regressor.state_dict(), args.save_path+"/reg_ep%d_%.4f_stage2.pth" % (epoch+1, error))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Occlusion robust 3D face reconstruction')
    parser.add_argument('--masked_data_path', required=True, help='path containing training data folders')
    parser.add_argument('--origin_data_path', required=True, help='path containing training data folders')
    parser.add_argument('--val_data_path', required=True, type=str, help='path containing validation data folders')
    parser.add_argument('--save_path', default='saved_models', help='path to save checkpoints')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to resume checkpoint')
    
    parser.add_argument('--start_epoch', default='0', type=int, metavar='N')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')

    parser.add_argument('--cuda_id', default=0, type=int)

    args = parser.parse_args()

    main(args)