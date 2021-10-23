import os, glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import cv2
import random, math

class FirstStageDataset(Dataset):
    def __init__(self, occ_path, img_path, lmk_path, test=False, flag=None):
        self.occ_path = occ_path
        self.img_path = img_path
        self.lmk_path = lmk_path
        self.test = test
        self.flag = flag

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.color_aug = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

        self.occ_list = glob.glob(occ_path+'/*.jpg')
        print("The number of data: {}".format(len(self.occ_list)))

    def __len__(self):
        return len(self.occ_list)

    def __rmul__(self, v):
        self.occ_list = v * self.occ_list
        return self
    
    def get_rot_mat(self, angle):
        """
        angle : radian
        """
        angle = torch.tensor(angle)
        return torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                             [torch.sin(angle), torch.cos(angle), 0]])
    
    def __getitem__(self, index):
        filename = os.path.basename(self.occ_list[index])
        occluded = Image.open(self.occ_list[index]).convert('RGB')
        img = Image.open(os.path.join(self.img_path, filename)).convert('RGB')
        lmk = np.load(os.path.join(self.lmk_path, filename[:-3]+'npy'))
        lmk = lmk[:,:2]

        # Non-occluded augmentation
        if not self.test and torch.rand(1) < 0.5:
            occluded = img.copy()

        # Flags to prevent from 3DDFAv2 error propagation
        flag = torch.ones(1)
        if self.flag is not None and filename[:len(self.flag)] == self.flag:
            flag = flag * 0.7

        # Brightness, contrast, saturation augmentation
        if torch.rand(1) < 0.5:
            color_trans = transforms.ColorJitter.get_params(
                self.color_aug.brightness, self.color_aug.contrast,
                self.color_aug.saturation, self.color_aug.hue
            )
            img = color_trans(img)
            occluded = color_trans(occluded)

        occluded = self.transform(occluded)
        img = self.transform(img)
        
        # Low-resolution augmentation
        if not self.test and torch.rand(1) < 0.25:
            # scale_factor = -0.4 * torch.rand(1) + 0.5
            occluded = F.interpolate(occluded.unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=True)
            occluded = F.interpolate(occluded, (224,224), mode='bilinear', align_corners=True).squeeze(0)

        # Rotation augmentation
        if not self.test and torch.rand(1) < 0.25:
            angle = random.random()*math.pi/2 - (math.pi / 4)
            M = self.get_rot_mat(angle)
            occluded = occluded.unsqueeze(0)
            img = img.unsqueeze(0)

            grid = F.affine_grid(M[None,...], occluded.size())
            occluded = F.grid_sample(occluded, grid)
            img = F.grid_sample(img, grid)
            
            occluded = occluded.squeeze(0)
            img = img.squeeze(0)

            ones = np.ones(shape=(68,1))
            M = cv2.getRotationMatrix2D((112,112), angle*(180/math.pi), 1.0)
            lmk = np.hstack([lmk, ones])
            lmk = M.dot(lmk.T).T

        if not self.test and torch.rand(1) < 0.25:
            sy = random.randint(0,56)
            sx = random.randint(0,56)
            h = random.randint(112, 168)
            w = random.randint(112, 168)

            occluded[:,:sy,:] = 0.0
            occluded[:,:,:sx] = 0.0
            occluded[:,sy+h:,sx+w:] = 0.0


        return occluded, img, lmk, flag


class LP_Dataset(Dataset):
    def __init__(self, img_path, lmk_path):
        self.img_path = img_path
        self.lmk_path = lmk_path
        # self.pose_path = pose_path

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.lmk_list = glob.glob(lmk_path+'/*.npy')
        # self.img_list = glob.glob(img_path+'/*.jpg')
        # self.img_list += glob.glob(img_path+'/*.png')

        print("The number of data: {}".format(len(self.lmk_list)))

    def __len__(self):
        return len(self.lmk_list)
    
    def __getitem__(self, index):
        filename = os.path.basename(self.lmk_list[index])
        lmk = np.load(self.lmk_list[index])
        if lmk.shape[1]==3:
            lmk = lmk[:,:2]
        img = Image.open(os.path.join(self.img_path, filename[:-3]+'jpg')).convert('RGB')

        lmk = torch.from_numpy(lmk)
        return self.transform(img), lmk


class MaskedFaceDataset(Dataset):
    def __init__(self, mfd_path, ori_path, p=0.5):
        self.mfd_path = mfd_path
        self.ori_path = ori_path
        # lp_path = '../hand_occluded_224/occluded'
        self.p = p

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        self.color_aug = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2/3.14)
        # self.erasing_transform_rand = transforms.RandomErasing(p=0.5, scale=(0.05,0.1), ratio=(0.1, 3.3), value='random')

        self.mfd_list = glob.glob(mfd_path+'/*.jpg')
        self.mfd_list += glob.glob(mfd_path+'/*.png')
        # self.lp_list = glob.glob(lp_path+'/*.jpg')

        print('The number of data: {}'.format(len(self.mfd_list)))
    
    def __len__(self):
        return len(self.mfd_list)
    
    def __getitem__(self, index):
        color_trans = transforms.ColorJitter.get_params(
            self.color_aug.brightness, self.color_aug.contrast,
            self.color_aug.saturation, self.color_aug.hue
        )
        
        # if torch.rand(1) < self.p:
        #     ori = Image.open(self.lp_list[index % len(self.lp_list)]).convert('RGB')
        #     ori = self.transform(ori)
        #     mf = ori.detach().clone()
        #     if torch.rand(1) < 0.25:
        #         x = random.randint(70,175)
        #         y = random.randint(70,175)
        #         w = random.randint(10, 70)
        #         h = random.randint(10, 70)
        #         random_box = -1 * torch.rand(3,1,1) + 1
        #         noise = -0.2*torch.rand(3,224,224) + 0.1
        #         random_box = (random_box.expand_as(ori) + noise)
                
        #         mf[:,y:y+h,x:x+w] = random_box[:,y:y+h,x:x+w]
        #         mf = torch.clamp(mf, 0.0, 1.0)
        name = os.path.basename(self.mfd_list[index]).split('_')[0]
        ori = Image.open(os.path.join(self.ori_path, 'f'+name+'.jpg')).convert('RGB')
        ori = color_trans(ori)
        ori = self.transform(ori)

        if torch.rand(1) < 0.5:
            mf = Image.open(self.mfd_list[index]).convert('RGB')
            mf = color_trans(mf)
            mf = self.transform(mf)
        else:
            mf = ori.detach().clone()
            

        if torch.rand(1) < self.p:
            mf = torch.flip(mf, dims=[2])
            ori = torch.flip(ori, dims=[2])

        if torch.rand(1) < 0.25:
            mf = F.interpolate(mf.unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=True)
            mf = F.interpolate(mf, scale_factor=4.0, mode='bilinear', align_corners=True).squeeze(0)

        return mf, ori
