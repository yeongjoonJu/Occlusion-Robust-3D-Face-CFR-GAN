import os, glob
import shutil
import random
from tqdm import tqdm

if __name__=='__main__':
    FROM_PATH = '../hand_occluded_224'
    TRAIN_PATH = '../hand_occluded_224/train'
    VAL_PATH = '../hand_occluded_224/val'
    filelist = glob.glob('../hand_occluded_224/ori_img/*.jpg')
    print(len(filelist))

    random.shuffle(filelist)
    train_len = int(len(filelist)*0.95)

    for filename in tqdm(filelist[:train_len]):
        name = os.path.basename(filename)
        if os.path.exists(FROM_PATH+'/landmarks/'+name[:-3]+'npy'):
            shutil.move(FROM_PATH+'/landmarks/'+name[:-3]+'npy', TRAIN_PATH+'/landmarks/'+name[:-3]+'npy')
            shutil.move(FROM_PATH+'/ori_img/'+name, TRAIN_PATH+'/ori_img/'+name)

        if os.path.exists(FROM_PATH+'/occluded/'+name):
            shutil.move(FROM_PATH+'/occluded/'+name, TRAIN_PATH+'/occluded/'+name)
    
    for filename in tqdm(filelist[train_len:]):
        name = os.path.basename(filename)
        if os.path.exists(FROM_PATH+'/landmarks/'+name[:-3]+'npy'):
            shutil.move(FROM_PATH+'/landmarks/'+name[:-3]+'npy', VAL_PATH+'/landmarks/'+name[:-3]+'npy')
            shutil.move(FROM_PATH+'/ori_img/'+name, VAL_PATH+'/ori_img/'+name)

        if os.path.exists(FROM_PATH+'/occluded/'+name):
            shutil.move(FROM_PATH+'/occluded/'+name, VAL_PATH+'/occluded/'+name)