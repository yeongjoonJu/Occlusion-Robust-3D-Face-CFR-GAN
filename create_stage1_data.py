import cv2
import os, glob, argparse
import numpy as np
from tqdm import tqdm
import random

def generate_hand_occluded_images(args):
    lmk_list = glob.glob(args.lmk_path + '/*')

    ori_hand = np.load('data/hand.npy')
    tr_range = int(args.img_size*0.09)

    for i in tqdm(range(len(lmk_list))):
        filename = os.path.basename(lmk_list[i])
        lmk = np.load(lmk_list[i])

        filename = filename[:-3]+'jpg'
        img = cv2.imread(os.path.join(args.img_path, filename))

        # The number of hands
        freq = random.randint(1,5)
        if freq < 5:
            freq = 1
        else:
            freq = 2

        for k in range(freq):
            # Transform hand
            hand = ori_hand.copy()
            if k!=0:
                w = int(args.img_size*0.25)
                h = w
            else:
                w = random.randint(int(args.img_size*0.25), int(args.img_size*0.67))
                h = random.randint(int(args.img_size*0.25), int(args.img_size*0.67))
            l = random.randint(0, 67)
            ang = random.randint(-90, 90)
            tx = random.randint(-tr_range, tr_range)
            ty = random.randint(-tr_range, tr_range)

            hand = cv2.resize(hand, (w, h))
            M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1.0)
            hand = cv2.warpAffine(hand, M, (w, h))

            l = lmk[l]
            l = l.astype('uint8')
            color = np.mean(img[l[1]-20:l[1]+20,l[0]-20:l[0]+20], axis=(0,1))

            l[0] += tx
            l[1] += ty

            _hand = np.zeros((args.img_size,args.img_size))
            top = l[1]-h//2
            bottom = l[1]+h//2
            left = l[0]-w//2
            right = l[0]+w//2
            
            if top < 0:
                top = 0
            if bottom > (args.img_size-1):
                bottom = args.img_size - 1
            if left < 0:
                left = 0
            if right > (args.img_size-1):
                right = args.img_size

            try:
                _hand[top:bottom,left:right] = hand[:bottom-top,:right-left]
                img[_hand==1.0] = color
            except Exception as e:
                print(e)

        
        cv2.imwrite(args.save_path+'/'+filename, img)


if __name__=='__main__':
    parser = argparse.ArgumentParser('Generate hand-occluded facial image')
    parser.add_argument('--img_path', type=str, required=True, help='Path for images to augment')
    parser.add_argument('--lmk_path', type=str, required=True, help='Path for facial landmarks corresponding the images')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save')
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()
    
    generate_hand_occluded_images(args)