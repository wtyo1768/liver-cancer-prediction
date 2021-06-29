import torch
from torch import nn
import pandas as pd 
from torch.utils.data import DataLoader, Dataset
import sys
import cv2
import numpy as np
from ssl_byol import v
from evaluate import normalize
import os 
sys.path.append('../liver-canser-prediction')
from src.preprocess.dataset import padding_and_resize




# normalize -> Aug -> pad -> resize
def img_pipe(f):
    assert(os.path.isfile(f))
    img = cv2.imread(f)
    # img = padding_and_resize(img)
    img = np.moveaxis(img, -1, 0)
    img = img / 255.0

    img = torch.tensor(img, dtype=torch.float32)
    return img

class ImagesDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.paths = []
        df = pd.read_csv('/home/rockyo/liver-canser-prediction/data/feature_path.csv')
        imgs = {'T1 HB':[], 'T2':[], 'out':[],}

        for row in df.iterrows():
            f=row[1]['filename'].replace('dy','rockyo').replace('chime-pj', 'liver-canser-prediction')
            for mri_type in imgs.keys():
                fname = f.replace('mri_type', mri_type)+'.jpg'
                img = img_pipe(fname)
                img = normalize(v[mri_type])(img)
                imgs[mri_type].append(img)

        img_tensor={}
        for k in imgs.keys():
            img_tensor[k] = torch.stack(imgs[k])
        self.imgs = img_tensor
        self.label = torch.tensor(df['class_one_hot'].values.tolist())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        item = {
            **{ k:v[idx] for k, v in self.imgs.items()},
            'label': self.label[idx],
        }
        return item


if __name__ == '__main__':
    print(ImagesDataset().__getitem__(1)['T2'])