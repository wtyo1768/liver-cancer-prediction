import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('/home/rockyo/liver-canser-prediction')
import torch
from torchvision import transforms as T
from torch import nn
import os
import random
from cfg import image_size, v, channel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from PIL import Image



class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class ImgPad(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
    def forward(self, img):
        h, w = img.shape[-2], img.shape[-1]
        left =  (self.img_size - w) // 2
        right =  self.img_size - w - left
        top =   (self.img_size - h) //2
        bottom = self.img_size - h - top
        left, top = max(left, 0), max(top, 0) 
        right, bottom = max(right, 0), max(bottom, 0)
        im = T.Pad(padding=(left, top, right, bottom))(img)
        return im


class crop(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio
    def forward(self, img):
        ratio = random.uniform(self.ratio, 1)
        # print('crop',type(img))
        w, h = img.size
        # h, w = img.shape[-2], img.shape[-1]
        return T.RandomCrop((int(h*ratio), int(w*ratio)))(img)

# normalize -> Aug -> pad -> resize
# large img -> resize
# small img -> pad

train_aug = lambda ele:[
    RandomApply(
        T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        p=0.8
        # p = 0.5
    ),
    # T.RandomGrayscale(p=0.2),
    # RandomApply(
    #     T.GaussianBlur((3, 3), (1.0, 2.0)),
    #     p = 0.2
    # ),
    # RandomApply(crop(.75), p = 0.5),                
    T.RandomRotation(2.8),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize(mean=torch.tensor(ele[0]),std=torch.tensor(ele[1])),
    ImgPad(image_size),  
    T.Resize((image_size, image_size)),
]
test_aug = lambda ele:[     
    T.ToTensor(),
    T.Normalize(mean=torch.tensor(ele[0]),std=torch.tensor(ele[1])),
    ImgPad(image_size),  
    T.Resize((image_size, image_size)),
]

data_pipe = {
    'train': lambda stat:T.Compose(train_aug(stat)),
    'eval' : lambda stat:T.Compose(test_aug(stat)),
}
print(data_pipe['train'](v['T2']))
print(data_pipe['eval'](v['T2']))


def img_pipe(fname, mri_type, mode="train"):
    assert(os.path.isfile(fname))
    img = Image.open(fname)
    # print(img.format, img.size, img.mode)
# 
    # img = cv2.imread(fname)
    # img = np.moveaxis(img, -1, 0)
    # img = img / 255.0
    # img = torch.tensor(img, dtype=torch.float32)
    # if mode=='train':
    #     img = T.ToTensor()(img)

        # print(img[0])
    img = data_pipe[mode](v[mri_type])(img)
    # print(img.shape)
    return img 


df = pd.read_csv('/home/rockyo/liver-canser-prediction/data/feature_path.csv')
features = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_features(model, mri_type='T1 HB'):
    imgs = []
    y = []
    print(f'============{mri_type}===========')
    for i in range(df.shape[0]):
        fname = df.iloc[i].loc['filename']
        fname = fname.replace('dy','rockyo').replace('chime-pj', 'liver-canser-prediction')
        fname = fname.replace('mri_type', mri_type) + '.jpg'
        label = df.iloc[i].loc['class_one_hot']
        img = img_pipe(fname, mri_type, mode="eval")
        imgs.append(img)
        y.append(label)
    imgs = torch.stack(imgs)
    if channel==1: imgs = imgs.unsqueeze(1)
    with torch.no_grad():
        projection, embedding = model(imgs, return_embedding=True)
    # print(projection)
    # print(y.shape)
    return projection.detach().numpy(), np.array(y)


def linear_evaluation(features_train, y_train, seed=0):
    # features_train = StandardScaler().fit_transform(features_train)

    clf = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=seed
    )
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    scores = cross_validate(clf, features_train, 
        y_train, cv=5, scoring=scoring, return_train_score=True)
    # print(scores.keys())
    # print('train acc:', np.mean(np.array(scores['train_accuracy'])))

    print('acc :', np.mean(np.array(scores['test_accuracy'])))
    print('precision:', np.mean(np.array(scores['test_precision'])))
    print('recall:', np.mean(np.array(scores['test_recall'])))
    print('f1:', np.mean(np.array(scores['test_f1'])))

    return np.mean(np.array(scores['test_accuracy']))
    


if  __name__ == '__main__':
    get_features(3)
