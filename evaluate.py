import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../liver-canser-prediction')
from src.preprocess.dataset import padding_and_resize
import torch
from torchvision import transforms as T
from torch import nn
import os
import random
from cfg import image_size, v, channel


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


# normalize -> Aug -> pad -> resize
aug = [
    RandomApply(
        T.ColorJitter(0.8, 0.8, 0.8, 0.4),
        p = 0.3
    ),
    # T.RandomGrayscale(p=0.2),
    # RandomApply(
    #     T.GaussianBlur((3, 3), (1.0, 2.0)),
    #     p = 0.2
    # ),
    # T.RandomResizedCrop((image_size, image_size)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
]
preprocess = lambda ele:[     
    T.Normalize(mean=torch.tensor(ele[0]),std=torch.tensor(ele[1])),
    ImgPad(image_size),  
    T.Resize((image_size, image_size)),
]
data_pipe = {
    'train': lambda stat:T.Compose(aug + preprocess(stat)),
    'eval' : lambda stat:T.Compose(preprocess(stat)),
}

def img_pipe(fname, mri_type, mode="train"):
    assert(os.path.isfile(fname))

    img = cv2.imread(fname)
    img = np.moveaxis(img, -1, 0)
    img = img / 255.0
    # print(img.shape)
    img = torch.tensor(img, dtype=torch.float32)
    img = data_pipe[mode](v[mri_type])(img)
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
        gray = -1 if channel==3 else 0
        img = img_pipe(fname, mri_type)
        imgs.append(img)
        # label = np.eye(2)[label]
        y.append(label)
    imgs = torch.stack(imgs)
    if channel==1: imgs = imgs.unsqueeze(1)
    with torch.no_grad():
        projection, embedding = model(imgs, return_embedding=True)
    # print(projection)
    # print(y.shape)
    return projection.detach().numpy(), np.array(y)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler


def linear_evaluation(features_train, y_train, seed=0):
    # features_train = StandardScaler().fit_transform(features_train)

    clf = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
    )
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    scores = cross_validate(clf, features_train, y_train, cv=5, scoring=scoring, return_train_score=True )
    # print(scores.keys())
    # print('train acc:', np.mean(np.array(scores['train_accuracy'])))

    print('acc :', np.mean(np.array(scores['test_accuracy'])))
    print('precision:', np.mean(np.array(scores['test_precision'])))
    print('recall:', np.mean(np.array(scores['test_recall'])))
    print('f1:', np.mean(np.array(scores['test_f1'])))

    if np.mean(np.array(scores['test_accuracy'])):
        return np.mean(np.array(scores['test_accuracy']))
    


if  __name__ == '__main__':
    get_features(3)
