import os
import argparse
import multiprocessing
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl
import sys
sys.path.append('../liver-canser-prediction')
from src.preprocess.dataset import padding_and_resize
import cv2
from loader import ImagesDataset
from efficientnet_pytorch import EfficientNet
import numpy as np
from torch import nn
from torchvision import transforms as T
import random
from evaluate import get_features, img_pipe, linear_evaluation
from cfg import image_size, v, channel


def random_mri():
    seed = np.random.rand(1)[0]
    if seed < 0.33: return 'T1 HB'
    elif seed >= 0.33 and  seed <= 0.67: return 'T2' 
    return 'out'


def aug_fn(fname):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgs = []
    d = -1 if channel==3 else 0
    
    for f in fname:
        mri_type = 'T1 HB'
        # mri_type = random_mri()
        f = f.replace('mri_type', mri_type)+'.jpg'
        img = img_pipe(f, mri_type)
        imgs.append(img)
    imgs = torch.stack(imgs).to(device)
    return imgs


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(
            net, 
            augment_fn=aug_fn,
            augment_fn2=aug_fn,
            **kwargs
        )
    def forward(self, images):
        # print('forward', images)
        return self.learner(images, return_embedding=False)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)
        # return torch.optim.SGD(self.parameters(), lr=LR)


    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


effnet = EfficientNet.from_pretrained(
    'efficientnet-b2',
    in_channels=channel,
)

model = SelfSupervisedLearner(
    effnet,
    image_size=image_size,
    hidden_layer='_avg_pooling',
    # hidden_layer='avgpool',
    projection_size=256,
    # projection_hidden_size=1408*2,
    projection_hidden_size=1792,
    moving_average_decay=0.99,
    use_momentum=False,
    channel=channel,
) 
#SILU
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='byol-lightning-test')
    parser.add_argument('--BATCH_SIZE', type=int, required = True,)
    parser.add_argument('--EPOCHS', type=int, required = True,)
    parser.add_argument('--LR', type=float, required = True,)
    parser.add_argument('--i', type=int, required = True,)

    args = parser.parse_args()

    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS     = args.EPOCHS
    LR         = args.LR
    NUM_GPUS   = 1
    NUM_WORKERS = multiprocessing.cpu_count()

    ds = ImagesDataset()
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    
    if args.i==0:
        print(model.learner)
    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        # accumulate_grad_batches=2,
        sync_batchnorm = False,
        # gradient_clip_val=2,
        # stochastic_weight_avg=True,
    )
    trainer.fit(model, train_loader)


    seed = np.random.randint(66)
    print('seed : ', seed)

    X, y = get_features(model.learner, 'T1 HB')
    t1 = linear_evaluation(X, y, seed)

    X, y = get_features(model.learner, 'T2')
    t2 = linear_evaluation(X, y, seed)

    X, y = get_features(model.learner, 'out')
    t3 = linear_evaluation(X, y, seed)
    

    if t1>0.60 or t2>0.60 and t3>0.60:
        torch.save(
            model.learner.state_dict(), 
            f'./model/t1_{(round(t1,2))}_{round(t2, 2)}_{round(t3, 2)}.pth'
        )
    print(X.shape, y.shape)