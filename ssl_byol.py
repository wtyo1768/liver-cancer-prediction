import os
import argparse
import multiprocessing
from pathlib import Path

import torch
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


# arguments

# parser = argparse.ArgumentParser(description='byol-lightning-test')

# parser.add_argument('--image_folder', type=str, required = True,
#                        help='path to your folder of images for self-supervised learning')

# args = parser.parse_args()


BATCH_SIZE = 12
EPOCHS     = 8
LR         = 8e-4
NUM_GPUS   = 1
IMAGE_SIZE = 140
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()


effnet = EfficientNet.from_name('efficientnet-b4')

# for  i in range(0,len(effnet._blocks)):
#     try:
#         # print("...........found BN layer  so Replacing Bn0 with GN.........")
#         gn0 = effnet._blocks[i]._bn0.num_features
#         effnet.enet._blocks[i]._bn0 = nn.GroupNorm(1,num_channels = gn0)
#         # print(effnet._blocks[i]._bn0)
#     except:
#         # print("BN layer Not  found!!")
        
#     gn1 = effnet._blocks[i]._bn1.num_features
#     gn2 = effnet._blocks[i]._bn2.num_features
    
#     effnet._blocks[i]._bn1 = nn.GroupNorm(1,num_channels = gn1)
#     effnet._blocks[i]._bn2 = nn.GroupNorm(1,num_channels = gn2)
#     # print(effnet._blocks[i]._bn1)
#     # print(effnet._blocks[i]._bn2)


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(
            net, 
            augment_fn=self.aug_fn(mri_type='T1 HB'),
            augment_fn2=self.aug_fn(mri_type='T2'),
            **kwargs
        )

    def aug_fn(fname, mri_type):
        
        def fn(fname, mri_type=mri_type):
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            imgs = []
            for f in fname:

                f = f.replace('mri_type', mri_type)+'.jpg'
                assert(os.path.isfile(f))
                img = cv2.imread(f)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = padding_and_resize(img)
                img = np.moveaxis(img, -1, 0)

                imgs.append(img)
            return torch.tensor(imgs, dtype=torch.float32).to(device)

        return fn


    def forward(self, images):
        # print('forward', images)
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()




if __name__ == '__main__':
    ds = ImagesDataset()
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model = SelfSupervisedLearner(
        effnet,
        image_size = IMAGE_SIZE,
        hidden_layer = '_avg_pooling',
        projection_size = 1024,
        projection_hidden_size = 1792,
        moving_average_decay = 0.99
    )
    # print(model.learner)
    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = True
    )

    trainer.fit(model, train_loader)

    from evaluate import get_features, linear_evaluation

    print('=======T1 HB=======')
    X, y = get_features(model.learner, 'T1 HB')
    linear_evaluation(X, y)

    print('=======T2=======')
    X, y = get_features(model.learner, 'T2')
    linear_evaluation(X, y)
    # print(X.shape, y.shape)
