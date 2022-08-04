import torch
import pandas as pd 
from torch.utils.data import DataLoader, Dataset
import numpy as np
from evaluate import img_pipe
import argparse
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, KFold
from pytorch_lightning.callbacks import ModelCheckpoint
from model import cls
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from cfg import data_path


class ImagesDataset(Dataset):
    def __init__(self, df, mode):
        super().__init__()
        # df = pd.read_csv('/home/rockyo/liver-canser-prediction/data/feature_path.csv')
        imgs = {
            'T1 HB': [], 
            'T2'   : [], 
            'out'  : [],
        }

        for row in df.iterrows():
            f=row[1]['filename'].replace('dy','rockyo').replace('chime-pj', 'byol')
            for mri_type in imgs.keys():
                fname = f.replace('mri_type', mri_type)+'.jpg'
                img = img_pipe(fname, mri_type, mode=mode)
                imgs[mri_type].append(img)

        img_tensor={}
        for k in imgs.keys():
            img_tensor[k] = torch.stack(imgs[k])
        self.imgs = img_tensor
        self.label = torch.tensor(df['class_one_hot'].values.tolist())
        self.class_weight  = np.unique(df['class_one_hot'].values.tolist(), return_counts=True)[1]
        print(self.class_weight)
        self.class_weight = 1 / (self.class_weight / df.shape[0]*2)
        # print('class weight:', self.class_weight)
        # print(f'{df.shape[0]} images found')

    def __len__(self):
        return len(self.imgs['T2'])

    def __getitem__(self, idx):
        item = {
            **{ k.replace(' ', '_'):v[idx] for k, v in self.imgs.items()},
            'label': self.label[idx],   
        }
        return item


parser = argparse.ArgumentParser()
parser.add_argument('--BATCH_SIZE', type=int, default=32)
parser.add_argument('--LR', type=float, default=4e-5)
parser.add_argument('--i', type=int, default=0)
parser.add_argument('--cls', type=str,)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--fold', type=int, default=-1)


parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

df = pd.read_csv(data_path)

seed = np.random.randint(66) if args.seed==-1 else args.seed
metric = []
K = 5
for i, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=K, random_state=seed, shuffle=True).split(df, df['class_one_hot'])):
    
    if not args.fold==-1:
        if not i==args.fold:
            continue
    print(f'-----------fold{i}-------------')
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_ds = ImagesDataset(train_df, 'train')
    val_ds = ImagesDataset(val_df, 'eval')

    checkpoint = ModelCheckpoint(
        monitor='val_acc', mode='max', 
        save_top_k=1, filename='{epoch}-{val_acc:.3f}-{f1:.3f}'#, verbose=True
    )
    es = EarlyStopping(
        monitor='val_acc', min_delta=0.00,
        patience=3, verbose=False, mode='max'
    )
    trainer = pl.Trainer.from_argparse_args(
        args,  log_every_n_steps=1500, logger=False,
        callbacks=[checkpoint,],       
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.BATCH_SIZE, 
        num_workers=16, 
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.BATCH_SIZE, 
        num_workers=16, 
        shuffle=False,
    )
    model = cls(args, class_weight=train_ds.class_weight, enc=args.cls, p=args.i==0 and i==0)
    trainer.fit(
        model, 
        train_loader, 
        val_loader,
    )
    ### Evaluation
    print('path', checkpoint.best_model_path)
    model = cls.load_from_checkpoint(checkpoint.best_model_path, class_weight=train_ds.class_weight, enc=args.cls)
    pred = trainer.validate(model, val_loader)
    metric.append(pred[0])

    if pred[0]['val_acc'] > .66:
        a, f = pred[0]['val_acc'], pred[0]['f1']
        torch.save(
            model.state_dict(), 
            f'./model/cgc_{(round(a,2))}_{round(f, 2)}.pth'
        )

if args.fold==-1:
    for k in metric[0].keys():
        if k=='val_loss':continue
        r = 0
        for e in metric: r += e[k]
        print(f'cv_{k}', ':', r/K)


print('seed:', seed)

