import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../liver-canser-prediction')
from src.preprocess.dataset import padding_and_resize
import torch


df = pd.read_csv('/home/rockyo/liver-canser-prediction/data/feature_path.csv')
features = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_features(model, mri_type='T1 HB'):

    imgs = []
    y = []

    for i in range(df.shape[0]):
        fname = df.iloc[i].loc['filename']
        fname = fname.replace('dy','rockyo').replace('chime-pj', 'liver-canser-prediction')
        fname = fname.replace('mri_type', mri_type) + '.jpg'
        label = df.iloc[i].loc['class_one_hot']
        
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        img = padding_and_resize(img)
        img = np.moveaxis(img, -1, 0)
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        # print(img.shape)
        # projection, embedding = model(img, return_embedding=True)
        # print(projection.shape)
        
        if len(imgs) == 0:
            imgs = img
        else:
            imgs = np.vstack([imgs, img])
        # label = np.eye(2)[label]
        y.append(label)
        # print(imgs.shape)
    # print(imgs.shape)
    projection, embedding = model(torch.tensor(imgs, dtype=torch.float32), return_embedding=True)
    # print(projection.shape, embedding.shape)
    # print(y.shape)
    return projection.detach().numpy(), np.array(y)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate

def linear_evaluation(features_train, y_train):

    clf = LogisticRegression(solver='liblinear', class_weight='balanced',
                            max_iter=1000, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    scores = cross_validate(clf, features_train, y_train, cv=5, scoring=scoring )
    print('evaluating...')
    print('result :', scores, 
            '\nacc :', np.mean(np.array(scores['test_accuracy'])))
    print('precision:', np.mean(np.array(scores['test_precision'])))
    print('recall:', np.mean(np.array(scores['test_recall'])))
    print('f1:', np.mean(np.array(scores['test_f1'])))


if  __name__ == '__main__':
    get_features(3)
