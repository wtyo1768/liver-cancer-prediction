import pandas as pd 
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('../liver-canser-prediction')

df = pd.read_csv('/home/rockyo/liver-canser-prediction/data/feature_path.csv')


class ImagesDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.paths = []

        for row in df.iterrows():
            self.paths.append(
                row[1]['filename'].replace('dy','rockyo').replace('chime-pj', 'liver-canser-prediction')
            )
        print(f'{len(self.paths)} images found')


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return path