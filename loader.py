import pandas as pd 
from torch.utils.data import DataLoader, Dataset
import sys

from cfg import data_path
df = pd.read_csv(data_path)


class ImagesDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.paths = []
        for row in df.iterrows():
            self.paths.append(
        # You can revise this line to fit ur path settings
                row[1]['filename'].replace('dy','rockyo').replace('chime-pj', 'byol')
            )
        print(f'{len(self.paths)} images found')


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return path

if __name__=='__main__':
    print(ImagesDataset().__getitem__(0))