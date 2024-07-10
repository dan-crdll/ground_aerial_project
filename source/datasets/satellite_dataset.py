import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import cv2 as cv
import numpy as np
import torch


class SatelliteImagesDs(Dataset):
    def __init__(self, ds_path, train=True, maxn=-1, start=-1):
        super(SatelliteImagesDs, self).__init__()
        self.ds = []
        
        path = 'CVPR_subset/splits/train-19zl.csv' if train else 'CVPR_subset/splits/val-19zl.csv'
        df = pd.read_csv(f"{ds_path}/{path}", delimiter=',').reset_index()

        with tqdm(total=len(df) if maxn == -1 else maxn) as pbar:
            for idx, row in df.iterrows():
                if idx < start:
                    continue
                
                satellite_map = cv.cvtColor(cv.imread(f"{ds_path}/CVPR_subset/{row['map']}"), cv.COLOR_BGR2RGB)
                satellite_map = cv.resize(satellite_map, (512, 512))
                
                satellite_map = (torch.tensor(satellite_map).permute(2, 0, 1) / 127.5) - 1
             
                self.ds.append(satellite_map)
                
                maxn -= 1
                pbar.update()
                
                if maxn == 0:
                    break
    
    def __getitem__(self, index):
        return self.ds[index]
    
    def __len__(self):
        return len(self.ds)
