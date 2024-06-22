import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import cv2 as cv
import py360convert
import numpy as np
import torch


class StreetviewSatDataset(Dataset):
    def __init__(self, ds_path, train=True, maxn=-1, start=-1):
        super(StreetviewSatDataset, self).__init__()
        self.ds = []
        
        if train:
            path = 'CVPR_subset/splits/train-19zl.csv'
        else:
            path = 'CVPR_subset/splits/val-19zl.csv'
        
        path = f"{ds_path}/{path}"
        df = pd.read_csv(path, delimiter=',')
        df = df.reset_index()
        
        with tqdm(total=df.shape[0] if maxn==-1 else maxn) as pbar:
            for idx, row in df.iterrows():
                if idx < start:
                    continue
                
                map_img = cv.imread(f"{ds_path}/CVPR_subset/{row['map']}", cv.IMREAD_COLOR)
                map_img = cv.resize(map_img, (512, 512))
                map_img = cv.cvtColor(map_img, cv.COLOR_BGR2RGB)
                
                street_map = cv.imread(f"{ds_path}/CVPR_subset/{row['streetview']}", cv.IMREAD_COLOR)
                street_map = cv.cvtColor(street_map, cv.COLOR_BGR2RGB)
                
                # street_map = cv.resize(street_map, dsize=(512, 512))
                street_map = torch.Tensor(street_map).permute(2, 0, 1) / 127.5
                street_map = street_map - 1
                
                map_img = torch.Tensor(map_img).permute(2, 0, 1) / 127.5
                map_img = map_img - 1
                
                semantic = cv.imread(f"{ds_path}/CVPR_subset/{row['annotations']}", cv.IMREAD_GRAYSCALE)
                semantic = torch.Tensor(semantic).unsqueeze(0)
                
                entry = {
                    'map': map_img,
                    'street': street_map,
                    'semantic': semantic
                }
                
                self.ds.append(entry)
                
                maxn = maxn - 1
                pbar.update()
                
                if maxn == 0:
                    break
    
    def __getitem__(self, index):
        m = self.ds[index]['map']
        s = self.ds[index]['street'] 
        semantic = self.ds[index]['semantic'] 
        return m, s, semantic
    
    def __len__(self):
        return len(self.ds)
        