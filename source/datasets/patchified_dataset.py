# import pandas as pd
# from torch.utils.data import Dataset
# from tqdm.auto import tqdm
# import cv2 as cv
# import py360convert
# import numpy as np
# import torch


# class PatchifiedDataset(Dataset):
#     def __init__(self, ds_path, train=True, maxn=-1, start=-1, patch_size=16):
#         super(PatchifiedDataset, self).__init__()
#         self.ds = []
        
#         if train:
#             path = 'CVPR_subset/splits/train-19zl.csv'
#         else:
#             path = 'CVPR_subset/splits/val-19zl.csv'
        
#         path = f"{ds_path}/{path}"
#         df = pd.read_csv(path, delimiter=',')
#         df = df.reset_index()
        
#         with tqdm(total=df.shape[0] if maxn==-1 else maxn) as pbar:
#             for idx, row in df.iterrows():
#                 if idx < start:
#                     continue
                
#                 street_map = cv.imread(f"{ds_path}/CVPR_subset/{row['streetview']}", cv.IMREAD_COLOR)
#                 street_map = cv.cvtColor(street_map, cv.COLOR_BGR2RGB)
                
#                 street_map = torch.Tensor(street_map).permute(2, 0, 1) / 127.5
#                 street_map = street_map - 1
            
                
#                 semantic = cv.imread(f"{ds_path}/CVPR_subset/{row['annotations']}", cv.IMREAD_GRAYSCALE)
                
#                 positions = np.zeros_like(semantic)

#                 for i in range(positions.shape[0]):
#                     for j in range(positions.shape[1]):
#                         pos = (i + 1) * j
#                         positions[i, j] = np.sin(pos) + np.cos(pos)
#                 positions = torch.Tensor(positions).unsqueeze(0)
                
#                 street_map = torch.cat([street_map, positions], 0)
                                
#                 patches = []
#                 C, H, W = street_map.shape
#                 recon_mask = torch.ones((H // patch_size, W // patch_size))

#                 for i in range(H // patch_size):
#                     for j in range(W // patch_size):
#                         semantic_patch = semantic[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
                        
#                         if (semantic_patch == 0).sum() / (patch_size ** 2) < 0.3:
#                             patches.append(street_map[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size].unsqueeze(0))
#                         else:
#                             recon_mask[i, j] = 0
                
#                 patches = torch.stack(patches, dim=0)
                
#                 self.ds.append(
#                     {
#                         'patches': patches,
#                         'recon_mask': recon_mask
#                     }
#                 )
                
#                 maxn = maxn - 1
#                 pbar.update()
                
#                 if maxn == 0:
#                     break
    
#     def __getitem__(self, index):
#         s = self.ds[index]['patches']
#         m = self.ds[index]['recon_mask'] 
#         return s, m
    
#     def __len__(self):
#         return len(self.ds)
        
        
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import cv2 as cv
import numpy as np
import torch


class PatchifiedDataset(Dataset):
    def __init__(self, ds_path, train=True, maxn=-1, start=-1, patch_size=16, out='streetview'):
        super(PatchifiedDataset, self).__init__()
        self.ds = []
        
        path = 'CVPR_subset/splits/train-19zl.csv' if train else 'CVPR_subset/splits/val-19zl.csv'
        df = pd.read_csv(f"{ds_path}/{path}", delimiter=',').reset_index()

        with tqdm(total=len(df) if maxn == -1 else maxn) as pbar:
            for idx, row in df.iterrows():
                if idx < start:
                    continue
                
                street_map = cv.cvtColor(cv.imread(f"{ds_path}/CVPR_subset/{row[out]}"), cv.COLOR_BGR2RGB)
                street_map = (torch.tensor(street_map).permute(2, 0, 1) / 127.5) - 1
                
                if out == 'streetview':
                    semantic = cv.imread(f"{ds_path}/CVPR_subset/{row['annotations']}", cv.IMREAD_GRAYSCALE)
                else:
                    semantic = np.zeros((street_map.shape[1], street_map.shape[2]))
                
                positions = torch.tensor(np.sin(np.arange(semantic.size) + 1).reshape(semantic.shape) + np.cos(np.arange(semantic.size) + 1).reshape(semantic.shape)).unsqueeze(0)
                
                street_map = torch.cat([street_map, positions], 0)
                
                C, H, W = street_map.shape
                recon_mask = torch.ones(H // patch_size, W // patch_size)
                
                patches = []
                for i in range(H // patch_size):
                    for j in range(W // patch_size):
                        semantic_patch = semantic[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                        
                        if (semantic_patch == 0).sum() / (patch_size ** 2) < 0.3 or out != 'streetview':
                            patches.append(street_map[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size])
                        else:
                            recon_mask[i, j] = 0
                
                if patches:
                    self.ds.append({'patches': torch.stack(patches, dim=0).float(), 'recon_mask': recon_mask.float()})
                
                maxn -= 1
                pbar.update()
                
                if maxn == 0:
                    break
    
    def __getitem__(self, index):
        return self.ds[index]['patches'], self.ds[index]['recon_mask']
    
    def __len__(self):
        return len(self.ds)
