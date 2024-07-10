import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import cv2 as cv
import numpy as np
import torch


class PatchifiedDataset(Dataset):
    """
    This class is a torch Dataset which outputs a patchified (and masked) version of the images in CVPR 
    subset.
    """
    def __init__(self, ds_path, train=True, maxn=-1, start=-1, patch_size=16, out='streetview'):
        
        assert out == 'streetview' or out == 'map'
        
        super(PatchifiedDataset, self).__init__()
        self.ds = []
        
        path = 'CVPR_subset/splits/train-19zl.csv' if train else 'CVPR_subset/splits/val-19zl.csv'
        df = pd.read_csv(f"{ds_path}/{path}", delimiter=',').reset_index()

        with tqdm(total=len(df) if maxn == -1 else maxn) as pbar:
            for idx, row in df.iterrows():
                if idx < start:
                    continue
                
                street_map = cv.cvtColor(cv.imread(f"{ds_path}/CVPR_subset/{row[out]}"), cv.COLOR_BGR2RGB)  # read the image
                
                if out == 'streetview':
                    # read the semantic in the case of the streetview image
                    semantic = cv.imread(f"{ds_path}/CVPR_subset/{row['annotations']}", cv.IMREAD_GRAYSCALE) 
                else:
                    street_map = cv.resize(street_map, (512, 512))  #   otherwise just reduce the size of the satellitar map
                
                street_map = (torch.tensor(street_map).permute(2, 0, 1) / 127.5) - 1    # normalize the image in [-1, 1]
                
                if out != 'streetview':
                    semantic = np.zeros((street_map.shape[1], street_map.shape[2])) 
                    
                positions = torch.tensor(np.sin(np.arange(semantic.size) + 1)
                                         .reshape(semantic.shape) + np.cos(np.arange(semantic.size) + 1)
                                         .reshape(semantic.shape)).unsqueeze(0) # creation of the position embeddings
                
                street_map = torch.cat([street_map, positions], 0)  # concatenation of embeddings
                
                C, H, W = street_map.shape
                recon_mask = torch.ones(H // patch_size, W // patch_size)   # mask to reconstruct the original image from patches
                
                patches = []
                for i in range(H // patch_size):
                    for j in range(W // patch_size):
                        semantic_patch = semantic[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                        
                        # selection of patches to discart based on the presence of sky or just randomly for satellite views
                        if (semantic_patch == 0).sum() / (patch_size ** 2) < 0.3 or (out != 'streetview' and torch.randn(1) > 0.3):
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
