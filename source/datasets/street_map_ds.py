import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import cv2 as cv
import numpy as np
import torch
from source.PatchesAE.autoencoder import Autoencoder
from source.PatchesAE.encoder import Encoder
from source.PatchesAE.decoder import Decoder


class StreetSatDataset(Dataset):
    def __init__(self, ds_path, model_path, train=True, maxn=-1, start=-1):
        super(StreetSatDataset, self).__init__()
        
        enc = Encoder(256, 4, 16, 5, 16, 0.3)
        enc.load_state_dict(torch.load(model_path))
        
        self.enc_street = enc
        
        self.enc_street.eval().to('cuda')
        print('Models loaded...')
        self.ds = []
        
        path = 'CVPR_subset/splits/train-19zl.csv' if train else 'CVPR_subset/splits/val-19zl.csv'
        df = pd.read_csv(f"{ds_path}/{path}", delimiter=',').reset_index()

        with tqdm(total=len(df) if maxn == -1 else maxn) as pbar:
            for idx, row in df.iterrows():
                if idx < start:
                    continue
                
                street_map = cv.cvtColor(cv.imread(f"{ds_path}/CVPR_subset/{row['streetview']}"), cv.COLOR_BGR2RGB)
                street_map = (torch.tensor(street_map).permute(2, 0, 1) / 127.5) - 1
                
                semantic = np.ones((street_map.shape[1], street_map.shape[2]))
                positions = torch.tensor(np.sin(np.arange(semantic.size) + 1).reshape(semantic.shape) + np.cos(np.arange(semantic.size) + 1).reshape(semantic.shape)).unsqueeze(0)
                street_map = torch.cat([street_map, positions], 0)
                street_map, _ = self.patchify(street_map, 16)

                with torch.no_grad():
                    street_map = self.enc_street(street_map.unsqueeze(0).to('cuda')).to('cpu').squeeze(0)
                
                satellite_map = cv.cvtColor(cv.imread(f"{ds_path}/CVPR_subset/{row['map']}"), cv.COLOR_BGR2RGB)
                satellite_map = cv.resize(satellite_map, (512, 512))
                satellite_map = (torch.tensor(satellite_map).permute(2, 0, 1) / 127.5) - 1

                self.ds.append({
                    'satellite': satellite_map,
                    'street': street_map
                })     
                maxn -= 1
                pbar.update()
                
                if maxn == 0:
                    break
        del self.enc_street
        del enc 
        del dec
        del ae_street
        torch.cuda.empty_cache()
    
    def __getitem__(self, index):
        return self.ds[index]['street'], self.ds[index]['satellite']
    
    def __len__(self):
        return len(self.ds)
    
    def patchify(self, img, patch_size):
        C, H, W = img.shape
        
        patches = torch.zeros((H // patch_size * W // patch_size, C, patch_size, patch_size))
        recon_mask = torch.ones((H // patch_size, W // patch_size))
        n = 0
        for i in range(H // patch_size):
            for j in range(W // patch_size):
                patches[n] = img[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
                n += 1
        return patches, recon_mask
