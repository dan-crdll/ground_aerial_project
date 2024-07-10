import lightning as L
from source.PatchesAE.encoder import Encoder
from source.PatchesAE.decoder import Decoder
from source.projector.projector import UpSamplerProjector
from torch import optim
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid
import torch 
import numpy as np


class Translator(L.LightningModule):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 projector: UpSamplerProjector,  
                 in_size=(224, 1232), 
                 lr=1e-5):
        super(Translator, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector
        
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        
        self.lr = lr
        self.criterion = nn.MSELoss()
        
        semantic = np.ones(in_size)
        self.positions = torch.tensor(np.sin(np.arange(semantic.size) + 1).reshape(semantic.shape) + np.cos(np.arange(semantic.size) + 1).reshape(semantic.shape)).unsqueeze(0)

        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.projector.parameters(), lr=self.lr)
        return optimizer
    
    def patchify(img, patch_size):
        C, H, W = img.shape
        
        patches = torch.zeros((H // patch_size * W // patch_size, C, patch_size, patch_size))
        recon_mask = torch.ones((H // patch_size, W // patch_size))
        n = 0
        for i in range(H // patch_size):
            for j in range(W // patch_size):
                patches[n] = img[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
                n += 1
        return patches, recon_mask
    
    def reconstruct_image(self, reconstruction_mask, patches):
        """
        Reconstruct image from patches using reconstruction mask (1 for patch - 0 for empty)
        """
        patch_size = patches.shape[3]
        H, W = reconstruction_mask.shape
        H, W = H * patch_size, W * patch_size

        image = torch.zeros((H, W, 3))
        n = 0

        for i in range(reconstruction_mask.shape[0]):
            for j in range(reconstruction_mask.shape[1]):
                if reconstruction_mask[i, j] == 1:
                    p = patches[n].permute(1, 2, 0)
                    n += 1
                    image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = p[:,:,:3]

        image = (image - image.min()) / (image.max() - image.min() + 1e-5)
        image = image.permute(2, 0, 1)

        return image
    
    def forward(self, x):
        x = torch.cat([x, self.positions], 0)
        x = self.patchify(x)
        x = self.encoder(x)
        x = self.projector(x)
        x = self.decoder(x)
        
        img = self.reconstruct_image(x)
        return img
        
    def training_step(self, batch):
        street, satellite = batch
        
        img = self.forward(street)
        satellite = F.interpolate(satellite, (img.shape[-1], img.shape[-1]))
        
        loss = self.criterion(img, satellite)
        
        self.log('training_loss', loss, prog_bar=True)
        grid = make_grid([satellite[0], img[0]], nrow=2)
        
        self.logger.experiment.add_image(grid)
        return loss
    
    def validation_step(self, batch):
        street, satellite = batch
        
        img = self.forward(street)
        satellite = F.interpolate(satellite, (img.shape[-1], img.shape[-1]))
        
        loss = self.criterion(img, satellite)
        
        self.log('training_loss', loss, prog_bar=True)
        grid = make_grid([satellite[0], img[0]], nrow=2)
        
        self.logger.experiment.add_image(grid)
        return loss