import lightning as L
from torch import optim
from torch.nn import MSELoss
from torchvision.utils import make_grid
import torch 


class Autoencoder(L.LightningModule):
    def __init__(self, encoder, decoder, lr=1e-5):
        super(Autoencoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = MSELoss()
        
        self.lr = lr

        
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
   
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        patches, recon_mask = batch
        
        with torch.autocast(device_type=self.device.type, enabled=False):
            pred = self.forward(patches)
            loss = self.criterion(pred.float(), patches[:, :, :3].float())
        
        self.log('train_loss', loss, prog_bar=True)
        
        if batch_idx == 0:
            grid = make_grid([self.reconstruct_image(recon_mask[0], patches[0]), self.reconstruct_image(recon_mask[0], pred[0].detach())], nrow=2)
            self.logger.experiment.add_image('train_images', grid, self.current_epoch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        patches, recon_mask = batch
        
        with torch.autocast(device_type=self.device.type, enabled=False):
            pred = self.forward(patches)
            loss = self.criterion(pred.float(), patches[:,:,:3].float())
        self.log('test_loss', loss, prog_bar=True)
        
        if batch_idx == 0:
            grid = make_grid([self.reconstruct_image(recon_mask[0], patches[0]), self.reconstruct_image(recon_mask[0], pred[0].detach())], nrow=2)
            self.logger.experiment.add_image('test_images', grid, self.current_epoch)
        return loss