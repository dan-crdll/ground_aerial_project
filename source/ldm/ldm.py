import lightning as L
from torch import nn
from torch import optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from diffusers import PNDMScheduler
from diffusers import AutoencoderKL
import torch


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.mha = nn.MultiheadAttention(embed_dim, 16, 0.3, batch_first=True)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.norm_2 = nn.LayerNorm(embed_dim)
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            
    def forward(self, x, conditioning):
        q = self.q_linear(conditioning)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        attention, _ = self.mha(q, k, v)
        x = self.norm_1(attention + x)
        
        out = self.mlp(x) + x 
        out = self.norm_2(out)
        return out


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(DownConvBlock, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.SiLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.SiLU()
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.max_pool = nn.MaxPool2d((2, 2), stride=2, ceil_mode=True)
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.max_pool(x)
        return x
    
    
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(UpConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.SiLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.SiLU()
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class DownLayerConditional(nn.Module):
    def __init__(self, in_channels, out_channels, condition_channels, condition_dim, input_dimension, dropout):
        super(DownLayerConditional, self).__init__()
        # conditions are 1078 x 256 = 275'968
        # inputs are 4 x 64 x 64 = 16'384
        self.projector = nn.Sequential(
            nn.Conv1d(condition_channels, in_channels, 5, padding='same'),
            nn.Linear(condition_dim, input_dimension ** 2),
            nn.SiLU()
        )
        self.cross_attention = CrossAttention(input_dimension ** 2)
        self.unflatten = nn.Unflatten(dim=2, unflattened_size=(input_dimension, input_dimension))
        self.down_layer = DownConvBlock(in_channels, out_channels, dropout)
        
    def forward(self, input, condition):
        condition = self.projector(condition)   # in_channels x input_dimension ** 2
        input = torch.flatten(input, start_dim=2)   # in_channels x input_dimension ** 2
        
        x = self.cross_attention(input, condition)
        x = self.unflatten(x)
        x = self.down_layer(x)
        return x
    
class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(UpLayer, self).__init__()
        self.upconv = UpConvBlock(in_channels * 2, out_channels, dropout)
        
    def forward(self, x, prec):
        x = torch.cat([x, prec], dim=1)
        x = self.upconv(x)
        return x
        


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, input_dimension=64, dropout=0.3, condition_channels=1078, condition_dimension=256, device='cuda'):
        super(UNet, self).__init__()
        
        self.projector = nn.Sequential(
            nn.Conv1d(condition_channels, in_channels, 5, padding='same'),
            nn.Linear(condition_dimension, input_dimension ** 2),
            nn.Unflatten(2, (input_dimension, input_dimension)),
            nn.SiLU()
        )
        
        # 16 x 32 x 32
        self.down_layer_1 = DownConvBlock(in_channels * 2, 16, dropout)
        # 32 x 16 x 16
        self.down_layer_2 = DownLayerConditional(16, 32, condition_channels, condition_dimension, input_dimension // 2, dropout)
        # 64 x 8 x 8
        self.down_layer_3 = DownConvBlock(32, 64, dropout)
        # 128 x 4 x 4
        self.down_layer_4 = DownLayerConditional(64, 128, condition_channels, condition_dimension, input_dimension // 8, dropout)
        
        self.encode_layer = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.SiLU(), 
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.SiLU(),
            nn.Dropout(dropout)
        )   # 128 x 4 x 4
        
        self.up_layer_1 = UpLayer(129, 64, dropout) # 64 x 8 x 8
        self.up_layer_2 = UpLayer(64, 32, dropout) # 32 x 16 x 16
        self.up_layer_3 = UpLayer(32, 16, dropout) # 16 x 32 x 32
        self.up_layer_4 = UpLayer(16, 8, dropout) # 8 x 64 x 64
        
        self.out_layer = nn.Sequential(
            nn.Conv2d(8, out_channels, (3, 3), padding='same'),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding='same'),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels)
        )
        
        self.device = device

    def create_time_embedding(self, t):
        positions = torch.zeros((1,1,4,4), device=self.device)
        
        for i in range(positions.shape[2]):
            for j in range(positions.shape[3]):
                pos = torch.Tensor([(i + 1) * (t + j) / 1000 ]).to(self.device)
                positions[0, 0, i, j] = torch.cos(pos)
        return positions
    
    def forward(self, x, t, c):
        N, *_ = x.shape
        time_embeddings = self.create_time_embedding(t)
        x = torch.cat([x, self.projector(c)], dim=1)
        x1 = self.down_layer_1(x)    # 16 x 32 x 32
        x2 = self.down_layer_2(x1, c)    # 32 x 16 x 16
        x3 = self.down_layer_3(x2)    # 64 x 8 x 8
        x4 = self.down_layer_4(x3, c)    # 128 x 4 x 4
        
        y = self.encode_layer(x4)   # 128 x 4 x 4

        y = torch.cat([y, time_embeddings.repeat(N, 1, 1, 1)], dim=1)
        x4 = torch.cat([x4, time_embeddings.repeat(N, 1, 1, 1)], dim=1)
        
        y = self.up_layer_1(y, x4)  # 64 x 8 x 8
        y = self.up_layer_2(y, x3)  # 32 x 16 x 16
        y = self.up_layer_3(y, x2)  # 16 x 32 x 32
        y = self.up_layer_4(y, x1)  # 8 x 64 x 64
        
        y = self.out_layer(y)   # 4 x 64 x 64
        return y


class DenoisingNetwork(L.LightningModule):
    def __init__(self, timesteps, noise_scheduler: PNDMScheduler, autoencoder: AutoencoderKL, guidance_scale=7.5, depth=10):
        super(DenoisingNetwork, self).__init__()
        self.timesteps = timesteps
        
        self.unet = nn.ModuleList(
            [UNet(4, 4, 64, 0, 1078, 256, device='cuda') for i in range(depth)]
        )
        self.noise_scheduler = noise_scheduler
        self.criterion_1 = nn.L1Loss()
        self.criterion_2 = StructuralSimilarityIndexMeasure()
        self.guidance_scale = guidance_scale
        self.vae = autoencoder
        self.vae.requires_grad_(False)
        self.vae.eval()

      
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        
        return optimizer
    
      
    def forward(self, x, conditioning, t):
        for u in self.unet:
            x = u(x, t, conditioning)
        return x
    
    def training_step(self, batch, idx):
        street, satellite = batch
        with torch.no_grad():
            latents = self.vae.encode(satellite).latent_dist
            satellite = latents.sample() * self.vae.config.scaling_factor
            
        noise = torch.randn_like(satellite, device=self.device)
        
        t = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (1,), device=self.device
        ).long()
        
        noisy_latents = self.noise_scheduler.add_noise(satellite, noise, t)
        
        
        with torch.autocast(self.device.type, enabled=False):
            noise_pred_cond = self.forward(noisy_latents, street, t)
            noise_pred_uncond = self.forward(noisy_latents, torch.zeros_like(street, device=self.device), t)
            
            noise_pred = noise_pred_cond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            loss = self.criterion_1(noise_pred, noise)
            
        self.log('training_loss', loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, idx):  
        street, satellite = batch
        with torch.no_grad():
            latents = self.vae.encode(satellite).latent_dist
            satellite = latents.sample() * self.vae.config.scaling_factor
            
        noise = torch.randn_like(satellite, device=self.device)
        
        t = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (1,), device=self.device
        ).long()
        
        noisy_latents = self.noise_scheduler.add_noise(satellite, noise, t)
        
        
        with torch.autocast(self.device.type, enabled=False):
            noise_pred_cond = self.forward(noisy_latents, street, t)
            noise_pred_uncond = self.forward(noisy_latents, torch.zeros_like(street, device=self.device), t)
            
            noise_pred = noise_pred_cond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            loss = self.criterion_1(noise_pred, noise)
            
        self.log('test_loss', loss, prog_bar=True)
        return loss
            