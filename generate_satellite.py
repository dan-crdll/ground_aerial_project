from argparse import ArgumentParser

import torch
import cv2 as cv
import numpy as np

from source.PatchesAE.encoder import Encoder
from source.datasets.street_map_ds import StreetSatDataset

from omegaconf import OmegaConf

from diffusers import AutoencoderKL, PNDMScheduler
from source.ldm.ldm import DenoisingNetwork
from tqdm.auto import tqdm

import matplotlib.pyplot as plt


def main(streetview_path):
    # READING IMAGE AND CONCATENATING POSITION
    sv_img = cv.imread(streetview_path, cv.IMREAD_COLOR)
    sv_img = cv.cvtColor(sv_img, cv.COLOR_BGR2RGB)
    
    plt.imshow(sv_img)
    plt.show()
    
    sv_img = torch.Tensor(sv_img).permute(2, 0, 1)
    sv_img = (sv_img / 127.5) - 1
    
    semantic = np.zeros((sv_img.shape[1], sv_img.shape[2]))
    
    positions = torch.tensor(np.sin(np.arange(semantic.size) + 1)
                                         .reshape(semantic.shape) + np.cos(np.arange(semantic.size) + 1)
                                         .reshape(semantic.shape)).unsqueeze(0) # creation of the position embeddings
                
    sv_img = torch.cat([sv_img, positions], dim=0) # 4 x H x W
    
    # LOAD THE PATCHES ENCODER
    cfg = OmegaConf.load('./conf/pretrain_patch_config.yaml')
    cfg = OmegaConf.to_object(cfg)['autoencoder']
    encoder = Encoder(
        embed_dim=cfg['embed_dim'],
        in_channels=cfg['in_channels'],
        patch_size=16,
        depth=cfg['encoder_depth'],
        num_heads=cfg['encoder_heads'],
        dropout=cfg['dropout']
    )
    encoder.load_state_dict(torch.load('./streetview_encoder_state.pth'))
    encoder.eval()
    encoder.to('cuda')
    
    # ENCODE THE CONDITIONING IMAGE
    sv_patches, _ = StreetSatDataset.patchify(sv_img, 16)
    with torch.no_grad():
        conditioning = encoder(sv_patches.unsqueeze(0).to('cuda'))
    
    del encoder
    torch.cuda.empty_cache()
    
    # BACKWARD DIFFUSION
    cfg = OmegaConf.load('./conf/ldm_config.yaml')
    cfg = OmegaConf.to_object(cfg)['ldm']
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, num_train_timesteps=cfg['timesteps'])    
    vae = AutoencoderKL.from_single_file('https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors')
    ldm = DenoisingNetwork(cfg['timesteps'], scheduler, vae, depth=cfg['depth'])
    ldm.load_state_dict(torch.load('./pretrained_ldm.pth'))
    ldm.eval()
    vae.eval()

    scheduler.set_timesteps(cfg['timesteps'])
    z = torch.randn((1, 4, 64, 64))
    
    conditioning = torch.cat([conditioning, torch.zeros_like(conditioning)]).to('cuda')
    z = torch.cat([z] * 2)
    
    ldm.to('cuda')
    for t in tqdm(reversed(range(int(cfg['timesteps']))), total=int(cfg['timesteps'])):
        z = scheduler.scale_model_input(z.to('cpu'), t)
        with torch.no_grad():
            pred = ldm(z.to('cuda'), conditioning.to('cuda'), torch.Tensor([t]).to('cuda'))
            
        cond, uncond = pred.chunk(2)
        pred = uncond + 7.5 * (cond - uncond)
        z = scheduler.step(pred.to('cpu'), t, z.to('cpu')).prev_sample
    
    with torch.no_grad():
        sat = (((vae.decode(z.to('cuda')).sample.clamp(-1.0, 1.0)) + 1) * 127.5).int()[0].permute(1, 2, 0)
    
    plt.imshow(sat.detach().cpu())
    plt.show()


if __name__=='__main__':
    parser = ArgumentParser('generate_satellite')
    parser.add_argument('--streetview', help="Streetview image path")
    
    args = parser.parse_args()
    args = args.__dict__
    
    main(args['streetview'])