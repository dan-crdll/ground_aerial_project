import copy 

import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import OmegaConf

from source.ldm.ldm import DenoisingNetwork
from diffusers import PNDMScheduler, AutoencoderKL

from source.datasets.street_map_ds import StreetSatDataset

from lightning.pytorch import seed_everything, Trainer
seed_everything(0)


@hydra.main('./conf', 'ldm_config.yaml', version_base=None)
def main(cfg):
    cfg = OmegaConf.to_object(cfg)
    ds_cfg = cfg['dataset']
    ldm_cfg = cfg['ldm']
    
    # LOADING TRAINING AND VALIDATION DATASET AND DATALOADER
    train_ds = StreetSatDataset(
        ds_path=ds_cfg['ds_path'],
        model_path=ds_cfg['model_path'],
        train=True,
        maxn=ds_cfg['max_train']
    )
    val_ds = StreetSatDataset(
        ds_path=ds_cfg['ds_path'],
        model_path=ds_cfg['model_path'],
        train=False,
        maxn=ds_cfg['max_val']
    )
    
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=ds_cfg['bsz'])
    val_dl = DataLoader(val_ds, batch_size=ds_cfg['bsz'])
    
    # MODEL DEFINITION
    vae = AutoencoderKL.from_single_file('https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors')
    
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, num_train_timesteps=ldm_cfg['timesteps'])
    model = DenoisingNetwork(
        depth=ldm_cfg['depth'], 
        timesteps=ldm_cfg['timesteps'], 
        noise_scheduler=copy.deepcopy(scheduler), 
        autoencoder=copy.deepcopy(vae)
    )
    
    # TRAINING PHASE
    trainer = Trainer(
        max_epochs=ldm_cfg['max_epochs'], 
        gradient_clip_val=1.0, 
        precision='16-mixed', 
        log_every_n_steps=1, 
        accumulate_grad_batches=ldm_cfg['accumulate_grad']
    )
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
    
    print('Saving model')
    torch.save(model.state_dict(), './pretrained_ldm.pth')
    print('Model successfully saved.')


if __name__ == "__main__":
    main()