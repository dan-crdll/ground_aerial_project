import torch

import hydra
from omegaconf import OmegaConf

from source.datasets.patchified_dataset import PatchifiedDataset
from torch.utils.data import DataLoader

from source.PatchesAE.encoder import Encoder
from source.PatchesAE.decoder import Decoder
from source.PatchesAE.autoencoder import Autoencoder

from lightning.pytorch import seed_everything, Trainer
seed_everything(0)


@hydra.main(config_path='.', config_name='pretrain_patch_config', version_base=None)
def main(cfg):
    cfg = OmegaConf.to_object(cfg)
    ds_cfg = cfg['dataset']
    model_cfg = cfg['autoencoder']
    
    # DATASET LOADING AND DATALOADERS
    train_ds = PatchifiedDataset(ds_cfg['ds_path'], maxn=ds_cfg['max_train'], patch_size=ds_cfg['patch_size'])
    val_ds = PatchifiedDataset(ds_cfg['ds_path'], train=False, maxn=ds_cfg['max_val'], patch_size=ds_cfg['patch_size'])

    train_dl = DataLoader(train_ds, shuffle=True)
    val_dl = DataLoader(val_ds)
    
    # MODEL DEFINITION
    encoder = Encoder(
        model_cfg['embed_dim'], 
        model_cfg['in_channels'], 
        ds_cfg['patch_size'], 
        model_cfg['encoder_depth'], 
        model_cfg['encoder_heads'], 
        model_cfg['dropout']
    )
    decoder = Decoder(
        model_cfg['embed_dim'],
        model_cfg['out_channels'],
        ds_cfg['patch_size'],
        model_cfg['decoder_depth'],
        model_cfg['decoder_heads'],
        model_cfg['dropout']
    )
    model = Autoencoder(encoder, decoder, model_cfg['lr'])
    
    # TRAINING PHASE
    trainer = Trainer(max_epochs=model_cfg['max_epochs'], precision='16-mixed', gradient_clip_val=1, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    print('Saving model state dictionary')
    torch.save(model.state_dict(), './streetview_autoencoder_state.pth')
    torch.save(model.encoder.state_dict(), './streetview_encoder_state.pth')
    print('Models successfully saved')


if __name__ == '__main__':
    main()