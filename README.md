# Ground to Aerial Image Translation
#### Daniele Santino Cardullo - 2127806

## Problem Description
Ground to Aerial Image Translation is an image-to-image translation task with various uses both in industry and research. In particular a non exhaustive list of possible domain of application is:
- Robotics, in particular mobile robotics;
- Autonomous driving cars;
- Geolocalization from images.

Hence finding solutions to tackle this task is useful.

The majority of existing approaches to this problem involve the use of GANs for image translation, despite the obtained results these approaches suffers from the limits imposed by the architecture: difficulty and instability during training and difficulty in controlling the generation during inference. 

To try to overcome these limitations, this project focuses on the implementation of a latent diffusion model to solve the task.

## Implemented Architecture
### Streetview Images Autoencoder
The approach taken to encode the streetview images is to use a masked autoencoder but with a modified training phase and structure with the one described in [1]. In this implementation of the autoencoder both the encoder and the decoder have same dimensionality, and uses a ViT like structure. During training phase the model is fed with a portion of the patches, based on their semantics; during inference the input is the complete patched image.

![Encoder Structure](imgs/encoder.png)

The above encoder structure is formed by a flatten operation on patches, which are later linearly embedded. Finally the flattened and embedded patches go through a series of ViT-like encoders.

![Decoder Structure](imgs/encoder.png)

The decoder structure is the specular architecture of the encoder, it receives the encoded patches that goes through a series of ViT decoders, then they are unflattened to have the image patch original structure (removing the positional encodings).

Encoder and decoder are combined to create the autoencoder, after training only the encoder part is used to encode the streetview images.

To pretrain the autoencoder it is sufficient to run `python pretrain_patch_autoencoder.py`, to change the hyperparameters and the training variables it is possible to modify `pretrain_patch_config.yaml` file.


# Reference
[1] Masked Autoencoders Are Scalable Vision Learners - K. He, et al. (https://arxiv.org/abs/2111.06377)

