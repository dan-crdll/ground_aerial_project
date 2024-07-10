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


# Reference
[1] Masked Autoencoders Are Scalable Vision Learners - K. He, et al. (https://arxiv.org/abs/2111.06377)[https://arxiv.org/abs/2111.06377]

