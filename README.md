# Neural Radiance Fields
> A Neural Radiance Field ( [NeRF](https://arxiv.org/abs/2003.08934) ) is a novel method to reconstruct continuous 3D scenes from a set of 2D images with known camera poses. A deep neural network (typically an MLP) is trained to map 5D input coordinates (3D position plus
a viewing direction) to volume density and emitted radiance. By integrating these outputs along camera rays using differentiable volume rendering, NeRF can generate photorealistic novel views of the scene.
## Index

1. [My Renders](#Renders)
2. [Setup](#Setup)
3. [Training](#Training)
4. [Evaluation](#Evaluation)
5. [References and Citations](#References-and-Citations)

### My Renders

<!--<div align="center">
  <img src="images/lego_16_epoch_400.gif" alt="Image">
</div>-->
<table>
  <tr>
    <td align="center"><img width="400" alt="nerf_output" src="images/lego_16_epoch_400.gif"><br>Novel Views</td>
    <td align="center"><img width="400" alt="sphere" src="images/speed_mesh.gif"><br>3D Reconstruction</td>
  </tr>
</table>

- Training graphs

<table>
  <tr>
    <td align="center"><img width="800" alt="loss" src="images/loss_wandb_graph.png"><br>MSE</td>
    <td align="center"><img width="800" alt="psnr" src="images/psnr_wandb_graph.png"><br>PSNR</td>
  </tr>
</table>

- Device used: [TITAN X (Pascal) 250W / 12GiB RAM](outputs/gpu.txt)

- Time to train: 12h 30m
 
   |      Testing Metrics    | Values   |
    | :---:     |  :---:                 |
    | **Avg MSE**  |  0.0012457877128773586 |
    | **Avg PSNR**  |  29.200356294523996   |


### Setup
1. #### Clone the repository and cd:

    ```sh
    git clone https://github.com/shifs999/Lego-NeRF.git
    ```

2. #### Create and activate the conda environment:

    ```sh
    conda env create -f environment.yaml
    conda activate nerf
    ```

3. #### Create a dataset directory and add the dataset to it:
- > Download Lego Dataset: https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a

### Training
After completing the [Setup](#Setup)

1. #### Change configurations 
    In the [config.py](config.py) file

    ```py
    """------------------------NeRF Config------------------------"""
    # data
    IMG_SIZE: int = 400
    BATCH_SIZE: int = 3072
    ...
    DEVICES: int = torch.cuda.device_count()
    MAX_EPOCHS: int = 17
    ```
2. #### Run the [train.py](train.py) script
    ```sh
    python train.py
    ```

    

