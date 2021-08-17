## SPatchGAN: Official TensorFlow Implementation

### Paper
- "SPatchGAN: A Statistical Feature Based Discriminator for Unsupervised Image-to-Image Translation"  (ICCV 2021)
    - [arxiv](https://arxiv.org/abs/2103.16219)

<img src='./images/s2a_cmp_github_downsized.jpg' align="center" width=1060>

<br><br>

<img src='./images/SPatchGAN_D_20210317_3x.jpg' align="center" width=612>


### Environment
- CUDA 10.0
- Python 3.6
- ``pip install -r requirements.txt``

### Dataset

- Dataset structure (dataset_struct='plain')
```
- dataset
    - <dataset_name>
        - trainA
            - 1.jpg
            - 2.jpg
            - ...
        - trainB
            - 3.jpg
            - 4.jpg
            - ...
        - testA
            - 5.jpg
            - 6.jpg
            - ...
        - testB
            - 7.jpg
            - 8.jpg
            - ...
```

- Supported extensions: jpg, jpeg, png
- An additional level of subdirectories is also supported by setting dataset_struct to 'tree', e.g.,
```
- trainA
    - subdir1
        - 1.jpg
        - 2.jpg
        - ...
    - subdir2
        - ...
```

- Selfie-to-anime:
     - The dataset can be downloaded from [U-GAT-IT](https://github.com/taki0112/UGATIT).

- Male-to-female and glasses removal:
     - The datasets can be downloaded from [Council-GAN](https://github.com/Onr/Council-GAN).
     - The images must be center cropped from 218x178 to 178x178 before training or testing.
     - For glasses removal, only the male images are used in the experiments in our paper.

### Training

- Set the suffix to anything descriptive, e.g., the date.
- Selfie-to-Anime
```bash
python main.py --dataset selfie2anime --augment_type resize_crop --suffix 20210317 --phase train
```

- Male-to-Female
```bash
python main.py --dataset male2female --cyc_weight 10 --suffix 20210317 --phase train
```

- Glasses Removal
```bash
python main.py --dataset glasses-male --cyc_weight 30 --suffix 20210317 --phase train
```
- Find the output in ``./output/SPatchGAN_<dataset_name>_<suffix>``
- The same command can be used to continue training based on the latest checkpoint.

### Testing with the latest checkpoint
- Replace ``--phase train`` with ``--phase test``

### Save a frozen model (.pb)
- Replace ``--phase train`` with ``--phase freeze_graph``
- Find the saved frozen model in ``./output/SPatchGAN_<dataset_name>_<suffix>/checkpoint/pb``

### Testing with the frozon model
```bash
cd frozen_model
python test_frozen_model.py --image <input_image_or_dir> --output_dir <output_dir> --model <frozen_model_path>
```

### Hyperparameters
- Check [configs.py](configs.py) for the hyperparameters.
- Check [tips](docs/tips.md) for more information about the hyperparameter tuning.

### Citation
```
@inproceedings{SPatchGAN2021,
  title={SPatchGAN: A Statistical Feature Based Discriminator for Unsupervised Image-to-Image Translation},
  author={Xuning Shao and Weidong Zhang},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

### Acknowledgement
- Our code is partially based on [U-GAT-IT](https://github.com/taki0112/UGATIT).
