## SPatchGAN: Official TensorFlow Implementation

### Environment
- CUDA 10.0
- requirements.txt

### Dataset
- dataset
  - selfie2anime
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

- Supported extensions: jpg, jpeg, png
- An additional level of subdirectories is also supported by setting 'dataset_struct' to 'tree', e.g.,
  - trainA
    - subdir1
      - 1.jpg
      - 2.jpg
      - ...
    - subdir2
      - ...

### Training

- Set the suffix to anything descriptive, e.g., the date.
- Selfie-to-Anime
```bash
python main.py --dataset selfie2anime --augment_type resize_crop --suffix 20210317
```

- Male-to-Female
```bash
python main.py --dataset male2female --cyc_weight 10 --suffix 20210317
```

- Glasses Removal
```bash
python main.py --dataset glasses-male --cyc_weight 30 --suffix 20210317
```

### Testing

```bash
python main.py --dataset <dataset_name> --phase test --suffix 20210317
```