# Practical Deep Dispersed Watermarking with Synchronization and Fusion
This repository is the official implementation of [Practical Deep Dispersed Watermarking with Synchronization and Fusion](https://dl.acm.org/doi/abs/10.1145/3581783.3612015).

## Introduction
![framework](IMG/framework.png)

This paper focuses on two important and practical aspects that are not well addressed in existing deep learning based works, i.e., embedding in arbitrary resolution (especially high resolution) images, and robustness against complex attacks.
To overcome these limitations, we propose a blind watermarking framework (called DWSF) which mainly consists of three novel components, i.e., dispersed embedding, watermark synchronization and message fusion.


## Dependencies

### environment
```
python         3.7.3
torch          1.10.0
numpy          1.21.6
Pillow         9.1.1
tqdm           4.64.1
kornia         0.6.8
crc8           0.1.0
opencv-python  4.6.0.66
torchsummary   1.5.1
torchvision    0.11.1
```

### dataset
[COCO2017](https://cocodataset.org/#home)

[ImageNet](https://www.image-net.org/)

[OpenImages](https://storage.googleapis.com/openimages/web/index.html)

[LabelMe](http://labelme2.csail.mit.edu/Release3.0/browserTools/php/publications.php)

## Usage

### training
train encoder_decoder
```
python train_ed.py --train_dataset_path train_dataset_path --val_dataset_path val_dataset_path --save_path pth_output_path
```
train segmentation model
```
# generate watermarked image and mask
python generate_segdata.py --img_path original_image_path --out_path watermarked_img_mask_path --weight_path encoder_decoder_pth_path

# train segmentation mode
python train_seg.py --train_path train_watermarked_img_mask_path --test_path test_watermarked_img_mask_path --output_path pth_output_path
```

### evaluating
```
python evaluate.py --ori_path original_image_path --pth_path encoder_decoder_pth_path --out_path output_path
```

## citation
If you find this work useful, please cite our paper:
```
@inproceedings{guo2023practical,
  title={Practical Deep Dispersed Watermarking with Synchronization and Fusion},
  author={Guo, Hengchang and Zhang, Qilong and Luo, Junwei and Guo, Feng and Zhang, Wenbin and Su, Xiaodong and Li, Minglei},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7922--7932},
  year={2023}
}
```

