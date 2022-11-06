# Exploring Implicit Domain-invariant Features for Domain Adaptive Object Detection (IEEE TCSVT)
(IEEE Transactions on Circuits and Systems for Video Technology)

A Pytorch Implementation of Implicit Domain-invariant Faster R-CNN (IDF) for Domain Adaptive Object Detection. 

## Introduction
Please follow [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) respository to setup the environment. In this project, we use Pytorch 1.0.1 and CUDA version is 10.0.130. 

## Datasets
### Datasets Preparation
* **Cityscape and FoggyCityscape:** Download the [Cityscape](https://www.cityscapes-dataset.com/) dataset, see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data).
* **KITTI:** Download the dataset from this [website](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) to prepare KITTI dataset.
* **Sim10k:** Download the dataset from this [website](https://fcav.engin.umich.edu/sim-dataset/).  

### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  
If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in ```lib/datasets/```, and add it to ```lib/datasets/factory.py```, ```lib/datasets/config_dataset.py```. Then, add the dataset option to ```lib/model/utils/parser_func.py```.

### Implied Transferable Samples and Target Pseudo Label
You should use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to generate the implied transferable samples. And the implied transferable target sample used to train a Faster R-CNN. Then the detection results of this model can be the target pseudo label.

## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and write the path in **__C.VGG_PATH** and **__C.RESNET_PATH** at ```lib/model/utils/config.py```.

## Train
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python trainval_idf.py \
       --dataset source_dataset --dataset_t target_dataset \
       --net vgg16/resnet101 
```
## Test
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python test_idf.py \
       --dataset source_dataset --dataset_t target_dataset \
       --net vgg16/resnet101  \
       --load_name path_to_model
```
## Citation
If you find this repository useful, please cite our paper:
```
@ARTICLE{9927485,
  author={Lang, Qinghai and Zhang, Lei and Shi, Wenxu and Chen, Weijie and Pu, Shiliang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Exploring Implicit Domain-invariant Features for Domain Adaptive Object Detection}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3216611}}
```
```
