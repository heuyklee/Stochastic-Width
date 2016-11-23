원본 주소: https://github.com/facebook/fb.resnet.torch

실행 시 opts.lua 에서 각종 옵션 조절 가능, 
현재 서버컴에서 작업 시 nGPU 1 에서만 제대로 동작, 
현재 bypass는 실험을 위하여 bypassRate을 0으로 설정해 놓았고(SpatialConvolution2.lua 내부
hyper parameter 설정 부분 참고) models/resnet.lua line176 쪽에서 모든 layer가 아닌 앞쪽 3개의 conv에서만 
bypass 발생하도록 하였음.

주요 작업 폴더
 - SpatialConvolution2.lua 파일 내 init 위에서 함수 추가 및 init에 내용 추가,
   updateOutput, updateGradInput 함수 수정

 - train.lua 파일 내 train 함수 수정

 - models/resnet.lua 파일 내 사용하지 않는 부분 주석처리 및 model 자체에서
   모델에 포함된 SpatialConvolution2.lua 모듈의 정보를 관리하기 위한 수정

학습 순서에 따른 코드 실행 내용(간략히)
1) main.lua에서 model, trainer 등 각종 테이블 생성

2) main.lua 내 for문에서 opts.lua 또는 th main.lua 실행 시 넘긴 epoch 수만큼 학습, 테스트 진행

3) trainer:train 함수를 통해 학습과정 진행
   현재는 forward 시 이전 conv의 output을 복사하는 코드를 SpatialConvolution2.lua 내부 updateOutput 함수에 구현
   현재 코드에서는 backward 시 별다른 추가적인 동작은 하지 않으며, backward 전에 bypass kernel weight를 모두 0으로
   만들기 위한 makeBKzero() 함수의 호출과, backward 후에 weight update가 optim.sgd() 함수의 호출을 통해 일어나기 전에 
   gradWeight를 앞 conv layer에 더해주는 동작을 통해 backward 시 뒷 conv layer의 gradient를 앞으로 전달하고자 하였음.

ResNet training in Torch
============================

This implements training of residual networks from [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) by Kaiming He, et. al.

[We wrote a more verbose blog post discussing this code, and ResNets in general here.](http://torch.ch/blog/2016/02/04/resnets.html)


## Requirements
See the [installation instructions](INSTALL.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
- Install [cuDNN v4 or v5](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
- Download the [ImageNet](http://image-net.org/download-images) dataset and [move validation images](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to labeled subfolders

If you already have Torch installed, update `nn`, `cunn`, and `cudnn`.

## Training
See the [training recipes](TRAINING.md) for addition examples.

The training scripts come with several options, which can be listed with the `--help` flag.
```bash
th main.lua --help
```

To run the training, simply run main.lua. By default, the script runs ResNet-34 on ImageNet with 1 GPU and 2 data-loader threads.
```bash
th main.lua -data [imagenet-folder with train and val folders]
```

To train ResNet-50 on 4 GPUs:
```bash
th main.lua -depth 50 -batchSize 256 -nGPU 4 -nThreads 8 -shareGradInput true -data [imagenet-folder]
```

## Trained models

Trained ResNet 18, 34, 50, 101, 152, and 200 models are [available for download](pretrained). We include instructions for [using a custom dataset](pretrained/README.md#fine-tuning-on-a-custom-dataset), [classifying an image and getting the model's top5 predictions](pretrained/README.md#classification), and for [extracting image features](pretrained/README.md#extracting-image-features) using a pre-trained model.

The trained models achieve better error rates than the [original ResNet models](https://github.com/KaimingHe/deep-residual-networks).

#### Single-crop (224x224) validation error rate

| Network       | Top-1 error | Top-5 error |
| ------------- | ----------- | ----------- |
| ResNet-18     | 30.43       | 10.76       |
| ResNet-34     | 26.73       | 8.74        |
| ResNet-50     | 24.01       | 7.02        |
| ResNet-101    | 22.44       | 6.21        |
| ResNet-152    | 22.16       | 6.16        |
| ResNet-200    | 21.66       | 5.79        |

## Notes

This implementation differs from the ResNet paper in a few ways:

**Scale augmentation**: We use the [scale and aspect ratio augmentation](datasets/transforms.lua#L130) from [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842), instead of [scale augmentation](datasets/transforms.lua#L113) used in the ResNet paper. We find this gives a better validation error.

**Color augmentation**: We use the photometric distortions from [Andrew Howard](http://arxiv.org/abs/1312.5402) in addition to the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)-style color augmentation used in the ResNet paper.

**Weight decay**: We apply weight decay to all weights and biases instead of just the weights of the convolution layers.

**Strided convolution**: When using the bottleneck architecture, we use stride 2 in the 3x3 convolution, instead of the first 1x1 convolution.
