# [CycleGAN](https://junyanz.github.io/CycleGAN/)

This is a simple PyTorch implementation of [CycleGAN](https://junyanz.github.io/CycleGAN/).

![](media/cyclegan.gif)

## Key Details

 - Patch Discriminator is used [16x16].  
 - Different Learning Rate is used for Generator and Discriminator. Discriminator uses a lower learning rate.    
 - Discriminator's learning is disabled during the training of Generator.   
 - Leaky Relu with a negative slope of 0.2 is used in Generator, but ReLU is used in Discriminator.  
 - Instance Normalization is used.  
 - Output of the Generator is clamped to output a value between [-0.5, 0.5]

## Generator Model Architecture

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 128, 128]           3,072
    InstanceNorm2d-2         [-1, 64, 128, 128]               0
         LeakyReLU-3         [-1, 64, 128, 128]               0
            Conv2d-4          [-1, 128, 64, 64]         131,072
    InstanceNorm2d-5          [-1, 128, 64, 64]               0
         LeakyReLU-6          [-1, 128, 64, 64]               0
            Conv2d-7          [-1, 256, 32, 32]         524,288
    InstanceNorm2d-8          [-1, 256, 32, 32]               0
         LeakyReLU-9          [-1, 256, 32, 32]               0
           Conv2d-10          [-1, 256, 16, 16]       1,048,576
   InstanceNorm2d-11          [-1, 256, 16, 16]               0
        LeakyReLU-12          [-1, 256, 16, 16]               0
           Conv2d-13          [-1, 256, 16, 16]         589,824
   InstanceNorm2d-14          [-1, 256, 16, 16]               0
        LeakyReLU-15          [-1, 256, 16, 16]               0
           Conv2d-16          [-1, 256, 16, 16]         589,824
   InstanceNorm2d-17          [-1, 256, 16, 16]               0
    ResidualBlock-18          [-1, 256, 16, 16]               0
           Conv2d-19          [-1, 256, 16, 16]         589,824
   InstanceNorm2d-20          [-1, 256, 16, 16]               0
        LeakyReLU-21          [-1, 256, 16, 16]               0
           Conv2d-22          [-1, 256, 16, 16]         589,824
   InstanceNorm2d-23          [-1, 256, 16, 16]               0
    ResidualBlock-24          [-1, 256, 16, 16]               0
  ConvTranspose2d-25          [-1, 256, 32, 32]       1,048,576
   InstanceNorm2d-26          [-1, 256, 32, 32]               0
        LeakyReLU-27          [-1, 256, 32, 32]               0
  ConvTranspose2d-28          [-1, 128, 64, 64]         524,288
   InstanceNorm2d-29          [-1, 128, 64, 64]               0
        LeakyReLU-30          [-1, 128, 64, 64]               0
  ConvTranspose2d-31         [-1, 64, 128, 128]         131,072
   InstanceNorm2d-32         [-1, 64, 128, 128]               0
        LeakyReLU-33         [-1, 64, 128, 128]               0
  ConvTranspose2d-34          [-1, 3, 256, 256]           3,072
================================================================
Total params: 5,773,312
Trainable params: 5,773,312
Non-trainable params: 0
----------------------------------------------------------------
```

## Discriminator Model Architecture

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 128, 128]           3,072
            Conv2d-2          [-1, 128, 64, 64]         131,072
    InstanceNorm2d-3          [-1, 128, 64, 64]               0
            Conv2d-4          [-1, 256, 32, 32]         524,288
    InstanceNorm2d-5          [-1, 256, 32, 32]               0
            Conv2d-6          [-1, 512, 16, 16]       2,097,152
    InstanceNorm2d-7          [-1, 512, 16, 16]               0
            Conv2d-8            [-1, 1, 16, 16]           4,608
================================================================
Total params: 2,760,192
Trainable params: 2,760,192
Non-trainable params: 0
----------------------------------------------------------------
```