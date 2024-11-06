## Datasets

Test-time adaptation methods are commonly investigated in image classification tasks with covariate shifts. Here, we provide the widely used datasets.

### Single training distribution

#### Corruption-based datasets: [CIFAR-10-C/CIFAR-100-C](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet-C](https://github.com/hendrycks/robustness?tab=readme-ov-file).

The model is trained on the original CIFAR-10/100 or ImageNet datasets and evaluated on their corrupted versions. The commonly used corruptions are:


<img align="center" src="figures/imagenet-c.png" width="750">

#### Digit adaptation



#### Other distribution shifts

A model trained on CIFAR can also be evaluated on other types of distribution shifts, such as [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1). 

Similarly, models trained on ImageNet have evaluation datasets designed for other kinds of distribution shifts, for example, [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), and [ImageNet-V2](https://imagenetv2.org/).






