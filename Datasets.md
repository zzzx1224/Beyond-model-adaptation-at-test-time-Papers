## Datasets

Test-time adaptation methods are commonly investigated in image classification tasks with covariate shifts. Here, we provide the widely used datasets.

### Single training distribution

#### Corruption-based datasets: [CIFAR-10-C/CIFAR-100-C](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet-C](https://github.com/hendrycks/robustness?tab=readme-ov-file).

The model is trained on the original CIFAR-10/100 or ImageNet datasets and evaluated on their corrupted versions. The commonly used corruptions are:


<img align="center" src="figures/imagenet-c.png" width="750">

#### Digit adaptation: [MNIST](https://yann.lecun.com/exdb/mnist/), [MNIST-M](https://github.com/zumpchke/keras_mnistm/releases/tag/1.0), [SVHN](http://ufldl.stanford.edu/housenumbers/), and [USPS](https://git-disl.github.io/GTDLBench/datasets/usps_dataset/).

Digit adaptation is a common benchmark in domain adaptation, where different distributions are different datasets. 
Commonly, test-time adaptation methods on digits datasets train their model on SVHN. The trained model is then adapted and evaluated on the different target distributions of MNIST, MNIST-M, and USPS.

#### Other distribution shifts: [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1); [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), and [ImageNet-V2](https://imagenetv2.org/).

A model trained on CIFAR-10 can also be evaluated on another distribution shift [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1). 

Similarly, models trained on ImageNet have evaluation datasets designed for other kinds of distribution shifts, for example, [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), and [ImageNet-V2](https://imagenetv2.org/).


### Multiple training distributions

Test-time adaptation methods are also conducted on the domain generalization datasets, where multiple source distributions are available during training. The trained model is then adapted and evaluated on the target distribution. The commonly utilized domain generalization datasets are [PACS](https://huggingface.co/datasets/flwrlabs/pacs), [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html), [VLCS](https://github.com/belaalb/G2DM?tab=readme-ov-file#download-vlcs), [TerraIncognita](https://beerys.github.io/CaltechCameraTraps/), and [DomainNet](https://ai.bu.edu/M3SDA/), where the distribution shifts are achieved by different image styles or datasets.

Beyond these datasets, [WILDS](https://wilds.stanford.edu/) contains 10 datasets across a diverse set of application areas, data modalities, and dataset sizes. Each dataset comprises data from different domains.


