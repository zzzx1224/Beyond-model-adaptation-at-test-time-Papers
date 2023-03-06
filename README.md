# Papers of Test-time adaptation

## What to adapt

### Model adaptation

### - Fine-tuning-based methods

- [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](http://proceedings.mlr.press/v119/sun20b.html), ICML 2020.
- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c), ICLR 2021.
- [Bayesian Adaptation for Covariate Shift](https://proceedings.neurips.cc/paper/2021/hash/07ac7cd13fd0eb1654ccdbd222b81437-Abstract.html), NeurIPS 2021.
- [Tailoring: encoding inductive biases by optimizing unsupervised objectives at prediction time](https://proceedings.neurips.cc/paper/2021/hash/f45cc474bff52cb1b2268a2f94a2abcf-Abstract.html), NeurIPS 2021.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [Contrastive Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.html), CVPR 2022.
- [Continual Test-Time Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html), CVPR 2022.
- [Efficient Test-Time Model Adaptation without Forgetting](https://arxiv.org/abs/2204.02610), ICML 2022.
- [TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html), NeurIPS 2022.
- [Test-Time Training with Masked Autoencoders](https://openreview.net/group?id=ICLR.cc/2023/Conference/Authors&referrer=%5BHomepage%5D(%2F)), NeurIPS 2022.
- [Test Time Adaptation via Conjugate Pseudo-labels](https://openreview.net/forum?id=2yvUYc-YNUH), NeurIPS 2022.
- [MEMO: Test Time Robustness via Adaptation and Augmentation](https://arxiv.org/abs/2110.09506), Arxiv.
- [MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption](https://proceedings.mlr.press/v151/bartler22a.html), AISTATS 2022.
- [A Probabilistic Framework for Lifelong Test-Time Adaptation](https://arxiv.org/abs/2212.09713), Arxiv.

### - Prototype-based/context-based methods (Function-inferring methods.)
Efficiently infer the functions/neural network parameters by a single forward pass.
- [Adaptive Methods for Real-World Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.html?ref=https://githubhelp.com), CVPR 2021.
- [Learning to Generalize across Domains on Single Test Samples](https://openreview.net/forum?id=CIaQKbTBwtU), ICLR 2022.
- [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html), NeruIPS 2022. (?)
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [Extrapolative Continuous-time Bayesian Neural Network for Fast Training-free Test-time Adaptation](https://openreview.net/forum?id=wiHzQWwg3l), NeurIPS 2022.
- [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913), Arxiv.

### Sample adaptation

### - Normalization-based methods
- [Improving robustness against common corruptions by covariate shift adaptation](https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html), NeurIPS 2020.
- [Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift](https://arxiv.org/abs/2006.10963), Arxiv.
- [MetaNorm: Learning to Normalize Few-Shot Batches Across Domains](https://openreview.net/forum?id=9z_dNsC4B5t), ICLR 2021.
- [SITA: Single Image Test-time Adaptation](https://arxiv.org/abs/2112.02355), Arxiv.
- [Test-time Batch Normalization](https://arxiv.org/abs/2205.10210), Arxiv.
- [MixNorm: Test-Time Adaptation Through Online Normalization Estimation](https://arxiv.org/abs/2110.11478), Arxiv.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.

### - Generation-based methods
- [Neural Networks with Recurrent Generative Feedback](https://proceedings.neurips.cc/paper/2020/hash/0660895c22f8a14eb039bfb9beb0778f-Abstract.html), NeurIPS 2020.
- [Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections](https://openaccess.thecvf.com/content/CVPR2021/html/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.html), CVPR 2021.
- [Back to the Source: Diffusion-Driven Test-Time Adaptation](https://arxiv.org/abs/2207.03442), Arxiv
- [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv), Openreview.

## How to adapt

### Training strategies

### - Single-source training
- [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](http://proceedings.mlr.press/v119/sun20b.html), ICML 2020.
- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c), ICLR 2021.
- [Bayesian Adaptation for Covariate Shift](https://proceedings.neurips.cc/paper/2021/hash/07ac7cd13fd0eb1654ccdbd222b81437-Abstract.html), NeurIPS 2021.
- [Tailoring: encoding inductive biases by optimizing unsupervised objectives at prediction time](https://proceedings.neurips.cc/paper/2021/hash/f45cc474bff52cb1b2268a2f94a2abcf-Abstract.html), NeurIPS 2021.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [Efficient Test-Time Model Adaptation without Forgetting](https://arxiv.org/abs/2204.02610), ICML 2022.
- [TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html), NeurIPS 2022.
- [Test Time Adaptation via Conjugate Pseudo-labels](https://openreview.net/forum?id=2yvUYc-YNUH), NeurIPS 2022.
- [MEMO: Test Time Robustness via Adaptation and Augmentation](https://arxiv.org/abs/2110.09506), Arxiv.
- [MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption](https://proceedings.mlr.press/v151/bartler22a.html), AISTATS 2022.
- [A Probabilistic Framework for Lifelong Test-Time Adaptation](https://arxiv.org/abs/2212.09713), Arxiv.
- [Improving robustness against common corruptions by covariate shift adaptation](https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html), NeurIPS 2020.
- [Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift](https://arxiv.org/abs/2006.10963), Arxiv.
- [SITA: Single Image Test-time Adaptation](https://arxiv.org/abs/2112.02355), Arxiv.
- [Test-time Batch Normalization](https://arxiv.org/abs/2205.10210), Arxiv.
- [MixNorm: Test-Time Adaptation Through Online Normalization Estimation](https://arxiv.org/abs/2110.11478), Arxiv.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [Back to the Source: Diffusion-Driven Test-Time Adaptation](https://arxiv.org/abs/2207.03442), Arxiv

### - Multi-source training
- [Adaptive Methods for Real-World Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.html?ref=https://githubhelp.com), CVPR 2021.
- [MetaNorm: Learning to Normalize Few-Shot Batches Across Domains](https://openreview.net/forum?id=9z_dNsC4B5t), ICLR 2021.
- [Learning to Generalize across Domains on Single Test Samples](https://openreview.net/forum?id=CIaQKbTBwtU), ICLR 2022.
- [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html), NeruIPS 2022.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections](https://openaccess.thecvf.com/content/CVPR2021/html/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.html), CVPR 2021.
- [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv), Openreview.

### Adaptation strategies

### - Online adaptation
- [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](http://proceedings.mlr.press/v119/sun20b.html), ICML 2020.
- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c), ICLR 2021.
- [Bayesian Adaptation for Covariate Shift](https://proceedings.neurips.cc/paper/2021/hash/07ac7cd13fd0eb1654ccdbd222b81437-Abstract.html), NeurIPS 2021.
- [Contrastive Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.html), CVPR 2022.
- [Efficient Test-Time Model Adaptation without Forgetting](https://arxiv.org/abs/2204.02610), ICML 2022.
- [TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html), NeurIPS 2022.
- [Test Time Adaptation via Conjugate Pseudo-labels](https://openreview.net/forum?id=2yvUYc-YNUH), NeurIPS 2022.
- [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html), NeruIPS 2022.

### - Batch-wise adaptation
- [Improving robustness against common corruptions by covariate shift adaptation](https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html), NeurIPS 2020.
- [Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift](https://arxiv.org/abs/2006.10963), Arxiv.
- [Test-time Batch Normalization](https://arxiv.org/abs/2205.10210), Arxiv.
- [MixNorm: Test-Time Adaptation Through Online Normalization Estimation](https://arxiv.org/abs/2110.11478), Arxiv.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [Adaptive Methods for Real-World Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.html?ref=https://githubhelp.com), CVPR 2021.


### - Sample-wise adaptation
With augmentation:
- [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](http://proceedings.mlr.press/v119/sun20b.html), ICML 2020.
- [SITA: Single Image Test-time Adaptation](https://arxiv.org/abs/2112.02355), Arxiv.
- [MEMO: Test Time Robustness via Adaptation and Augmentation](https://arxiv.org/abs/2110.09506), Arxiv.


Without augmentation:
- [MetaNorm: Learning to Normalize Few-Shot Batches Across Domains](https://openreview.net/forum?id=9z_dNsC4B5t), ICLR 2021.
- [Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections](https://openaccess.thecvf.com/content/CVPR2021/html/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.html), CVPR 2021.
- [Learning to Generalize across Domains on Single Test Samples](https://openreview.net/forum?id=CIaQKbTBwtU), ICLR 2022.
- [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913), Arxiv.
- [Back to the Source: Diffusion-Driven Test-Time Adaptation](https://arxiv.org/abs/2207.03442), Arxiv
- [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv), Openreview.

## Applications

### Medical imaging
- [Test-time Unsupervised Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_42), MICCAI 2020.
- [Fully Test-Time Adaptation for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24), MICCAI 2021.
- [Test-Time Adaptation with Shape Moments for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_70), MICCAI 2022.
- [Autoencoder based self-supervised test-time adaptation for medical image analysis](https://www.sciencedirect.com/science/article/pii/S1361841521001821), Medical Image Analysis, 2022.
- [On-the-Fly Test-time Adaptation for Medical Image Segmentation](https://arxiv.org/abs/2203.05574), Arxiv.
- [Single-domain Generalization in Medical Image Segmentation via Test-time Adaptation from Shape Dictionary](https://www.aaai.org/AAAI22Papers/AAAI-852.LiuQ.pdf), AAAI 2022.

### Image enhancement & restoration

### Segmentation
- [Fully Test-Time Adaptation for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24), MICCAI 2021.
- [On the Road to Online Adaptation for Semantic Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/html/Volpi_On_the_Road_to_Online_Adaptation_for_Semantic_Image_Segmentation_CVPR_2022_paper.html), CVPR 2022.
- [AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/html/Zhang_AuxAdapt_Stable_and_Efficient_Test-Time_Adaptation_for_Temporally_Consistent_Video_WACV_2022_paper.html), WACV 2022

### Video
- [Video Test-Time Adaptation for Action Recognition](https://arxiv.org/abs/2211.15393), Arxiv.
- [Self-Supervised Test-Time Adaptation on Video Data](https://openaccess.thecvf.com/content/WACV2022/html/Azimi_Self-Supervised_Test-Time_Adaptation_on_Video_Data_WACV_2022_paper.html), WACV 2022
- [AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/html/Zhang_AuxAdapt_Stable_and_Efficient_Test-Time_Adaptation_for_Temporally_Consistent_Video_WACV_2022_paper.html), WACV 2022

### Other applications

## Techniques

### Self-training

### Meta-learning

### Teacher-student network

## Datasets

## Related topics

Domain adaptation

Domain generalization ([papers](https://github.com/junkunyuan/Awesome-Domain-Generalization#theory--analysis))

Survey papers: 
- Generalizing to Unseen Domains: A Survey on Domain Generalization [[IJCAI 2021](https://arxiv.53yu.com/pdf/2103.03097)] [[Slides](http://jd92.wang/assets/files/DGSurvey-ppt.pdf)]
- Domain Generalization in Vision: A Survey [[TPAMI 2022](https://arxiv.org/abs/2103.02503)] 

Source-free domain adaptation ([papers](https://github.com/YuejiangLIU/awesome-source-free-test-time-adaptation))

Survey paper:
- Source-Free Unsupervised Domain Adaptation: A Survey [[arxiv](https://arxiv.org/pdf/2301.00265.pdf)]

