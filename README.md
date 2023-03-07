# Papers of Test-time adaptation

## What to adapt

### Model adaptation

### - Fine-tuning-based methods

Self-supervised (Auxillary model or loss during training)

- [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](http://proceedings.mlr.press/v119/sun20b.html), ICML 2020.
- [TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html), NeurIPS 2022.


Entropy minimization or pseudo labeling (without change training objective)

- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c), ICLR 2021.
- [If your data distribution shifts, use self-learning](https://arxiv.org/abs/2104.12928), Arxiv.
- [Bayesian Adaptation for Covariate Shift](https://proceedings.neurips.cc/paper/2021/hash/07ac7cd13fd0eb1654ccdbd222b81437-Abstract.html), NeurIPS 2021.
- [Tailoring: encoding inductive biases by optimizing unsupervised objectives at prediction time](https://proceedings.neurips.cc/paper/2021/hash/f45cc474bff52cb1b2268a2f94a2abcf-Abstract.html), NeurIPS 2021.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [Contrastive Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.html), CVPR 2022.

- [Efficient Test-Time Model Adaptation without Forgetting](https://arxiv.org/abs/2204.02610), ICML 2022.
- [Test-Time Training with Masked Autoencoders](https://openreview.net/group?id=ICLR.cc/2023/Conference/Authors&referrer=%5BHomepage%5D(%2F)), NeurIPS 2022.
- [Test Time Adaptation via Conjugate Pseudo-labels](https://openreview.net/forum?id=2yvUYc-YNUH), NeurIPS 2022.
- [MEMO: Test Time Robustness via Adaptation and Augmentation](https://openreview.net/forum?id=XrGEkCOREX2), NeurIPS 2022.
- [Revisiting Realistic Test-Time Training: Sequential Inference and Adaptation by Anchored Clustering](https://openreview.net/forum?id=W-_4hgRkwb), NeurIPS 2022. Sequential test-time training (sTTT): one pass test-time adaptation without change training objective; Iteratively update target cluster and do cluster alignment of source and target domain; inspired by semisupervised learning, introduce global feature alignment to avoid bad pseudo labels (also TTT++).
- [MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption](https://proceedings.mlr.press/v151/bartler22a.html), AISTATS 2022.
- [Towards Stable Test-time Adaptation in Dynamic Wild World](https://openreview.net/forum?id=g2YraF75Tj), ICLR 2023. Replace BN by IN and GN. To deal with the collapse of IN and GN, remove the samples with large gradients based on the entropy; and use ahrpness aware cross-entropy loss.


More insight of the fine-tuning based method.

- [Uncovering Adversarial Risks of Test-Time Adaptation](https://arxiv.org/abs/2301.12576), Arxiv. Find that test-time adaptation is vulnerable to malicious data at test time in contrast to conventional machine learning and propose a new attack for test-time adaptation.

Other variants

- [Continual Test-Time Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html), CVPR 2022. Current self-training methods is effective when the test data are drawn from the same stationary domain, but unstable in the countinually changing environment. (Also in stable TTA in ICLR 2023). Error accumulation and catastrophic forgetting (ICML 2022); Use weighted averaged and augmentation averaged predictions for better pseudo labels against error accumulation; stochastically small parts of source model parameters storage for forgetting.

- [A Probabilistic Framework for Lifelong Test-Time Adaptation](https://arxiv.org/abs/2212.09713), Arxiv.

- [Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models](https://openreview.net/forum?id=e8PVEkSa4Fq), NeurIPS 2022. Boost generalization in zero-shot manner. Fine-tune prompts at test time, with entropy minimization and confidence selection (only high confident ones) on the augmented single test sample.

### - Prototype-based/context-based methods (Function-inferring methods.)
Efficiently infer the functions/neural network parameters by a single forward pass.
- [Adaptive Methods for Real-World Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.html?ref=https://githubhelp.com), CVPR 2021.
- [Learning to Generalize across Domains on Single Test Samples](https://openreview.net/forum?id=CIaQKbTBwtU), ICLR 2022.
- [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html), NeruIPS 2022. (?)
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [Extrapolative Continuous-time Bayesian Neural Network for Fast Training-free Test-time Adaptation](https://openreview.net/forum?id=wiHzQWwg3l), NeurIPS 2022. 
Time-evolving neural network parameters; Continous time BNN; Infer the BNN parameters by particle filter differential equation (PFDE) and neural networks. The prior distribution is generated by the time step t and the previous data (B_{t-1}), the prior distribution is generated by the current batch data.
- [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913), Arxiv.

### Sample adaptation

### - Normalization-based methods
- [Revisiting Batch Normalization For Practical Domain Adaptation](https://openreview.net/forum?id=Hk6dkJQFx), ICLR 2017 Workshop. Replacing source BN statistics to target ones, but not at test time. 
- [Improving robustness against common corruptions by covariate shift adaptation](https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html), NeurIPS 2020.
- [Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift](https://arxiv.org/abs/2006.10963), Arxiv.
- [MetaNorm: Learning to Normalize Few-Shot Batches Across Domains](https://openreview.net/forum?id=9z_dNsC4B5t), ICLR 2021.
- [SITA: Single Image Test-time Adaptation](https://arxiv.org/abs/2112.02355), Arxiv.
- [Test-time Batch Normalization](https://arxiv.org/abs/2205.10210), Arxiv.
- [MixNorm: Test-Time Adaptation Through Online Normalization Estimation](https://arxiv.org/abs/2110.11478), Arxiv.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [Domain-Conditioned Normalization for Test-Time Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-031-25085-9_17), ECCV 2022 Workshop. Similar to MetaNorm, but inferring both BN statistics and rescaling parameters.
- [NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation](https://openreview.net/forum?id=E9HNxrCFZPV), NeurIPS 2022. Handle temporally correlated test-time samples; Instance-aware batch normalization. Calculate the difference between the previous BN statistics and the IN statistics of each sample to determine whether use BN or IN; only one forward pass with each single sample for test-time adaptation; Prediction-balanced reservoir sampling for mimicking iid samples from non-iid streams.
- [TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation](https://openreview.net/forum?id=EQfeudmWLQ), ICLR 2023. Propose post-training stage to learn the coefficient for combining the source BN statistics and target BN statistics. The coefficient is initialized by a proposed gradient distance score. (measure the domain-shift sensitivity by comparing gradients.)

### - Generation-based methods
- [Neural Networks with Recurrent Generative Feedback](https://proceedings.neurips.cc/paper/2020/hash/0660895c22f8a14eb039bfb9beb0778f-Abstract.html), NeurIPS 2020.
- [Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections](https://openaccess.thecvf.com/content/CVPR2021/html/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.html), CVPR 2021.
- [Back to the Source: Diffusion-Driven Test-Time Adaptation](https://arxiv.org/abs/2207.03442), CVPR 2023.
- [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv), ICLR 2023.

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
- [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv), ICLR 2023.

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
- [Back to the Source: Diffusion-Driven Test-Time Adaptation](https://arxiv.org/abs/2207.03442), CVPR 2023
- [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv), ICLR 2023.

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

### DeepFake detection
-[OST: Improving Generalization of DeepFake Detection via One-Shot Test-Time Training](https://openreview.net/forum?id=YPoRoad6gzY), NeurIPS 2022. Meta-learning; MAML; Generate pseudo training data for one-shot training at test time.

### Pose estimation
-[Test-Time Personalization with a Transformer for Human Pose Estimation](https://proceedings.neurips.cc/paper/2021/hash/1517c8664be296f0d87d9e5fc54fdd60-Abstract.html), NeurIPS 2021. Adapt pose estimator at test time to exploit person-specific information; TTT manner, supervised and self-supervised loss; use transformer to do transformation betrween supervised keypoints and self-supervised keypoints.

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

