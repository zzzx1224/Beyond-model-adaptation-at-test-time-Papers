# Papers of Test-time adaptation

## What to adapt

### Model adaptation

### - Model Fine-tuning

Test-time training

- [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](http://proceedings.mlr.press/v119/sun20b.html), ICML 2020. First test-time training; auxilliary self-supervised loss at training and test time; Online & offline one-shot
- [Test-time Unsupervised Domain Adaptation](https://arxiv.org/abs/2010.01926), MICCAI 2020. Adversarial loss and augmentation consistency in "one-shot" case; medical image; MRI.
- [Test-Time Personalization with a Transformer for Human Pose Estimation](https://proceedings.neurips.cc/paper/2021/hash/1517c8664be296f0d87d9e5fc54fdd60-Abstract.html), NeurIPS 2021. Adapt pose estimator at test time to exploit person-specific information; TTT manner, supervised and self-supervised loss; use transformer to do transformation betrween supervised keypoints and self-supervised keypoints.
- [TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html), NeurIPS 2022. When TTT fails? overfitting to the auxiliary task. When TTT Thrives? The self-supervised task is closely related the main task. Replace rotation prediction with SimCLR; Introduce online feature alignment.
- [Test-Time Training with Masked Autoencoders](https://openreview.net/forum?id=SHMi1b7sjXk), NeurIPS 2022. Optimizing a model for each test input using self-supervision through masked autoencoders.
- [MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption](https://proceedings.mlr.press/v151/bartler22a.html), AISTATS 2022. Meta TTT; BYOL as the self-supervised loss.
- [Towards Multi-domain Single Image Dehazing via Test-time Training](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Towards_Multi-Domain_Single_Image_Dehazing_via_Test-Time_Training_CVPR_2022_paper.html), CVPR 2022. Test-time training; Helper network to evaluate dehazing quality and adjust model parameters via self-supervision; Meta-learning to make the objectives of the dehazing and helper networks consistent with each other.


Fully test-time adaptation

- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c), ICLR 2021.
- [Bayesian Adaptation for Covariate Shift](https://proceedings.neurips.cc/paper/2021/hash/07ac7cd13fd0eb1654ccdbd222b81437-Abstract.html), NeurIPS 2021. Probabilistic modeling for covariate shift and bayesian adaptation.


Update different parts of parameters:

- [Deep Matching Prior: Test-Time Optimization for Dense Correspondence](https://openaccess.thecvf.com/content/ICCV2021/html/Hong_Deep_Matching_Prior_Test-Time_Optimization_for_Dense_Correspondence_ICCV_2021_paper.html), ICCV 2021. Test-time optimization for image-pair-specific prior of the matching network.
- [Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models](https://openreview.net/forum?id=e8PVEkSa4Fq), NeurIPS 2022. Boost generalization in zero-shot manner. Fine-tune prompts at test time, with entropy minimization and confidence selection (only high confident ones) on the augmented single test sample.
- [Surgical Fine-Tuning Improves Adaptation to Distribution Shifts](https://openreview.net/forum?id=APuPRxjHvZ), ICLR 2023. Tune only one block of parameters and freeze the remaining parameters, outperforms full fine-tuning on a range of distribution shifts; Tuning different blocks performs best for different types of distribution shifts. Theoretically, we prove that for two-layer neural networks in an idealized setting, first-layer tuning can outperform fine-tuning all layers. Intuitively, fine-tuning more parameters on a small target dataset can cause information learned during pre-training to be forgotten, and the relevant information depends on the type of shift. Propose automatically finding an adequate subset of layers to perform surgical fine-tuning on.
- [Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation](https://arxiv.org/abs/2303.00914), Arxiv. Fine-tune the first convoutional layer by bottom-up feed-forward adaptation process, Hebbian learning and fine-tune the neuromodulator by top-down feedback-based optimization like entropy minimization.

Investigation on loss functions.
- [Fully Test-Time Adaptation for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24), MICCAI 2021. Like tent, fine-tune by Regional nuclear-norm loss and Contour regularization loss.
- [Test time Adaptation through Perturbation Robustness](https://openreview.net/forum?id=GbBeI5z86uD), NeurIPS 2021, Workshop. Fully TTA; data augmentation consistency loss & entropy minimization.
- [Efficient Test-Time Model Adaptation without Forgetting](https://arxiv.org/abs/2204.02610), ICML 2022. Adaptive entropy minimization; adaptation on low-entropy samples makes more contribution than high- entropy ones; adaptation on test samples with very high entropy may hurt performance; Select low-entropy (high confident) samples; Anti-forgetting with Fisher Regularization.
- [Improving Test-Time Adaptation via Shift-agnostic Weight Regularization and Nearest Source Prototypes](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_26), ECCV 2022. Learn layer-wise penalty by the gradient similarity of original and transformed source data; Use the layer-wise weight penalty to less update the shift-agnostic weights; use nearest source prototypes as an auxilliary loss for adaptation.
- [Revisiting Realistic Test-Time Training: Sequential Inference and Adaptation by Anchored Clustering](https://openreview.net/forum?id=W-_4hgRkwb), NeurIPS 2022. Sequential test-time training (sTTT): one pass test-time adaptation without change training objective; Iteratively update target cluster and do cluster alignment of source and target domain; inspired by semisupervised learning, introduce global feature alignment to avoid bad pseudo labels (also TTT++).
- [Test Time Adaptation via Conjugate Pseudo-labels](https://openreview.net/forum?id=2yvUYc-YNUH), NeurIPS 2022. Reformulate the loss function of test-time adaptation. Not only entropy minimization.

Pseudo labels (Accumulated errors from imperfect pseudo labels)

- [If your data distribution shifts, use self-learning](https://arxiv.org/abs/2104.12928), Arxiv. Hard pseudo-labling; Soft pseudo labeling (Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks; Empirical comparison of hard and soft label propagation for
relational classification); Entropy minimization; Robust pseudo-labeling.
- [On-target Adaptation](https://arxiv.org/abs/2109.01087), Arxiv. Learn the representation purely from target the while taking only the source predictions for supervision. Several stages; First adapt teacher model by InfoMax; then initialize target (student) model by contrastive learning from scratch; use teacher model to generate pseudo labels to fine-tune student model; Replace the teacher model by the latest student model to eliminate the accumulated errors from imperfect pseudo labels.
- [TeST: Test-time Self-Training under Distribution Shift](https://openaccess.thecvf.com/content/WACV2023/html/Sinha_TeST_Test-Time_Self-Training_Under_Distribution_Shift_WACV_2023_paper.html), WACV 2023. Several stages; Intialize teacher and student model by source-trained model; Train teacher network by self-supervised contrastive learning; Train student network by pseudo label from teacher network and entropy minimization.
- [Contrastive Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.html), CVPR 2022. Online pseudo label refinement with self-supervised contrastive learning.
- [Test-Time Adaptation via Self-Training with Nearest Neighbor Information](https://openreview.net/forum?id=EzLtB4M1SbM), ICLR 2023. Generate pseudo labels using the nearest neighbors from a set composed of previous test data; fine-tune the trained classifier with the pseudo labels.
- [Guiding Pseudo-labels with Uncertainty Estimation for Test-Time Adaptation](https://arxiv.org/abs/2303.03770), CVPR 2023. Reweight the loss based on the uncertainty (reliability) of the pseudo-labels; Refine pseudo labels by aggragating knoledge from neighboring samples.

Problems of Tent (single sample; forgetting; not stable).
- [MEMO: Test Time Robustness via Adaptation and Augmentation](https://openreview.net/forum?id=XrGEkCOREX2), NeurIPS 2022. Marginal Entropy minimization; Perform different data augmentations on each single test sample and then adapt (all of) the model parameters by minimizing the entropy of the model’s marginal output distribution across the augmentations. (TTT, prompt tuning also use augmentation)
- [Efficient Test-Time Model Adaptation without Forgetting](https://arxiv.org/abs/2204.02610), ICML 2022. Adaptive entropy minimization; adaptation on low-entropy samples makes more contribution than high- entropy ones; adaptation on test samples with very high entropy may hurt performance; Select low-entropy (high confident) samples; Anti-forgetting with Fisher Regularization.
- [Towards Stable Test-time Adaptation in Dynamic Wild World](https://openreview.net/forum?id=g2YraF75Tj), ICLR 2023. Replace BN by IN and GN. To deal with the collapse of IN and GN, remove the samples with large gradients based on the entropy; and use sharpness aware cross-entropy loss.

(Continual TTA (forgetting, not stable):)

- [Continual Test-Time Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html), CVPR 2022. Current self-training methods is effective when the test data are drawn from the same stationary domain, but unstable in the countinually changing environment. (Also in stable TTA in ICLR 2023). Error accumulation and catastrophic forgetting (ICML 2022); Use weighted averaged and augmentation averaged predictions for better pseudo labels against error accumulation; stochastically small parts of source model parameters storage for forgetting.

- [A Probabilistic Framework for Lifelong Test-Time Adaptation](https://arxiv.org/abs/2212.09713), Arxiv. Bayesian adaptation for continual test-time adaptation; with moving average of student to teacher (like continue TTA); Fisher information matrix based data-driven parameter restoration.


- [Uncovering Adversarial Risks of Test-Time Adaptation](https://arxiv.org/abs/2301.12576), Arxiv. Find that test-time adaptation is vulnerable to malicious data at test time in contrast to conventional machine learning and propose a new attack for test-time adaptation.

Meta-learning
- [MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption](https://proceedings.mlr.press/v151/bartler22a.html), AISTATS 2022. Meta TTT; BYOL as the self-supervised loss.
- [Tailoring: encoding inductive biases by optimizing unsupervised objectives at prediction time](https://proceedings.neurips.cc/paper/2021/hash/f45cc474bff52cb1b2268a2f94a2abcf-Abstract.html), NeurIPS 2021. Adapt model with auxiliary losses on each test sample; meta-tailoring; Train the affine parameters of conditional normalization at inner loop; Train other parameters at outer loop.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021. meta test-time training.
- [Test-Time Fast Adaptation for Dynamic Scene Deblurring via Meta-Auxiliary Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Chi_Test-Time_Fast_Adaptation_for_Dynamic_Scene_Deblurring_via_Meta-Auxiliary_Learning_CVPR_2021_paper.html), CVPR 2021. Meta-learn; image reconstruction as the auxiliary task; adapt on each single image.


### - Model/Function-inferring methods, optimization free)
Efficiently infer the functions/neural network parameters by a single forward pass. (Online or adapt to each sample)
- [Adaptive Methods for Real-World Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.html?ref=https://githubhelp.com), CVPR 2021.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021. Like dubey et al.
- [Learning to Generalize across Domains on Single Test Samples](https://openreview.net/forum?id=CIaQKbTBwtU), ICLR 2022.
- [Variational On-the-Fly Personalization](https://proceedings.mlr.press/v162/kim22e.html), ICML 2022. Estimate model weights on-the-fly based on the personality of a small amount of personal data, through a variational hyper-personalizer.
- [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html), NeruIPS 2022. Online update/adjust the classifier with pseudo labels generated by previous test data features; Filter unreliable pseudo-labeled data by prediction entropy. 
- [Extrapolative Continuous-time Bayesian Neural Network for Fast Training-free Test-time Adaptation](https://openreview.net/forum?id=wiHzQWwg3l), NeurIPS 2022. Time-evolving neural network parameters; Continous time BNN; Infer the BNN parameters by particle filter differential equation (PFDE) and neural networks. The prior distribution is generated by the time step t and the previous data (B_{t-1}), the prior distribution is generated by the current batch data.
- [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913), Arxiv. Meta adjuster; adjusting the model without training.
- [Domain Prompt Learning for Efficiently Adapting CLIP to Unseen Domains](http://128.84.21.203/abs/2111.12853), Arxiv. Generate domain prompt according to the domain representation.

### Sample adaptation

### - Normalization-based methods
- [Revisiting Batch Normalization For Practical Domain Adaptation](https://openreview.net/forum?id=Hk6dkJQFx), ICLR 2017 Workshop. Replacing source BN statistics to target ones, but not at test time. 


Directly utilize target statistics.

- [Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift](https://arxiv.org/abs/2006.10963), Arxiv. Prediction-time BN, recompute statistics for each test batch.
- [Be Like Water: Robustness to Extraneous Variables Via Adaptive Feature Normalization](https://arxiv.org/abs/2002.04019), Arxiv. Adaptive Norm. Use test statistics or instance norm statistics at test time.
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021. Training batches are from each single domain; recompute statistics at test time.

Combine source and target statistics.
- [Improving robustness against common corruptions by covariate shift adaptation](https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html), NeurIPS 2020. Correct the BN statistics by the target statistics; weighted sum of the source and target statistics.
- [Test-time Batch Statistics Calibration for Covariate Shift](https://arxiv.org/abs/2110.04065), Arxiv. α-BN to calibrate the batch statistics by mixing up the source and target statistics for both alleviating the domain shift and preserving the discriminative structures.
- [SITA: Single Image Test-time Adaptation](https://arxiv.org/abs/2112.02355), Arxiv. Combine source and target BN statistics by weighted sum; Get target BN statistics with augmentation of single sample.
- [TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation](https://openreview.net/forum?id=EQfeudmWLQ), ICLR 2023. Propose post-training stage to learn the coefficient for combining the source BN statistics and target BN statistics. The coefficient is initialized by a proposed gradient distance score. (measure the domain-shift sensitivity by comparing gradients.)

(online update)
- [Test-time Batch Normalization](https://arxiv.org/abs/2205.10210), Arxiv. Gradient preserving batch norm. Moving average of source and target statistics.
- [MixNorm: Test-Time Adaptation Through Online Normalization Estimation](https://arxiv.org/abs/2110.11478), Arxiv. Global statistics: initialized by training statistics and updated by the statistics on each new sample; Local statistics: statistics of the augmented target samples; Mix global and local statistics.
- [The Norm Must Go On: Dynamic Unsupervised Domain Adaptation by Normalization](https://openaccess.thecvf.com/content/CVPR2022/html/Mirza_The_Norm_Must_Go_On_Dynamic_Unsupervised_Domain_Adaptation_by_CVPR_2022_paper.html), CVPR 2022. Dynamically update the BN statistics; Adaptive momentum for updating the statistics, mean and variance. Single sample with augmentation.

(selectively update)
- [NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation](https://openreview.net/forum?id=E9HNxrCFZPV), NeurIPS 2022. Handle temporally correlated test-time samples; Instance-aware batch normalization. Calculate the difference between the previous BN statistics and the IN statistics of each sample to determine whether use BN or IN; only one forward pass with each single sample for test-time adaptation; Prediction-balanced reservoir sampling for mimicking iid samples from non-iid streams.
- [GANs Spatial Control via Inference-Time Adaptive Normalization](https://openaccess.thecvf.com/content/WACV2022/html/Jakoel_GANs_Spatial_Control_via_Inference-Time_Adaptive_Normalization_WACV_2022_paper.html), WACV 2022. Inference time adaptive normalization; apply different normalizations at different region of the image for spatial control.

Infer target statistics.

- [MetaNorm: Learning to Normalize Few-Shot Batches Across Domains](https://openreview.net/forum?id=9z_dNsC4B5t), ICLR 2021.
- [Domain-Conditioned Normalization for Test-Time Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-031-25085-9_17), ECCV 2022 Workshop. Similar to MetaNorm, but inferring both BN statistics and rescaling parameters.

### - Generation-based methods
- [Neural Networks with Recurrent Generative Feedback](https://proceedings.neurips.cc/paper/2020/hash/0660895c22f8a14eb039bfb9beb0778f-Abstract.html), NeurIPS 2020. The first to do recurrent adaptation/update/generation for robust object recognition. (Use the motivation in this paper.)
- [Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections](https://openaccess.thecvf.com/content/CVPR2021/html/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.html), CVPR 2021.
- [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv), ICLR 2023.
- [Back to the Source: Diffusion-Driven Test-Time Adaptation](https://arxiv.org/abs/2207.03442), CVPR 2023. Diffusion model for sample adaptation. Good motivation and intuition.

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
- [Fully Test-Time Adaptation for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24), MICCAI 2021. Like tent
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
- [OST: Improving Generalization of DeepFake Detection via One-Shot Test-Time Training](https://openreview.net/forum?id=YPoRoad6gzY), NeurIPS 2022. Meta-learning; MAML; Generate pseudo training data for one-shot training at test time.

### Pose estimation
- [Test-Time Personalization with a Transformer for Human Pose Estimation](https://proceedings.neurips.cc/paper/2021/hash/1517c8664be296f0d87d9e5fc54fdd60-Abstract.html), NeurIPS 2021. Adapt pose estimator at test time to exploit person-specific information; TTT manner, supervised and self-supervised loss; use transformer to do transformation betrween supervised keypoints and self-supervised keypoints.

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

