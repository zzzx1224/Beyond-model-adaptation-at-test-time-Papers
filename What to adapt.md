## What to adapt

We categorize the test-time adaptation methods into five main categories: **model adaptation**, **inference adaptation**, **normalization adaptation**, **sample adaptation**, or **prompt adaptation**.
Related literature and links are in the following:

### Model Adaptation

Auxiliary Self-supervision

- [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](http://proceedings.mlr.press/v119/sun20b.html), ICML 2020. 
- [Test-time Unsupervised Domain Adaptation](https://arxiv.org/abs/2010.01926), MICCAI 2020.
- [Tailoring: encoding inductive biases by optimizing unsupervised objectives at prediction time](https://proceedings.neurips.cc/paper/2021/hash/f45cc474bff52cb1b2268a2f94a2abcf-Abstract.html), NeurIPS 2021.
- [Test-Time Personalization with a Transformer for Human Pose Estimation](https://proceedings.neurips.cc/paper/2021/hash/1517c8664be296f0d87d9e5fc54fdd60-Abstract.html), NeurIPS 2021.
- [Sketch3T: Test-Time Training for Zero-Shot SBIR](https://openaccess.thecvf.com/content/CVPR2022/html/Sain_Sketch3T_Test-Time_Training_for_Zero-Shot_SBIR_CVPR_2022_paper.html), CVPR 2022.
- [Towards Multi-domain Single Image Dehazing via Test-time Training](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Towards_Multi-Domain_Single_Image_Dehazing_via_Test-Time_Training_CVPR_2022_paper.html), CVPR 2022.
- [TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html), NeurIPS 2022.
- [OST: Improving Generalization of DeepFake Detection via One-Shot Test-Time Training](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html), NeurIPS 2022.
- [Test-Time Training with Masked Autoencoders](https://openreview.net/forum?id=SHMi1b7sjXk), NeurIPS 2022. 
- [MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption](https://proceedings.mlr.press/v151/bartler22a.html), AISTATS 2022. 
- [ActMAD: Activation Matching to Align Distributions for Test-Time-Training](https://openaccess.thecvf.com/content/CVPR2023/html/Mirza_ActMAD_Activation_Matching_To_Align_Distributions_for_Test-Time-Training_CVPR_2023_paper.html), CVPR 2023.
- [MATE: Masked Autoencoders are Online 3D Test-Time Learners](https://openaccess.thecvf.com/content/ICCV2023/html/Mirza_MATE_Masked_Autoencoders_are_Online_3D_Test-Time_Learners_ICCV_2023_paper.html), ICCV 2023.
- [ClusT3: Information Invariant Test-Time Training](https://openaccess.thecvf.com/content/ICCV2023/html/Hakim_ClusT3_Information_Invariant_Test-Time_Training_ICCV_2023_paper.html), ICCV 2023.
- [On the Robustness of Open-World Test-Time Training: Self-Training with Dynamic Prototype Expansion](https://openaccess.thecvf.com/content/ICCV2023/html/Li_On_the_Robustness_of_Open-World_Test-Time_Training_Self-Training_with_Dynamic_ICCV_2023_paper.html), ICCV 2023.
- [Diffusion-TTA: Test-time Adaptation of Discriminative Models via Generative Feedback](https://openreview.net/forum?id=gUTVpByfVX), NeurIPS 2023.

Entropy Minimization

- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c), ICLR 2021.
- [Fully Test-Time Adaptation for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24), MICCAI 2021.
- [Bayesian Adaptation for Covariate Shift](https://proceedings.neurips.cc/paper/2021/hash/07ac7cd13fd0eb1654ccdbd222b81437-Abstract.html), NeurIPS 2021.
- [Efficient Test-Time Model Adaptation without Forgetting](https://proceedings.mlr.press/v162/niu22a.html), ICML 2022.
- [Improving Test-Time Adaptation via Shift-agnostic Weight Regularization and Nearest Source Prototypes](https://arxiv.org/pdf/2207.11707), ECCV 2022.
- [MEMO: Test Time Robustness via Adaptation and Augmentation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fc28053a08f59fccb48b11f2e31e81c7-Abstract-Conference.html), NeurIPS 2022.
- [Surgical Fine-Tuning Improves Adaptation to Distribution Shifts](https://openreview.net/forum?id=APuPRxjHvZ), ICLR 2023.
- [Towards Stable Test-Time Adaptation in Dynamic Wild World](https://openreview.net/forum?id=g2YraF75Tj), ICLR 2023.
- [Delta: Degradation-free Fully Test-time Adaptation](https://openreview.net/forum?id=eGm22rqG93), ICLR 2023.
- [Improved Test-Time Adaptation for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Improved_Test-Time_Adaptation_for_Domain_Generalization_CVPR_2023_paper.html), CVPR 2023.
- [EcoTTA: Memory-Efficient Continual Test-time Adaptation via Self-distilled Regularization](https://arxiv.org/abs/2303.01904), CVPR 2023.
- [Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation](https://arxiv.org/abs/2303.00914), CVPR 2023.
- [Towards Open-Set Test-Time Adaptation Utilizing the Wisdom of Crowds in Entropy Minimization](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Towards_Open-Set_Test-Time_Adaptation_Utilizing_the_Wisdom_of_Crowds_in_ICCV_2023_paper.pdf), ICCV 2023.
- [DomainAdaptor: A Novel Approach to Test-time Adaptation](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_DomainAdaptor_A_Novel_Approach_to_Test-time_Adaptation_ICCV_2023_paper.html), ICCV 2023.
- [SoTTA: Robust Test-Time Adaptation on Noisy Data Streams](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2da53cd1abdae59150e35f4693834f32-Abstract-Conference.html), NeurIPS 2023.
- [Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors](https://openreview.net/forum?id=9w3iw8wDuE), ICLR 2024.
- [TEA: Test-time Energy Adaptation](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_TEA_Test-time_Energy_Adaptation_CVPR_2024_paper.pdf), CVPR 2024.
- [Unified Entropy Optimization for Open-Set Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Unified_Entropy_Optimization_for_Open-Set_Test-Time_Adaptation_CVPR_2024_paper.html), CVPR 2024.
- [Stationary Latent Weight Inference for Unreliable Observations from Online Test-Time Adaptation](https://openreview.net/pdf?id=HmKMpJXH67), ICML 2024.

Pseudo-labeling

- [If your data distribution shifts, use self-learning](https://arxiv.org/abs/2104.12928), TMLR 2021.
- [On-target Adaptation](https://arxiv.org/abs/2109.01087), arXiv 2021.
- [Continual Test-Time Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html), CVPR 2022.
- [Contrastive Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.html), CVPR 2022.
- [Test Time Adaptation via Conjugate Pseudo-labels](https://proceedings.neurips.cc/paper_files/paper/2022/hash/28e9eff897f98372409b40ae1ed3ea4c-Abstract-Conference.html), NeurIPS 2022.
- [TeST: Test-time Self-Training under Distribution Shift](https://openaccess.thecvf.com/content/WACV2023/html/Sinha_TeST_Test-Time_Self-Training_Under_Distribution_Shift_WACV_2023_paper.html), WACV 2023.
- [Test-Time Adaptation via Self-Training with Nearest Neighbor Information](https://openreview.net/forum?id=EzLtB4M1SbM), ICLR 2023.
- [Train/Test-Time Adaptation With Retrieval](https://openaccess.thecvf.com/content/CVPR2023/html/Zancato_TrainTest-Time_Adaptation_With_Retrieval_CVPR_2023_paper.html), CVPR 2023.
- [TeSLA: Test-Time Self-Learning With Automatic Adversarial Augmentation](https://openaccess.thecvf.com/content/CVPR2023/html/Tomar_TeSLA_Test-Time_Self-Learning_With_Automatic_Adversarial_Augmentation_CVPR_2023_paper.html), CVPR 2023.
- [Guiding Pseudo-labels with Uncertainty Estimation for Test-Time Adaptation](https://arxiv.org/abs/2303.03770), CVPR 2023.
- [Probabilistic Test-Time Generalization by Variational Neighbor-Labeling](https://arxiv.org/abs/2307.04033), CoLLAs 2024.
- [SODA: Robust Training of Test-Time Data Adaptors](https://proceedings.neurips.cc/paper_files/paper/2023/hash/893ca2e5ff5bb258da30e0a82f4c8de9-Abstract-Conference.html), NeurIPS 2023.
- [PROGRAM: PROtotype GRAph Model based Pseudo-Label Learning for Test-Time Adaptation](https://openreview.net/forum?id=x5LvBK43wg), ICLR 2024.

Feature Alignment

- [Test time Adaptation through Perturbation Robustness](https://openreview.net/forum?id=GbBeI5z86uD), NeurIPS workshop 2021.
- [Revisiting Realistic Test-Time Training: Sequential Inference and Adaptation by Anchored Clustering](https://openreview.net/forum?id=W-_4hgRkwb), NeurIPS 2022.
- [Video Test-Time Adaptation for Action Recognition](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Video_Test-Time_Adaptation_for_Action_Recognition_CVPR_2023_paper.html), CVPR 2023.
- [Feature Alignment and Uniformity for Test Time Adaptation](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Feature_Alignment_and_Uniformity_for_Test_Time_Adaptation_CVPR_2023_paper.html), CVPR 2023.
- [TIPI: Test Time Adaptation With Transformation Invariance](https://openaccess.thecvf.com/content/CVPR2023/html/Nguyen_TIPI_Test_Time_Adaptation_With_Transformation_Invariance_CVPR_2023_paper.html), CVPR 2023.
- [CAFA: Class-Aware Feature Alignment for Test-Time Adaptation](https://openaccess.thecvf.com/content/ICCV2023/html/Jung_CAFA_Class-Aware_Feature_Alignment_for_Test-Time_Adaptation_ICCV_2023_paper.html), ICCV 2023.
- [Leveraging Proxy of Training Data for Test-Time Adaptation](https://proceedings.mlr.press/v202/kang23a.html), ICML 2023.


### Inference Adaptation

Batch-wise

- [Adaptive Methods for Real-World Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.html?ref=https://githubhelp.com), CVPR 2021.
- [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html), NeruIPS 2022.
- [Extrapolative Continuous-time Bayesian Neural Network for Fast Training-free Test-time Adaptation](https://openreview.net/forum?id=wiHzQWwg3l), NeurIPS 2022.
- [AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation](https://proceedings.mlr.press/v202/zhang23am.html), ICML 2023.
- [Test-Time Distribution Normalization for Contrastively Learned Vision-language Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/931db0b5a61f9db6c97c7e4bf068147d-Paper-Conference.pdf), NeurIPS 2023.

Sample-wise

- [Learning to Generalize across Domains on Single Test Samples](https://openreview.net/forum?id=CIaQKbTBwtU), ICLR 2022.
- [Variational On-the-Fly Personalization](https://proceedings.mlr.press/v162/kim22e.html), ICML 2022.
- [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913), arXiv 2022.
- [Dynamically Instance-Guided Adaptation: A Backward-Free Approach for Test-Time Domain Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Dynamically_Instance-Guided_Adaptation_A_Backward-Free_Approach_for_Test-Time_Domain_Adaptive_CVPR_2023_paper.html), CVPR 2023.
- [Towards Instance-adaptive Inference for Federated Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_Towards_Instance-adaptive_Inference_for_Federated_Learning_ICCV_2023_paper.html), ICCV 2023.
- [Rapid Network Adaptation: Learning to Adapt Neural Networks Using Test-Time Feedback](https://openaccess.thecvf.com/content/ICCV2023/html/Yeo_Rapid_Network_Adaptation_Learning_to_Adapt_Neural_Networks_Using_Test-Time_ICCV_2023_paper.html), ICCV 2023.



### Normalization Adaptation

Target statistics.

- [Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift](https://arxiv.org/abs/2006.10963), arXiv 2020.
- [Be Like Water: Robustness to Extraneous Variables Via Adaptive Feature Normalization](https://arxiv.org/abs/2002.04019), arXiv 2020. 
- [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), NeurIPS 2021.
- [NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ae6c7dbd9429b3a75c41b5fb47e57c9e-Abstract-Conference.html), NeurIPS 2022.
- [Delta: Degradation-free Fully Test-time Adaptation](https://openreview.net/forum?id=eGm22rqG93), ICLR 2023.

Statistics combination

- [Improving robustness against common corruptions by covariate shift adaptation](https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html), NeurIPS 2020. 
- [Test-time Batch Statistics Calibration for Covariate Shift](https://arxiv.org/abs/2110.04065), arXiv 2021.
- [SITA: Single Image Test-time Adaptation](https://arxiv.org/abs/2112.02355), arXiv 2021.
- [MixNorm: Test-Time Adaptation Through Online Normalization Estimation](https://arxiv.org/abs/2110.11478), arXiv 2021.
- [Test-time Batch Normalization](https://arxiv.org/abs/2205.10210), arXiv 2022.
- [The Norm Must Go On: Dynamic Unsupervised Domain Adaptation by Normalization](https://openaccess.thecvf.com/content/CVPR2022/html/Mirza_The_Norm_Must_Go_On_Dynamic_Unsupervised_Domain_Adaptation_by_CVPR_2022_paper.html), CVPR 2022.
- [TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation](https://openreview.net/forum?id=EQfeudmWLQ), ICLR 2023.
- [Generalized Lightness Adaptation with Channel Selective Normalization](https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Generalized_Lightness_Adaptation_with_Channel_Selective_Normalization_ICCV_2023_paper.html), ICCV 2023.
- [Test-Time Domain Generalization for Face Anti-Spoofing](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Test-Time_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2024_paper.html), CVPR 2024.

Statistics inference.

- [MetaNorm: Learning to Normalize Few-Shot Batches Across Domains](https://openreview.net/forum?id=9z_dNsC4B5t), ICLR 2021.
- [Domain-Conditioned Normalization for Test-Time Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-031-25085-9_17), ECCV 2022.


### Sample adaptation

Feature adjustment

- [Neural Networks with Recurrent Generative Feedback](https://proceedings.neurips.cc/paper/2020/hash/0660895c22f8a14eb039bfb9beb0778f-Abstract.html), NeurIPS 2020.
- [Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections](https://openaccess.thecvf.com/content/CVPR2021/html/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.html), CVPR 2021.
- [Diffusion Models for Adversarial Purification](https://arxiv.org/abs/2205.07460), ICML 2022. 
- [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv), ICLR 2023.
- [Test-Time Style Shifting: Handling Arbitrary Styles in Domain Generalization](https://proceedings.mlr.press/v202/park23d.html), ICML 2023.
- [A-STAR: Test-time Attention Segregation and Retention for Text-to-image Synthesis](https://openaccess.thecvf.com/content/ICCV2023/html/Agarwal_A-STAR_Test-time_Attention_Segregation_and_Retention_for_Text-to-image_Synthesis_ICCV_2023_paper.html), ICCV 2023.

Input adjustment

- [Back to the Source: Diffusion-Driven Test-Time Adaptation](https://arxiv.org/abs/2207.03442), CVPR 2023.

### Prompt adaptation

Text-prompting

- [TEMPERA: Test-Time Prompt Editing via Reinforcement Learning](https://openreview.net/forum?id=gSHyqBijPFO), ICLR 2023.
- [Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts](https://openaccess.thecvf.com/content/ICCV2023W/MMFM/html/Maniparambil_Enhancing_CLIP_with_GPT-4_Harnessing_Visual_Descriptions_as_Prompts_ICCVW_2023_paper.html), ICCV 2023.
- [PODA: Prompt-driven Zero-shot Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2023/html/Fahes_PODA_Prompt-driven_Zero-shot_Domain_Adaptation_ICCV_2023_paper.html), ICCV 2023.
- [ChatGPT-Powered Hierarchical Comparisons for Image Classification](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dc81297c791bb989deade65c6bd8c1d8-Abstract-Conference.html), NeurIPS 2023.
- [DomainVerse: A Benchmark Towards Real-World Distribution Shifts For Tuning-Free Adaptive Domain Generalization](https://arxiv.org/abs/2403.02714), arXiv 2024.
- [Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs](https://arxiv.org/abs/2403.11755), arXiv 2024.
- [Improving Text-to-Image Consistency via Automatic Prompt Optimization](https://arxiv.org/abs/2403.17804), arXiv 2024.

Embedding-prompting

- [Visual Prompt Tuning for Test-time Domain Adaptation](https://arxiv.org/abs/2210.04831), arXiv 2022.
- [Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models](https://openreview.net/forum?id=e8PVEkSa4Fq), NeurIPS 2022.
- [Decorate the Newcomers: Visual Domain Prompt for Continual Test Time Adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/25922), AAAI 2023.
- [Test-Time Personalization with Meta Prompt for Gaze Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/28151), AAAI 2023.
- [Cloud-Device Collaborative Adaptation to Continual Changing Environments in the Real-World](https://openaccess.thecvf.com/content/CVPR2023/html/Pan_Cloud-Device_Collaborative_Adaptation_to_Continual_Changing_Environments_in_the_Real-World_CVPR_2023_paper.html), CVPR 2023.
- [Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_Diverse_Data_Augmentation_with_Diffusions_for_Effective_Test-time_Prompt_Tuning_ICCV_2023_paper.html), ICCV 2023.
- [Convolutional Visual Prompt for Robust Visual Perception](https://proceedings.neurips.cc/paper_files/paper/2023/hash/58be158bf831a706b1a66cffbc401cac-Abstract-Conference.html), NeurIPS 2023.
- [SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html), NeurIPS 2023.
- [Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe8debfd5a36ada52e038c8b2078b2ce-Abstract-Conference.html), NeurIPS 2023.
- [Robust Test-Time Adaptation for Zero-Shot Prompt Tuning](https://ojs.aaai.org/index.php/AAAI/article/view/29611), AAAI 2024.
- [Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models](https://openreview.net/forum?id=kIP0duasBb), ICLR 2024.
- [C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion](https://openreview.net/forum?id=jzzEHTBFOT), ICLR 2024.
- [Any-Shift Prompting for Generalization over Distributions](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Any-Shift_Prompting_for_Generalization_over_Distributions_CVPR_2024_paper.html), CVPR 2024.
- [Test-Time Model Adaptation with Only Forward Passes](https://openreview.net/forum?id=qz1Vx1v9iK), ICML 2024.
