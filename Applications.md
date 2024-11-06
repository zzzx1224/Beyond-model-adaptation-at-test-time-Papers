
## Applications

Test-time adaptation has recently been applied to an expanding range of tasks. We review the literature on test-time adaptation across image-level, video-level, and 3D tasks, as well as applications beyond the field of vision.

### Image-level

#### Image classification

Image classification is the most investigated task in test-time adaptation due to its ease of implementing various distribution shifts, straightforward evaluation, and generalization to other tasks. Most works in [What to adapt.md](https://github.com/zzzx1224/Papers-of-test-time-adaptation/blob/main/What%20to%20adapt.md) are conducted on the image classification task.

#### Dense prediction

Segmentation
- [SITA: Single Image Test-time Adaptation](https://arxiv.org/abs/2112.02355), arXiv 2021.
- [On the Road to Online Adaptation for Semantic Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/html/Volpi_On_the_Road_to_Online_Adaptation_for_Semantic_Image_Segmentation_CVPR_2022_paper.html), CVPR 2022.
- [AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/html/Zhang_AuxAdapt_Stable_and_Efficient_Test-Time_Adaptation_for_Temporally_Consistent_Video_WACV_2022_paper.html), WACV 2022
- [Learning Instance-Specific Adaptation for Cross-Domain Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_27), ECCV 2022.
- [To Adapt or Not to Adapt? Real-Time Adaptation for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Colomer_To_Adapt_or_Not_to_Adapt_Real-Time_Adaptation_for_Semantic_ICCV_2023_paper.html), CVPR 2023.
- [Dynamically Instance-Guided Adaptation: A Backward-Free Approach for Test-Time Domain Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Dynamically_Instance-Guided_Adaptation_A_Backward-Free_Approach_for_Test-Time_Domain_Adaptive_CVPR_2023_paper.html), CVPR 2023.
- [Towards Open-Set Test-Time Adaptation Utilizing the Wisdom of Crowds in Entropy Minimization](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Towards_Open-Set_Test-Time_Adaptation_Utilizing_the_Wisdom_of_Crowds_in_ICCV_2023_paper.html), ICCV 2023.
- [Exploring Sparse Visual Prompt for Domain Adaptive Dense Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/29569), AAAI 2024.
- [Relax Image-Specific Prompt Requirement in SAM: A Single Generic Prompt for Segmenting Camouflaged Objects](https://ojs.aaai.org/index.php/AAAI/article/view/29144), AAAI 2024.
- [TALoS: Enhancing Semantic Scene Completion via Test-time Adaptation on the Line of Sight](https://arxiv.org/abs/2410.15674), NeurIPS 2024.

Depth
- [Deep Matching Prior: Test-Time Optimization for Dense Correspondence](https://openaccess.thecvf.com/content/ICCV2021/html/Hong_Deep_Matching_Prior_Test-Time_Optimization_for_Dense_Correspondence_ICCV_2021_paper.html), ICCV 2023.
- [SfM-TTR: Using Structure From Motion for Test-Time Refinement of Single-View Depth Networks](https://openaccess.thecvf.com/content/CVPR2023/html/Izquierdo_SfM-TTR_Using_Structure_From_Motion_for_Test-Time_Refinement_of_Single-View_CVPR_2023_paper.html), CVPR 2023.
- [Rapid Network Adaptation: Learning to Adapt Neural Networks Using Test-Time Feedback](https://openaccess.thecvf.com/content/ICCV2023/html/Yeo_Rapid_Network_Adaptation_Learning_to_Adapt_Neural_Networks_Using_Test-Time_ICCV_2023_paper.html), ICCV 2023.
- [Test-Time Adaptation for Depth Completion](https://openaccess.thecvf.com/content/CVPR2024/html/Park_Test-Time_Adaptation_for_Depth_Completion_CVPR_2024_paper.html), CVPR 2024.
- [Metric from Human: Zero-shot Monocular Metric Depth Estimation via Test-time Adaptation](https://nips.cc/virtual/2024/poster/95921), NeurIPS 2024.

Object detection
- [TeST: Test-Time Self-Training Under Distribution Shift](https://openaccess.thecvf.com/content/WACV2023/html/Sinha_TeST_Test-Time_Self-Training_Under_Distribution_Shift_WACV_2023_paper.html), WACV 2023.
- [Test Time Adaptation With Regularized Loss for Weakly Supervised Salient Object Detection](https://openaccess.thecvf.com/content/CVPR2023/html/Veksler_Test_Time_Adaptation_With_Regularized_Loss_for_Weakly_Supervised_Salient_CVPR_2023_paper.html), CVPR 2023.

#### Image enhancement

Super-resolution
- [Test-Time Adaptation for Super-Resolution: You Only Need to Overfit on a Few More Images](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Rad_Test-Time_Adaptation_for_Super-Resolution_You_Only_Need_to_Overfit_on_ICCVW_2021_paper.html), ICCV 2021.
- [Efficient Test-Time Adaptation for Super-Resolution with Second-Order Degradation and Reconstruction](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ec3d49763c653ad7c8d587f52220c129-Abstract-Conference.html), NeurIPS 2023.

Low-light image enhancement
- [Generalized Lightness Adaptation with Channel Selective Normalization](https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Generalized_Lightness_Adaptation_with_Channel_Selective_Normalization_ICCV_2023_paper.html), ICCV 2023

Image restoration
- [Test-Time Degradation Adaptation for Open-Set Image Restoration](https://openreview.net/forum?id=XLlQb24X2o), ICML 2024.

Image denoising
- [TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01921.pdf), ECCV 2024.

Image dehazing
- [Towards Multi-Domain Single Image Dehazing via Test-Time Training](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Towards_Multi-Domain_Single_Image_Dehazing_via_Test-Time_Training_CVPR_2022_paper.html), CVPR 2022.
- [Prompt-based Test-time Real Image Dehazing: A Novel Pipeline](https://arxiv.org/abs/2309.17389), ECCV 2024.

Image deblurring
- [Test-Time Fast Adaptation for Dynamic Scene Deblurring via Meta-Auxiliary Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Chi_Test-Time_Fast_Adaptation_for_Dynamic_Scene_Deblurring_via_Meta-Auxiliary_Learning_CVPR_2021_paper.html), CVPR 2021.

#### Medical imaging
Classification
- [Test-time Adaptation with Calibration of Medical Image Classification Nets for Label Distribution Shift](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_30), MICCAI 2022.
- [Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement](https://openreview.net/forum?id=ZvJ2lQQKjz), ICML 2024.

Segmentation
- [Test-time Unsupervised Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_42), MICCAI 2020.
- [Fully Test-Time Adaptation for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24), MICCAI 2021.
- [Test-Time Adaptation with Shape Moments for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_70), MICCAI 2022.
- [Autoencoder based self-supervised test-time adaptation for medical image analysis](https://www.sciencedirect.com/science/article/pii/S1361841521001821), Medical Image Analysis, 2022.
- [Single-domain Generalization in Medical Image Segmentation via Test-time Adaptation from Shape Dictionary](https://www.aaai.org/AAAI22Papers/AAAI-852.LiuQ.pdf), AAAI 2022.
- [Feature Alignment and Uniformity for Test Time Adaptation](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Feature_Alignment_and_Uniformity_for_Test_Time_Adaptation_CVPR_2023_paper.html), CVPR 2023.
- [On-the-Fly Test-time Adaptation for Medical Image Segmentation](https://proceedings.mlr.press/v227/valanarasu24a.html), Medical Imaging with Deep Learning, 2024.

Reconstruction
- [Test-Time Model Adaptation for Image Reconstruction Using Self-supervised Adaptive Layers](https://bpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/8/10877/files/2024/07/ECCV_2024_adaption.pdf), ECCV 2024.

MRI acceleration
- [Test-Time Training Can Close the Natural Distribution Shift Performance Gap in Deep Learning Based Compressed Sensing](https://proceedings.mlr.press/v162/darestani22a.html), ICML 2022.
- [MotionTTT: 2D Test-Time-Training Motion Estimation for 3D Motion Corrected MRI](https://arxiv.org/abs/2409.09370), NeurIPS 2024.

#### Others

Pose estimation
- [Test-Time Personalization with a Transformer for Human Pose Estimation](https://proceedings.neurips.cc/paper/2021/hash/1517c8664be296f0d87d9e5fc54fdd60-Abstract.html), NeurIPS 2021.
- [Self-Constrained Inference Optimization on Structural Groups for Human Pose Estimation](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_42), ECCV 2022.
- [Meta-Auxiliary Learning for Adaptive Human Pose Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/25760), AAAI 2023.
- [Self-Correctable and Adaptable Inference for Generalizable Human Pose Estimation](https://openaccess.thecvf.com/content/CVPR2023/html/Kan_Self-Correctable_and_Adaptable_Inference_for_Generalizable_Human_Pose_Estimation_CVPR_2023_paper.html), CVPR 2023.
- [TTA-COPE: Test-Time Adaptation for Category-Level Object Pose Estimation](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_TTA-COPE_Test-Time_Adaptation_for_Category-Level_Object_Pose_Estimation_CVPR_2023_paper.html), CVPR 2023.

Person re-identification 
- [Generalizable Person Re-identification via Self-Supervised Batch Norm Test-Time Adaption](https://ojs.aaai.org/index.php/AAAI/article/view/19963), AAAI 2022.
- [Heterogeneous Test-Time Training for Multi-Modal Person Re-identification](https://ojs.aaai.org/index.php/AAAI/article/view/28398), AAAI 2024.

DeepFake detection
- [OST: Improving Generalization of DeepFake Detection via One-Shot Test-Time Training](https://openreview.net/forum?id=YPoRoad6gzY), NeurIPS 2022.
- [Test-Time Domain Generalization for Face Anti-Spoofing](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Test-Time_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2024_paper.html), CVPR 2024.

Out-of-distribution detection
- [ATTA: Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8dcc306a2522c60a78f047ab8739e631-Abstract-Conference.html), NeurIPS 2023.
- [When Model Meets New Normals: Test-Time Adaptation for Unsupervised Time-Series Anomaly Detection](https://ojs.aaai.org/index.php/AAAI/article/view/29210), AAAI 2024.
- [Test-Time Linear Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Test-Time_Linear_Out-of-Distribution_Detection_CVPR_2024_paper.html), CVPR 2024.

Style transfer
- [Deep Translation Prior: Test-Time Training for Photorealistic Style Transfer](https://ojs.aaai.org/index.php/AAAI/article/view/20004), AAAI 2022.

Federated learning
- [Towards Instance-adaptive Inference for Federated Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_Towards_Instance-adaptive_Inference_for_Federated_Learning_ICCV_2023_paper.html), ICCV 2023.
- [Adaptive Test-Time Personalization for Federated Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f555b62384279b98732204cb1a670a23-Abstract-Conference.html), NeurIPS 2023.
- [Is Heterogeneity Notorious? Taming Heterogeneity to Handle Test-Time Shift in Federated Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/565f995643da6329cec701f26f8579f5-Abstract-Conference.html), NeurIPS 2023.



### Video-level

#### Action and behavior classification

Action recognition
- [Video Test-Time Adaptation for Action Recognition](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Video_Test-Time_Adaptation_for_Action_Recognition_CVPR_2023_paper.html), CVPR 2023.
- [Temporal Coherent Test Time Optimization for Robust Video Classification](https://openreview.net/forum?id=-t4D61w4zvQ), ICLR 2023.
- [Modality-Collaborative Test-Time Adaptation for Action Recognition](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Modality-Collaborative_Test-Time_Adaptation_for_Action_Recognition_CVPR_2024_paper.html), CVPR 2024.

Expression recognition
- [TempT: Temporal Consistency for Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2023W/ABAW/html/Mutlu_TempT_Temporal_Consistency_for_Test-Time_Adaptation_CVPRW_2023_paper.html), CVPR 2023.

Action localization
- [Test-Time Zero-Shot Temporal Action Localization](https://openaccess.thecvf.com/content/CVPR2024/html/Liberatori_Test-Time_Zero-Shot_Temporal_Action_Localization_CVPR_2024_paper.html), CVPR 2024.


#### Dense prediction

Segmentation
- [Self-Supervised Test-Time Adaptation on Video Data](https://openaccess.thecvf.com/content/WACV2022/html/Azimi_Self-Supervised_Test-Time_Adaptation_on_Video_Data_WACV_2022_paper.html), WACV 2022.
- [AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/html/Zhang_AuxAdapt_Stable_and_Efficient_Test-Time_Adaptation_for_Temporally_Consistent_Video_WACV_2022_paper.html), WACV 2022.
- [Test-time Training for Matching-based Video Object Segmentation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4267d84ca2f6fbb4aa5172b76b433aca-Abstract-Conference.html), NeurIPS 2023.
- [Depth-aware Test-Time Training for Zero-shot Video Object Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Depth-aware_Test-Time_Training_for_Zero-shot_Video_Object_Segmentation_CVPR_2024_paper.html), CVPR 2024.

Detection
- [Context Enhanced Transformer for Single Image Object Detection in Video Data](https://ojs.aaai.org/index.php/AAAI/article/view/27825), AAAI 2024.

Depth
- [Meta-Auxiliary Learning for Future Depth Prediction in Videos](https://openaccess.thecvf.com/content/WACV2023/html/Liu_Meta-Auxiliary_Learning_for_Future_Depth_Prediction_in_Videos_WACV_2023_paper.html), WACV 2023.

Tracking
- [DARTH: Holistic Test-time Adaptation for Multiple Object Tracking](https://openaccess.thecvf.com/content/ICCV2023/html/Segu_DARTH_Holistic_Test-time_Adaptation_for_Multiple_Object_Tracking_ICCV_2023_paper.html), ICCV 2023.

#### Video enhancement
- [Genuine Knowledge from Practice: Diffusion Test-Time Adaptation for Video Adverse Weather Removal](https://arxiv.org/abs/2403.07684), CVPR 2024.
- [TTA-EVF: Test-Time Adaptation for Event-based Video Frame Interpolation via Reliable Pixel and Sample Estimation](https://openaccess.thecvf.com/content/CVPR2024/html/Cho_TTA-EVF_Test-Time_Adaptation_for_Event-based_Video_Frame_Interpolation_via_Reliable_CVPR_2024_paper.html), CVPR 2024
- [Domain-Adaptive Video Deblurring via Test-Time Blurring](https://link.springer.com/chapter/10.1007/978-3-031-73404-5_8), ECCV 2024.

### 3D-level

#### Classification
- [MATE: Masked Autoencoders are Online 3D Test-Time Learners](https://openaccess.thecvf.com/content/ICCV2023/html/Mirza_MATE_Masked_Autoencoders_are_Online_3D_Test-Time_Learners_ICCV_2023_paper.html), ICCV 2023.
- [Backpropagation-free Network for 3D Test-time Adaptation](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Backpropagation-free_Network_for_3D_Test-time_Adaptation_CVPR_2024_paper.html), CVPR 2024.
- [CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation](https://link.springer.com/content/pdf/10.1007/978-3-031-72751-1_26.pdf), ECCV 2024.

#### Dense prediction

Segmentation
- [MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/html/Shin_MM-TTA_Multi-Modal_Test-Time_Adaptation_for_3D_Semantic_Segmentation_CVPR_2022_paper.html), CVPR 2022.
- [GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_33), ECCV 2022.
- [Test-time Adaptation with Slot-Centric Models](https://proceedings.mlr.press/v202/prabhudesai23a.html), ICML 2023.
- [Multi-Modal Continual Test-Time Adaptation for 3D Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Cao_Multi-Modal_Continual_Test-Time_Adaptation_for_3D_Semantic_Segmentation_ICCV_2023_paper.html), ICCV 2023.
- [Reliable Spatial-Temporal Voxels For Multi-modal Test-Time Adaptation](https://link.springer.com/chapter/10.1007/978-3-031-73390-1_14), ECCV 2024.
- [HGL: Hierarchical Geometry Learning for Test-time Adaptation in 3D Point Cloud Segmentation](https://arxiv.org/abs/2407.12387), ECCV 2024.
- [TTT-KD: Test-Time Training for 3D Semantic Segmentation through Knowledge Distillation from Foundation Models](https://arxiv.org/abs/2403.11691), arXiv 2024.

Detection
- [MonoTTA: Fully Test-Time Adaptation for Monocular 3D Object Detection](https://link.springer.com/chapter/10.1007/978-3-031-72784-9_6), ECCV 2024.
- [Reg-TTA3D: Better Regression Makes Better Test-Time Adaptive 3D Object Detection](https://link.springer.com/chapter/10.1007/978-3-031-72775-7_12), ECCV 2024.

#### Others

Pose estimation
- [Inference Stage Optimization for Cross-scenario 3D Human Pose Estimation](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1943102704f8f8f3302c2b730728e023-Abstract.html), NeurIPS 2020.

Pose forecasting
- [Test-time Personalizable Forecasting of 3D Human Poses](https://openaccess.thecvf.com/content/ICCV2023/html/Cui_Test-time_Personalizable_Forecasting_of_3D_Human_Poses_ICCV_2023_paper.html), ICCV 2023.
- [Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-Time Adaptation Framework](https://link.springer.com/content/pdf/10.1007/978-3-031-72751-1_25.pdf), ECCV 2024.

Point-cloud registration
- [Point-TTA: Test-Time Adaptation for Point Cloud Registration Using Multitask Meta-Auxiliary Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Hatem_Point-TTA_Test-Time_Adaptation_for_Point_Cloud_Registration_Using_Multitask_Meta-Auxiliary_ICCV_2023_paper.html), ICCV 2023.

Flow estimation 
- [Dual-frame Fluid Motion Estimation with Test-time Optimization and Zero-divergence Loss](https://arxiv.org/abs/2410.11934), NeurIPS 2024. 

Human mesh reconstruction 
- [Cyclic Test-Time Adaptation on Monocular Video for 3D Human Mesh Reconstruction](https://openaccess.thecvf.com/content/ICCV2023/html/Nam_Cyclic_Test-Time_Adaptation_on_Monocular_Video_for_3D_Human_Mesh_ICCV_2023_paper.html), ICCV 2023.
- [Incorporating Test-Time Optimization into Training with Dual Networks for Human Mesh Recovery](https://arxiv.org/abs/2401.14121), NeurIPS 2024.

Multi-task point cloud understanding
- [PCoTTA: Continual Test-Time Adaptation for Multi-Task Point Cloud Understanding](https://nips.cc/virtual/2024/poster/96487), NeurIPS 2024.

### Beyond vision

#### Reinforcement learning

Policy adaptation
- [Self-Supervised Policy Adaptation during Deployment](https://openreview.net/forum?id=o_V-MjyyGV), ICLR 2021.
- [Diagnosis, Feedback, Adaptation: A Human-in-the-Loop Framework for Test-Time Policy Adaptation](https://proceedings.mlr.press/v202/peng23c.html), ICML 2023.
- [Design from Policies: Conservative Test-Time Adaptation for Offline Policy Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/31610e68fe41a62e460e044216a10766-Abstract-Conference.html), NeurIPS 2023.
- [MoVie: Visual Model-Based Policy Adaptation for View Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/43b77cef2a83a25aa27d3271d209e4fd-Abstract-Conference.html), NeurIPS 2023.

Combinatorial optimization
- [Meta-SAGE: Scale Meta-Learning Scheduled Adaptation with Guided Exploration for Mitigating Scale Shift on Combinatorial Optimization](https://proceedings.mlr.press/v202/son23a.html), ICML 2023.

#### Natural language processing

Question answering
- [Self-Supervised Test-Time Learning for Reading Comprehension](https://aclanthology.org/2021.naacl-main.95/), ACL 2021.
- [Robust Question Answering against Distribution Shifts with Test-Time Adaptation: An Empirical Study], EMNLP 2023.

Text-to-SQL
- [Conditional Tree Matching for Inference-Time Adaptation of Tree Prediction Models](https://proceedings.mlr.press/v202/varma23a.html), ICML 2023.

LLM adaptation
- [Test-Time Training on Nearest Neighbors for Large Language Models](https://arxiv.org/abs/2305.18466), ICLR 2024.


#### Multimodal learning

Vision-language classification
- [TEMPERA: Test-Time Prompt Editing via Reinforcement Learning](https://openreview.net/forum?id=gSHyqBijPFO), ICLR 2023.
- [Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts](https://openaccess.thecvf.com/content/ICCV2023W/MMFM/html/Maniparambil_Enhancing_CLIP_with_GPT-4_Harnessing_Visual_Descriptions_as_Prompts_ICCVW_2023_paper.html), ICCV 2023.
- [PODA: Prompt-driven Zero-shot Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2023/html/Fahes_PODA_Prompt-driven_Zero-shot_Domain_Adaptation_ICCV_2023_paper.html), ICCV 2023.
- [ChatGPT-Powered Hierarchical Comparisons for Image Classification](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dc81297c791bb989deade65c6bd8c1d8-Abstract-Conference.html), NeurIPS 2023.
- [Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs](https://arxiv.org/abs/2403.11755), arXiv 2024.
- [Improving Text-to-Image Consistency via Automatic Prompt Optimization](https://arxiv.org/abs/2403.17804), arXiv 2024.
- [Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models](https://openreview.net/forum?id=e8PVEkSa4Fq), NeurIPS 2022.
- [Cloud-Device Collaborative Adaptation to Continual Changing Environments in the Real-World](https://openaccess.thecvf.com/content/CVPR2023/html/Pan_Cloud-Device_Collaborative_Adaptation_to_Continual_Changing_Environments_in_the_Real-World_CVPR_2023_paper.html), CVPR 2023.
- [Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_Diverse_Data_Augmentation_with_Diffusions_for_Effective_Test-time_Prompt_Tuning_ICCV_2023_paper.html), ICCV 2023.
- [SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html), NeurIPS 2023.
- [Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe8debfd5a36ada52e038c8b2078b2ce-Abstract-Conference.html), NeurIPS 2023.
- [Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models](https://openreview.net/forum?id=kIP0duasBb), ICLR 2024.
- [C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion](https://openreview.net/forum?id=jzzEHTBFOT), ICLR 2024.
- [Any-Shift Prompting for Generalization over Distributions](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Any-Shift_Prompting_for_Generalization_over_Distributions_CVPR_2024_paper.html), CVPR 2024.
- [Frustratingly Easy Test-Time Adaptation of Vision-Language Models](https://arxiv.org/abs/2405.18330), NeurIPS 2024.
- [Historical Test-time Prompt Tuning for Vision Foundation Models](https://arxiv.org/abs/2410.20346), NeurIPS 2024.

Visual document understanding
- [Is it an i or an l: Test-time Adaptation of Text Line Recognition Models](https://arxiv.org/abs/2308.15037), arXiv 2022.
- [Test-Time Adaptation for Visual Document Understanding](https://arxiv.org/abs/2206.07240), arXiv 2023.

Vision question answering
- [Test-Time Model Adaptation for Visual Question Answering With Debiased Self-Supervisions](https://ieeexplore.ieee.org/abstract/document/10173554?casa_token=Mt5tk_Tc5GsAAAAA:cYy3YfNKZeNLxcIH-30H6tcKL5o5f17LW1t1dQf9uneU2aWoyAtU1mZ-Av-5_zX2EYDJtu6QrdPa), T-Multimedia 2023.
- [Question Type-Aware Debiasing for Test-time Visual Question Answering Model Adaptation](https://ieeexplore.ieee.org/abstract/document/10550013?casa_token=8akukywOnrsAAAAA:Xkn1W1ngJqLLKNv5WOrgAKL2N-oTWwP0yhWXt5VwQESqw6T9PjOTCm5QB0m0S302kkQADtajBLWI), T-CSVT 2024.

Vision-and-language navigation
- [Fast-Slow Test-Time Adaptation for Online Vision-and-Language Navigation](https://openreview.net/forum?id=Zos5wsaB5r), ICML 2024.

#### Others

Speech
- [Variational On-the-Fly Personalization](https://proceedings.mlr.press/v162/kim22e.html), ICML 2022.
- [Test-Time Training for Speech](https://arxiv.org/abs/2309.10930), arXiv 2023.
- [SGEM: Test-Time Adaptation for Automatic Speech Recognition via Sequential-Level Generalized Entropy Minimization(https://arxiv.org/abs/2306.01981), arXiv 2023.]

Forecasting
- [Self-Adaptive Forecasting for Improved Deep Learning on Non-Stationary Time-Series](https://arxiv.org/abs/2202.02403), arXiv 2022.
- [Test-Time Training for Spatial-Temporal Forecasting](https://epubs.siam.org/doi/abs/10.1137/1.9781611978032.54), SDM 2024
- [T4P: Test-Time Training of Trajectory Prediction via Masked Autoencoder and Actor-specific Token Memory](https://openaccess.thecvf.com/content/CVPR2024/html/Park_T4P_Test-Time_Training_of_Trajectory_Prediction_via_Masked_Autoencoder_and_CVPR_2024_paper.html), CVPR 2024.

Tabular data
- [TabLog: Test-Time Adaptation for Tabular Data Using Logic Rules](https://openreview.net/forum?id=LZeixIvQcB), ICML 2024.
