
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
- [Test-Time Degradation Adaptation for Open-Set Image Restoration], ICML 2024.

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

#### Other image-level applications

DeepFake detection
- [OST: Improving Generalization of DeepFake Detection via One-Shot Test-Time Training](https://openreview.net/forum?id=YPoRoad6gzY), NeurIPS 2022.

Pose estimation
- [Test-Time Personalization with a Transformer for Human Pose Estimation](https://proceedings.neurips.cc/paper/2021/hash/1517c8664be296f0d87d9e5fc54fdd60-Abstract.html), NeurIPS 2021. 
### Image enhancement & restoration


### Video-level
- [Video Test-Time Adaptation for Action Recognition](https://arxiv.org/abs/2211.15393), Arxiv.
- [Self-Supervised Test-Time Adaptation on Video Data](https://openaccess.thecvf.com/content/WACV2022/html/Azimi_Self-Supervised_Test-Time_Adaptation_on_Video_Data_WACV_2022_paper.html), WACV 2022
- [AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/html/Zhang_AuxAdapt_Stable_and_Efficient_Test-Time_Adaptation_for_Temporally_Consistent_Video_WACV_2022_paper.html), WACV 2022



### Other applications
