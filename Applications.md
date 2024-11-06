
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
- [SfM-TTR: Using Structure From Motion for Test-Time Refinement of Single-View Depth Networks](https://openaccess.thecvf.com/content/CVPR2023/html/Izquierdo_SfM-TTR_Using_Structure_From_Motion_for_Test-Time_Refinement_of_Single-View_CVPR_2023_paper.html), CVPR 2023.
- [Rapid Network Adaptation: Learning to Adapt Neural Networks Using Test-Time Feedback](https://openaccess.thecvf.com/content/ICCV2023/html/Yeo_Rapid_Network_Adaptation_Learning_to_Adapt_Neural_Networks_Using_Test-Time_ICCV_2023_paper.html), ICCV 2023.
- [Test-Time Adaptation for Depth Completion](https://openaccess.thecvf.com/content/CVPR2024/html/Park_Test-Time_Adaptation_for_Depth_Completion_CVPR_2024_paper.html), CVPR 2024.
- [Metric from Human: Zero-shot Monocular Metric Depth Estimation via Test-time Adaptation](https://nips.cc/virtual/2024/poster/95921), NeurIPS 2024.

Object detection

Dense correspondence

#### Image enhancement

#### Medical imaging
- [Test-time Unsupervised Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_42), MICCAI 2020.
- [Fully Test-Time Adaptation for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24), MICCAI 2021.
- [Test-Time Adaptation with Shape Moments for Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_70), MICCAI 2022.
- [Autoencoder based self-supervised test-time adaptation for medical image analysis](https://www.sciencedirect.com/science/article/pii/S1361841521001821), Medical Image Analysis, 2022.
- [On-the-Fly Test-time Adaptation for Medical Image Segmentation](https://arxiv.org/abs/2203.05574), Arxiv.
- [Single-domain Generalization in Medical Image Segmentation via Test-time Adaptation from Shape Dictionary](https://www.aaai.org/AAAI22Papers/AAAI-852.LiuQ.pdf), AAAI 2022.

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
