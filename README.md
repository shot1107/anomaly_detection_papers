# anomaly_detection_papers
summary of anomaly detection papers accepted at top conferences in CV, ML, and AI(CVPR, NeurIPS, ICCV, ECCV, AAAI, ICML, IJCAI, ICLR, ICIP, BMVC, ACCV)   
(My main interest is in the two-class classification of abnormal/normal and segmentation of abnormal areas)

CV・ML・AI 分野の Top Conferences(CVPR, NeurIPS, ICCV, ECCV, AAAI, ICML, IJCAI, ICLR, ICIP, BMVC, ACCV) で Accept された Anomaly Detection 関連の論文をまとめる  
(主な自分の興味は、画像を対象とした異常・正常の２クラス分類や異常箇所のセグメンテーション)  
検索ワード : anom*, abnom*, defect, outlier 

# 2022
## CVPR2022 Jun 24, 2022
## ICLR2022 Apr 29, 2022
## AAAI2022  Mar 1, 2022

# 2021
## NeurIPS2021 Dec 6, 2021
- #### Online false discovery rate control for anomaly detection in time series [pdf](https://openreview.net/forum?id=NvN_B_ZEY5c)
- #### Detecting Anomalous Event Sequences with Temporal Point Processes [pdf](https://openreview.net/forum?id=MTMyxzrIKsM)
- #### Learned Robust PCA: A Scalable Deep Unfolding Approach for High-Dimensional Outlier Detection [pdf](https://openreview.net/forum?id=G7W2mriQLxf)

## BMVC2021 Nov 25, 2021
- #### Student-Teacher Feature Pyramid Matching for Anomaly Detection [arXiv](https://arxiv.org/abs/2103.04257)
- #### Learning Not to Reconstruct Anomalies [arXiv](https://arxiv.org/abs/2110.09742)
- #### ESAD: End-to-end Deep Semi-supervised Anomaly Detection [arXiv](https://arxiv.org/abs/2012.04905)
- #### Elsa: Energy-based learning for semi-supervised anomaly detection [arXiv](https://arxiv.org/abs/2103.15296)

## ICIP2021 Sep 22, 2021
- #### A Two-Stage Autoencoder For Visual Anomaly Detection [pdf](https://ieeexplore.ieee.org/document/9506538)
   - **Domain:** Image / **Dataset:** MNIST, FMNIST, CIFAR10, Fastener / **Index Terms:** Autoencoder, RotNet  
   - Deep convolutional autoencoder（DCAE)が異常箇所までも再構成してしまう問題を解決するため、2ステージで学習を行う非対称DCAEを提案。まず、RotNet をEncoderとして学習し、学習済みRotNetを凍結したまま2種類のDecoderを学習(1つはloss関数にMSEを、もう1つはSSIMを使用)。2つのDecoderの結果を組み合わせるて最終的な異常スコアを得る。FMNIST, Fastenerにおいてベースライン手法より優れた性能。
- #### Anomaly Detection via Self-organizing Map [arXiv](https://arxiv.org/abs/2107.09903)
- #### Deep Unsupervised Image Anomaly Detection: An Information Theoretic Framework [arXiv](https://arxiv.org/abs/2012.04837)
- #### Joint Anomaly Detection and Inpainting for Microscopy Images Via Deep Self-Supervised Learning [pdf](https://ieeexplore.ieee.org/abstract/document/9506454)
- #### Multi-Scale Background Suppression Anomaly Detection In Surveillance Videos [pdf](https://ieeexplore.ieee.org/document/9506580)
- #### Particle Swarm And Pattern Search Optimisation Of An Ensemble Of Face Anomaly Detectors [pdf](https://ieeexplore.ieee.org/document/9506251)
- #### SAGAN: Skip-Attention GAN For Anomaly Detection [pdf](https://ieeexplore.ieee.org/abstract/document/9506332)
- #### Toward Unsupervised 3d Point Cloud Anomaly Detection Using Variational Autoencoder [pdf](https://ieeexplore.ieee.org/document/9506795)
- #### Unsupervised Variability Normalization For Anomaly Detection [pdf](https://ieeexplore.ieee.org/document/9506742)
- #### Effort-free Automated Skeletal Abnormality Detection of Rat Fetuses on Whole-body Micro-CT Scans [arXiv](https://arxiv.org/abs/2106.01830)
- #### Cam-Guided U-Net With Adversarial Regularization For Defect Segmentation [pdf](https://ieeexplore.ieee.org/document/9506582)
- #### Defect Inspection using Gravitation Loss and Soft Labels [pdf](https://ieeexplore.ieee.org/document/9506327)
- #### Identification Of In-Field Sensor Defects In The Context Of Image Age Approximation [pdf](https://ieeexplore.ieee.org/document/9506023)
- #### S2D2Net: An Improved Approach For Robust Steel Surface Defects Diagnosis With Small Sample Learning [pdf](https://ieeexplore.ieee.org/document/9506405)

## ICCV2021 Oct 17, 2021
- #### Weakly Supervised Temporal Anomaly Segmentation with Dynamic Time Warping [arXiv](https://arxiv.org/abs/2108.06816)
- #### A Hierarchical Transformation-Discriminating Generative Model for Few Shot Anomaly Detection [arXiv](https://arxiv.org/abs/2104.14535)
- #### Divide-and-Assemble: Learning Block-wise Memory for Unsupervised Anomaly Detection [arXiv](https://arxiv.org/abs/2107.13118)
- #### DRAEM -- A discriminatively trained reconstruction embedding for surface anomaly detection [arXiv](https://arxiv.org/abs/2108.07610)
- #### Dance With Self-Attention: A New Look of Conditional Random Fields on Anomaly Detection in Videos [pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Purwanto_Dance_With_Self-Attention_A_New_Look_of_Conditional_Random_Fields_ICCV_2021_paper.pdf)
- #### Learning Unsupervised Metaformer for Anomaly Detection [pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Learning_Unsupervised_Metaformer_for_Anomaly_Detection_ICCV_2021_paper.pdf)
- #### A Hybrid Video Anomaly Detection Framework via Memory-Augmented Flow Reconstruction and Flow-Guided Frame Prediction [arXiv](https://arxiv.org/abs/2108.06852)
- #### Road Anomaly Detection by Partial Image Reconstruction With Segmentation Coupling [pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Vojir_Road_Anomaly_Detection_by_Partial_Image_Reconstruction_With_Segmentation_Coupling_ICCV_2021_paper.pdf)
- #### Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning [arXiv](https://arxiv.org/abs/2101.10030)

## IJCAI2021 Aug 26, 2021
- #### Masked Contrastive Learning for Anomaly Detection [arXiv](https://arxiv.org/abs/2105.08793)
- #### Weakly-Supervised Spatio-Temporal Anomaly Detection in Surveillance Video [arXiv](https://arxiv.org/abs/2108.03825)
- #### Understanding the Effect of Bias in Deep Anomaly Detection [arXiv](https://arxiv.org/abs/2105.07346)

## ICML2021 Jul 24, 2021
- #### Quantifying and Reducing Bias in Maximum Likelihood Estimation of Structured Anomalies [arXiv](https://arxiv.org/abs/2007.07878)
- #### Neural Transformation Learning for Deep Anomaly Detection Beyond Images [arXiv](https://arxiv.org/abs/2103.16440)
- #### Transfer-Based Semantic Anomaly Detection [pdf](http://proceedings.mlr.press/v139/deecke21a.html)
- #### A General Framework For Detecting Anomalous Inputs to DNN Classifiers [arXiv](https://arxiv.org/abs/2007.15147)
- #### Near-Optimal Entrywise Anomaly Detection for Low-Rank Matrices with Sub-Exponential Noise [arXiv](https://arxiv.org/abs/2006.13126)
- #### Event Outlier Detection in Continuous Time [arXiv](https://arxiv.org/abs/1912.09522)

## CVPR2021 Jun 19, 2021
- #### MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection [arXiv](http://arxiv.org/abs/2101.00529)
- #### CutPaste: Self-Supervised Learning for Anomaly Detection and Localization [arXiv](http://arxiv.org/abs/2104.04015)
- #### Pixel-Wise Anomaly Detection in Complex Driving Scenes [arXiv](http://arxiv.org/abs/2103.05445)
- #### PANDA: Adapting Pretrained Features for Anomaly Detection and Segmentation [arXiv](http://arxiv.org/abs/2010.05903)
- #### Glancing at the Patch: Anomaly Localization With Global and Local Feature Comparison [pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Glancing_at_the_Patch_Anomaly_Localization_With_Global_and_Local_CVPR_2021_paper.pdf)
- #### Anomaly Detection in Video via Self-Supervised and Multi-Task Learning [arXiv](http://arxiv.org/abs/2011.07491)
- #### Multiresolution Knowledge Distillation for Anomaly Detection [arXiv](http://arxiv.org/abs/2011.11108)
- #### Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark [pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Yao_Joint-DetNAS_Upgrade_Your_Detector_With_NAS_Pruning_and_Dynamic_Distillation_CVPR_2021_paper.pdf)

## ICLR 2021
関連論文なし

## AAAI2021 Feb 2, 2021
- #### LREN: Low-Rank Embedded Network for Sample-Free Hyperspectral Anomaly Detection [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16536)
- #### GAN Ensemble for Anomaly Detection [arXiv](https://arxiv.org/abs/2012.07988)
  - **Domain:** Image / **Dataset:** MNIST, CIFAR-10, OCT, KDD99 / **Index Terms:** GAN, Ensemble  
  - GANの学習には不安定性やモード崩壊などいくつかの欠点があるが、近年の研究でGeneratorやDiscriminatorを複数用意することでそれらの問題を解決できることが示されている。この論文では、複数のGenerator(実際にはEncoder-Decoder)とDiscriminatorを用意し、それらをアンサンブルすることで画像の異常検出問題を解決している。訓練時にはGeneratorは複数のDiscriminatorからフィードバックを受け、Discriminatorは複数のGeneratorの出力を識別する。推論時はすべてネットワーク(Encoder-Decoder⇨Discriminator)の出力の平均をとる。ベースモデルとして、f-AnoGAN・EGBAD・GANomaly・Skip-GANomalyを使用し、すべてのデータセットで単体モデルよりアンサンブルの方が優れた性能。
- #### Anomaly Attribution with Likelihood Compensation [pdf](https://ide-research.net/papers/2021_AAAI_Ide.pdf)
- #### Regularizing Attention Networks for Anomaly Detection in Visual Question Answering [arXiv](https://arxiv.org/abs/2009.10054)
- #### Appearance-Motion Memory Consistency Network for Video Anomaly Detection [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16177)
- #### Learning Semantic Context from Normal Samples for Unsupervised Anomaly Detection [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16420)
- #### Graph Neural Network-Based Anomaly Detection in Multivariate Time Series [arXiv](https://arxiv.org/abs/2106.06947)
- #### Time Series Anomaly Detection with Multiresolution Ensemble Decoding [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/17152)
- #### A New Window Loss Function for Bone Fracture Detection and Localization in X-ray Images with Point-based Annotation [arXiv](https://arxiv.org/abs/2012.04066)
- #### Towards Balanced Defect Prediction with Better information Propagation [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16157)
- #### Accelerated Combinatorial Search for Outlier Detection with Provable Bound on Sub-Optimality [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/17475)
- #### Neighborhood Consensus Networks for Unsupervised Multi-view Outlier Detection [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16873)
- #### Outlier Impact Characterization for Time Series Data [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/17379)

## IJCAI2020 Jan 7, 2021
- #### Inductive Anomaly Detection on Attributed Networks [pdf](https://www.ijcai.org/proceedings/2020/179)
- #### Robustness of Autoencoders for Anomaly Detection Under Adversarial Impact [pdf](https://www.ijcai.org/proceedings/2020/173)
- #### Towards a Hierarchical Bayesian Model of Multi-View Anomaly Detection [pdf](https://www.ijcai.org/proceedings/2020/335)
- #### Cross-Interaction Hierarchical Attention Networks for Urban Anomaly Prediction [pdf](https://www.ijcai.org/proceedings/2020/601)
- #### Latent Regularized Generative Dual Adversarial Network For Abnormal Detection [pdf](https://www.ijcai.org/proceedings/2020/106)

## Other important papers published in 2021 / その他2021年の重要論文
- #### Image-based Plant Disease Diagnosis with Unsupervised Anomaly Detection Based on Reconstructability of Colors (accepted by IMPROVE 2021) [arXiv](https://arxiv.org/abs/2011.14306) 
  - **Domain:** Image / **Dataset:**  PlantVillage / **Index Terms:** pix2pix, CIEDE2000  
  - 画像間スタイル変換手法である pix2pix を用いた異常検出。訓練画像(RGB)とそれをグレースケールに変換した画像をペアにして pix2pix を学習。そのように学習されたモデルは、テスト画像(RGB)->グレースケール->再構成画像(RGB) と変換する際に、欠陥箇所の色を再構成できない。テスト画像(RGB)と再構成画像(RGB)の差分を CIEDE2000 に基づいてとり、異常度を計算。AnoGAN, AE等と比較し優れた性能。

# 2020
## NeurIPS2020 Dec 6, 2020
- #### Timeseries Anomaly Detection using Temporal Hierarchical One-Class Network [pdf](https://papers.nips.cc/paper/2020/hash/97e401a02082021fd24957f852e0e475-Abstract.html)
- #### Understanding Anomaly Detection with Deep Invertible Networks through Hierarchies of Distributions and Features [arXiv](https://arxiv.org/abs/2006.10848)
- #### Further Analysis of Outlier Detection with Deep Generative Models [arXiv](https://arxiv.org/abs/2010.13064)

## ACCV2020 Dec 4, 2020
- #### A Day on Campus - An Anomaly Detection Dataset for Events in a Single Camera [pdf](https://openaccess.thecvf.com/content/ACCV2020/html/Pranav_A_Day_on_Campus_-_An_Anomaly_Detection_Dataset_for_ACCV_2020_paper.html)
- #### Patch SVDD: Patch-level SVDD for Anomaly Detection and Segmentation [arXiv](http://arxiv.org/abs/2006.16067)
- #### Learning to Adapt to Unseen Abnormal Activities under Weak Supervision [pdf](https://openaccess.thecvf.com/content/ACCV2020/html/Park_Learning_to_Adapt_to_Unseen_Abnormal_Activities_under_Weak_Supervision_ACCV_2020_paper.html)

## ICIP2020 Oct 25, 2020
- #### A Stacking Ensemble for Anomaly Based Client-Specific Face Spoofing Detection [pdf](https://ieeexplore.ieee.org/document/9190814)
- #### Anomalous Motion Detection on Highway Using Deep Learning [arXiv](https://arxiv.org/abs/2006.08143)
- #### Ensemble Learning Using Bagging And Inception-V3 For Anomaly Detection In Surveillance Videos [pdf](https://ieeexplore.ieee.org/abstract/document/9190673)
- #### Discriminative Clip Mining for Video Anomaly Detection [pdf](https://ieeexplore.ieee.org/document/9191072)
- #### A Siamese Network Utilizing Image Structural Differences For Cross-Category Defect Detection [pdf](https://ieeexplore.ieee.org/document/9191128)
- #### CAM-UNET: Class Activation MAP Guided UNET with Feedback Refinement for Defect Segmentation [pdf](https://ieeexplore.ieee.org/document/9190900)
- #### Weakly-Supervised Defect Segmentation Within Visual Inspection Images of Liquid Crystal Displays in Array Process [pdf](https://ieeexplore.ieee.org/document/9190907)

## BMVC2020 Sep 7, 2020
- #### Superpixel Masking and Inpainting for Self-Supervised Anomaly Detection [pdf](https://www.bmvc2020-conference.com/assets/papers/0275.pdf)

## ECCV2020 Aug 23, 2020
- #### Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation [arXiv](https://arxiv.org/abs/2003.08440)
- #### Few-Shot Scene-Adaptive Anomaly Detection [arXiv](https://arxiv.org/abs/2007.07843)
- #### Clustering Driven Deep Autoencoder for Video Anomaly Detection [pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600324.pdf)
- #### Attention Guided Anomaly Localization in Images [arXiv](https://arxiv.org/abs/1911.08616)
- #### Encoding Structure-Texture Relation with P-Net for Anomaly Detection in Retinal Images [arXiv](https://arxiv.org/abs/2008.03632)
- #### Backpropagated Gradient Representations for Anomaly Detection [arXiv](https://arxiv.org/abs/2007.09507)
- #### CLAWS: Clustering Assisted Weakly Supervised Learning with Normalcy Suppression for Anomalous Event Detection [arXiv](https://arxiv.org/abs/2011.12077)
- #### Neural Batch Sampling with Reinforcement Learning for Semi-Supervised Anomaly Detection [pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710749.pdf)
- #### ALRe: Outlier Detection for Guided Refinement [pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520766.pdf)
- #### Handcrafted Outlier Detection Revisited [pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640766.pdf)
- #### OID: Outlier Identifying and Discarding in Blind Image Deblurring [pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700596.pdf)
- #### Rotational Outlier Identification in Pose Graphs Using Dual Decomposition [pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750392.pdf)

## ICML2020 Jul 13, 2020
- #### Interpretable, Multidimensional, Multimodal Anomaly Detection with Negative Sampling for Detection of Device Failure [arXiv](https://arxiv.org/abs/2007.10088)
- #### Robust Outlier Arm Identification [arXiv](https://arxiv.org/abs/2009.09988)
<!--## ICCV2020 Jun 22, 2020-->

## CVPR2020 Jun 14, 2020
- #### Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings [arXiv](https://arxiv.org/abs/1911.02357)
- #### Graph Embedded Pose Clustering for Anomaly Detection [arXiv](https://arxiv.org/abs/1912.11850)
- #### Learning Memory-guided Normality for Anomaly Detection [arXiv](https://arxiv.org/abs/2003.13228)
- #### Self-trained Deep Ordinal Regression for End-to-End Video Anomaly Detection [arXiv](https://arxiv.org/abs/2003.06780)
- #### Background Data Resampling for Outlier-Aware Classification [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Background_Data_Resampling_for_Outlier-Aware_Classification_CVPR_2020_paper.pdf)

## ICLR2020 Apr 26, 2020
- #### Iterative energy-based projection on a normal data manifold for anomaly localization [arXiv](https://arxiv.org/abs/2002.03734)
- #### Deep Semi-Supervised Anomaly Detection [arXiv](https://arxiv.org/abs/1906.02694)
- #### Classification-Based Anomaly Detection for General Data [arXiv](https://arxiv.org/abs/2005.02359)
- #### Robust Subspace Recovery Layer for Unsupervised Anomaly Detection [arXiv](https://arxiv.org/abs/1904.00152)
- #### Robust Anomaly Detection and Backdoor Attack Detection Via Differential Privacy [arXiv](https://arxiv.org/abs/1911.07116)

## AAAI2020 Feb 7, 2020
- #### MixedAD: A Scalable Algorithm for Detecting Mixed Anomalies in Attributed Graphs [pdf](https://ojs.aaai.org//index.php/AAAI/article/view/5482)
- #### Detecting semantic anomalies [arXiv](https://arxiv.org/abs/1908.04388)
- #### Multi-scale Anomaly Detection on Attributed Networks [arXiv](https://arxiv.org/abs/1912.04144)
- #### Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection [pdf](https://ojs.aaai.org//index.php/AAAI/article/view/6068)
- #### MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams [arXiv](https://arxiv.org/abs/1911.04464)
- #### Adaptive Double-Exploration Tradeoff for Outlier Detection [arXiv](https://arxiv.org/abs/2005.06092)
- #### Outlier Detection Ensemble with Embedded Feature Selection [arXiv](https://arxiv.org/abs/2001.05492)

# 2019
## NeurIPS2019 Dec 8, 2019
- #### Transfer Anomaly Detection by Inferring Latent Domain Representations [pdf](https://papers.nips.cc/paper/2019/hash/7895fc13088ee37f511913bac71fa66f-Abstract.html)
- #### Statistical Analysis of Nearest Neighbor Methods for Anomaly Detection [arXiv](https://arxiv.org/abs/1907.03813)
- #### PIDForest: Anomaly Detection via Partial Identification [arXiv](https://arxiv.org/abs/1912.03582)
- #### Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network [pdf](https://papers.nips.cc/paper/2019/hash/6c4bb406b3e7cd5447f7a76fd7008806-Abstract.html)
- #### Quantum Entropy Scoring for Fast Robust Mean Estimation and Improved Outlier Detection [arXiv](https://arxiv.org/abs/1906.11366)
- #### Outlier Detection and Robust PCA Using a Convex Measure of Innovation [pdf](https://proceedings.neurips.cc/paper/2019/hash/e9287a53b94620249766921107fe70a3-Abstract.html)

## ICCV2019 Oct 27, 2019
- #### Anomaly Detection in Video Sequence with Appearance-Motion Correspondence [arXiv](https://arxiv.org/abs/1908.06351)
- #### Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection [arXiv](https://arxiv.org/abs/1904.02639)
- #### GODS: Generalized One-class Discriminative Subspaces for Anomaly Detection [arXiv](https://arxiv.org/abs/1908.05884)

## ICIP2019 Sep 22, 2019
- #### Detection of Small Anomalies on Moving Background [pdf](https://ieeexplore.ieee.org/document/8803176)
- #### Temporal Convolutional Network with Complementary Inner Bag Loss for Weakly Supervised Anomaly Detection [pdf](https://ieeexplore.ieee.org/document/8803657)
- #### DefectNET: multi-class fault detection on highly-imbalanced datasets [arXiv](https://arxiv.org/abs/1904.00863)
- #### Directional-Aware Automatic Defect Detection in High-Speed Railway Catenary System [pdf](https://ieeexplore.ieee.org/abstract/document/8803483)
- #### An Effective Adversarial Training Based Spatial-Temporal Network for Abnormal Behavior Detection [pdf](https://ieeexplore.ieee.org/document/8803571)
- #### Semi-Supervised Robust One-Class Classification in RKHS for Abnormality Detection in Medical Images [pdf](https://ieeexplore.ieee.org/document/8803816)

## BMVC2019 Sep 9, 2019
- #### Motion-Aware Feature for Improved Video Anomaly Detection [arXiv](https://arxiv.org/abs/1907.10211)
- #### Hybrid Deep Network for Anomaly Detection [arXiv](https://arxiv.org/abs/1908.06347)

## IJCAI2019 Aug 10, 2019
- #### AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based Temporal GCN [pdf](https://www.ijcai.org/proceedings/2019/614)
- #### BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series [pdf](https://www.ijcai.org/proceedings/2019/616)
- #### LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs [pdf](https://www.ijcai.org/proceedings/2019/658)
- #### Margin Learning Embedded Prediction for Video Anomaly Detection with A Few Anomalies [pdf](https://www.ijcai.org/proceedings/2019/419)
- #### A Decomposition Approach for Urban Anomaly Detection Across Spatiotemporal Data [pdf](https://www.ijcai.org/proceedings/2019/837)
- #### Outlier Detection for Time Series with Recurrent Autoencoder Ensembles [pdf](https://www.ijcai.org/proceedings/2019/378)

## CVPR2019 Jun 16, 2019
- #### Graph Convolutional Label Noise Cleaner: Train a Plug-and-play Action Classifier for Anomaly Detection [arXiv](https://arxiv.org/abs/1903.07256)
- #### Object-centric Auto-encoders and Dummy Anomalies for Abnormal Event Detection in Video [arXiv](https://arxiv.org/abs/1812.04960)
- #### ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries With Anomalous Features [pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_ManTra-Net_Manipulation_Tracing_Network_for_Detection_and_Localization_of_Image_CVPR_2019_paper.html)
- #### MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection [pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html)
- #### Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos [arXiv](https://openaccess.thecvf.com/CVPR2019?day=2019-06-20)
- #### Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset [arXiv](https://arxiv.org/abs/1904.08486)

## ICML2019 Jun 9, 2019
- #### Anomaly Detection With Multiple-Hypotheses Predictions [arXiv](https://arxiv.org/abs/1810.13292)

## ICLR2019 May 6, 2019
- #### Deep Anomaly Detection with Outlier Exposure [arXiv](https://arxiv.org/abs/1812.04606)

## AAAI2019 Jan 27, 2019
- #### Temporal anomaly detection: calibrating the surprise [arXiv](https://arxiv.org/abs/1705.10085)
- #### Robust Anomaly Detection in Videos Using Multilevel Representations [pdf](https://ojs.aaai.org//index.php/AAAI/article/view/4456)
- #### Multi-View Anomaly Detection: Neighborhood in Locality Matters [pdf](https://ojs.aaai.org//index.php/AAAI/article/view/4418)
- #### Learning Competitive and Discriminative Reconstructions for Anomaly Detection [arXiv](https://arxiv.org/abs/1903.07058)
- #### A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data [arXiv](https://arxiv.org/abs/1811.08055)
- #### Uncovering Specific-Shape Graph Anomalies in Attributed Graphs [pdf](https://ojs.aaai.org//index.php/AAAI/article/view/4483)
- #### Robustness Can Be Cheap: A Highly Efficient Approach to Discover Outliers under High Outlier Ratios [pdf](https://ojs.aaai.org//index.php/AAAI/article/view/4468)
- #### Embedding-Based Complex Feature Value Coupling Learning for Detecting Outliers in Non-IID Categorical Data [pdf](https://ojs.aaai.org//index.php/AAAI/article/view/4495)

## Other important papers published in 2019 / その他2019年の重要論文
- #### f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks (Medical Image Analysis · January 2019) [pdf](https://www.researchgate.net/publication/330796048_f-AnoGAN_Fast_Unsupervised_Anomaly_Detection_with_Generative_Adversarial_Networks) [実装(著者)](https://github.com/tSchlegl/f-AnoGAN)
  - **Domain:** image / **Dataset:** 網膜のボリュームデータ(AnoGAN の論文と同様のもの) / **Index Terms:** GAN, WGAN  
  - GANベースの異常検出手法 f-AnoGAN を提案。GAN のアーキテクチャには WGAN を採用。入力画像 x からランダムノイズ z のマッピングのため、Encoder を導入。Encoder 学習の損失関数には izi_f を使用、izi_f は 入力画像 x と生成画像 x’ の差と、Discriminator の中間層の出力の差を元に計算される。AE, ALI, iterative ベースの手法と比較し、高いROCAUC。


# 2018
## NeurIPS2018 Dec 2, 2018
- #### Deep Anomaly Detection Using Geometric Transformations [arXiv](https://arxiv.org/abs/1805.10917)
- #### A loss framework for calibrated anomaly detection [pdf](https://papers.nips.cc/paper/2018/hash/959a557f5f6beb411fd954f3f34b21c3-Abstract.html)
- #### Efficient Anomaly Detection via Matrix Sketching [arXiv](https://arxiv.org/abs/1804.03065)
- #### A Practical Algorithm for Distributed Clustering and Outlier Detection [arXiv](https://arxiv.org/abs/1805.09495)

## ACCV2018 Dec 2, 2018 [list](https://link.springer.com/book/10.1007/978-3-030-20887-5)
- #### A Defect Inspection Method for Machine Vision Using Defect Probability Image with Deep Convolutional Neural Network [pdf](https://www.semanticscholar.org/paper/A-Defect-Inspection-Method-for-Machine-Vision-Using-Jang-Yun/53b901258cfd4e4741ae9ae176977c2525621a0d)
- #### AVID: Adversarial Visual Irregularity Detection [arXiv](https://arxiv.org/abs/1805.09521)
- #### GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training [arXiv](https://arxiv.org/abs/1805.06725) [実装(著者)](https://github.com/samet-akcay/ganomaly)
  - **Domain:** Image / **Dataset:** MNIST, CIFAR10, UBA, FFOB / **Index Terms:** GAN, 
  - GANベースの異常検出手法 GANomaly を提案。Generator に encoder-decoder-encoderアーキテクチャを導入。最初の Encoder はテスト画像 x から z(ランダムノイズ)を生成、Decoder は z から訓練データ(正常画像)に近い画像 x’ を生成、2つめの Encoder は x’ から z に近いランダムノイズ z’ を生成する。 (z, z’)に関する Encoder Loss、 (x, x’)に関する Contextual Loss、Discriminator の出力(Softmaxの前)に関する Adversarial Loss、3種類の損失関数の重み付き和で Generator を訓練する。AnoGAN, EGBAD と比較し、ほぼすべてのデータセットで良い精度。

- #### Detecting Anomalous Trajectories via Recurrent Neural Networks [pdf](https://faculty.ucmerced.edu/mhyang/papers/accv2018_anomaly.pdf)

## ICIP2018 Oct 7, 2018
- #### Reducing Anomaly Detection in Images to Detection in Noise [arXiv](https://arxiv.org/abs/1904.11276)
- #### Abnormal Event Detection in Videos using Spatiotemporal Autoencoder [arXiv](https://arxiv.org/abs/1701.01546)
- #### Investigating Cross-Dataset Abnormality Detection in Endoscopy with A Weakly-Supervised Multiscale Convolutional Neural Network [pdf](https://ieeexplore.ieee.org/document/8451673)
- #### Fast Surface Defect Detection Using Improved Gabor Filters [pdf](https://ieeexplore.ieee.org/document/8451351)

## ECCV2018 Sep 8, 2018
関連論文なし

## BMVC2018 Sep 3, 2018
リスト見つからず

## IJCAI2018 Jul 13, 2018
- #### ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on Attributed Networks [pdf](https://www.ijcai.org/proceedings/2018/488)
- #### Deep into Hypersphere: Robust and Unsupervised Anomaly Discovery in Dynamic Networks [pdf](https://www.ijcai.org/proceedings/2018/378)
- #### Contextual Outlier Interpretation [arXiv](https://arxiv.org/abs/1711.10589)

## ICML2018 Jul 10, 2018
関連論文なし

## CVPR2018 Jun 18, 2018
- #### Real-world Anomaly Detection in Surveillance Videos [arXiv](https://arxiv.org/abs/1801.04264)
- #### Future Frame Prediction for Anomaly Detection -- A New Baseline [arXiv](https://arxiv.org/abs/1712.09867)

## ICLR2018 Apr 30, 2018
- #### Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection [pdf](https://openreview.net/forum?id=BJJLHbb0-)

## AAAI2018 Feb 2, 2018
- #### Latent Discriminant Subspace Representations for Multi-View Outlier Detection [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/11826)
- #### Non-Parametric Outliers Detection in Multiple Time Series A Case Study: Power Grid Data Analysis [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/11632)
- #### Partial Multi-View Outlier Detection Based on Collective Learning [pdf](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17166)
- #### Sparse Modeling-Based Sequential Ensemble Learning for Effective Outlier Detection in High-Dimensional Numeric Data [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/11692)

## Other important papers published in 2018 / その他2018年の重要論文
- #### EFFICIENT GAN-BASED ANOMALY DETECTION (ICLR2018 workshop) [arXiv](https://arxiv.org/abs/1802.06222)
  - **Domain:** image / **Dataset:** MNIST, KDD99 / **Index Terms:** GAN, Encoder  
  - GANベースの異常検出手法を提案(Efficient GAN, EGBADと呼ばれている)。AnoGANではテスト画像 x に対応する z(ランダムノイズ)を勾配降下法によって更新しながら求めていたが、この論文では、x から 直接 z を生成する Encoder を導入した。AnoGAN と比較し、性能の向上、推論時間は700~900倍速くなった。

# 2017

## important papers published in 2017 / 2017年の重要論文
- #### Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (accepted by  IPMI 2017) [arXiv](https://arxiv.org/pdf/1703.05921.pdf)
  - **Domain:** image / **Dataset:** 網膜のボリュームデータ(clinical high resolution SD-OCT volumes of the retina) / **Index Terms:** GAN  
  - GANを利用した異常検出手法(この論文が初出？)AnoGANを提案。GANに正常画像のみを学習させると、Generator Gはz(ランダムノイズ)から正常画像のみを生成する。テストデータ x が異常画像である場合、x と生成された画像には差分が生まれる。x に対応する z は residural loss と discrimination loss を元に誤差逆伝搬法で更新しながら、求めていく。
