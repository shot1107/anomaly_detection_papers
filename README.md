# anomaly_detection_papers
summary of anomaly detection papers accepted at top conferences in CV, ML, and AI(CVPR, NeurIPS, ICCV, ECCV, AAAI, ICML, IJCAI, ICLR, ICIP, BMVC, ACCV)   
(My main interest is in the two-class classification of abnormal/normal and segmentation of abnormal areas)

CV・ML・AI 分野の Top Conferences(CVPR, NeurIPS, ICCV, ECCV, AAAI, ICML, IJCAI, ICLR, ICIP, BMVC, ACCV) で Accept された Anomaly Detection 関連の論文をまとめる  
(主な自分の興味は、画像を対象とした異常・正常の２クラス分類や異常箇所のセグメンテーション)  
検索ワード : anom*, abnom*, defect, outlier 

# 2022
## NeurIPS2022 Nov 26, 2022 [link](https://nips.cc/Conferences/2022/Schedule?type=Poster) TODO: add pdf links
- #### Perturbation Learning Based Anomaly Detection [arXiv](https://arxiv.org/abs/2206.02704)
- #### SoftCore: Unsupervised Anomaly Detection with Noisy Data
- #### Dual-discriminative Graph Neural Network for Imbalanced Graph-level Anomaly Detection
- #### A Unified Model for Multi-class Anomaly Detection [arXiv](https://arxiv.org/abs/2206.03687)
- #### Few-Shot Fast-Adaptive Anomaly Detection 


## ECCV2022 Oct 24, 2022 [link](https://huggingface.co/spaces/ECCV2022/ECCV2022_papers) TODO: add pdf links
- #### Registration based Few-Shot Anomaly Detection [arXiv](http://arxiv.org/abs/2207.07361)
- #### Pixel-wise Energy-biased Abstention Learning for Anomaly Segmentation on Complex Urban Driving Scenes [arXiv](http://arxiv.org/abs/2111.12264) [実装(著者)](https://github.com/tianyu0207/PEBAL)
- #### Towards Open Set Video Anomaly Detection [arXiv](http://arxiv.org/abs/2208.11113)
- #### Self-Supervised Sparse Representation for Video Anomaly Detection
- #### Locally Varying Distance Transform for Unsupervised Visual Anomaly Detection
- #### SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation [arXiv](http://arxiv.org/abs/2207.14315)
- #### Hierarchical Semi-Supervised Contrastive Learning for Contamination-Resistant Anomaly Detection [arXiv](http://arxiv.org/abs/2207.11789) [実装(著者)](https://github.com/GaoangW/HSCL)
- #### Scale-aware Spatio-temporal Relation Learning for Video Anomaly Detection
- #### Dynamic Local Aggregation Network with Adaptive Clusterer for Anomaly Detection [arXiv](http://arxiv.org/abs/2207.10948) [実装(著者)](https://github.com/Beyond-Zw/DLAN-AC)
- #### Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles [arXiv](http://arxiv.org/abs/2207.10172)
- #### DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition [arXiv](http://arxiv.org/abs/2207.02606)
- #### Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization [arXiv](http://arxiv.org/abs/2109.15222) [実装(著者)](https://github.com/hmsch/natural-synthetic-anomalies)
- #### DSR -- A dual subspace re-projection network for surface anomaly detection [arXiv](http://arxiv.org/abs/2208.01521) [実装(著者)](https://github.com/VitjanZ/DSR)
   - **Domain:** Image / **Dataset:** KSDD2, MVTec AD / **Index Terms:**  vector quantization, encoder-decoder, ResNet, UNet <details>
      - VQ-VAE 等で用いられているベクトル量子化をとり入れた encoder-decoderベースのアーキテクチャ dual subspace re-projection network (DSR) を提案。Decoder は２種類用いられ、１つ目は general object appearance decoder (あらゆる自然画像の再構成に対応する, ImageNetを使用) 、２つ目は object-specific decoder  (あるオブジェクト(カテゴリ)の再構成に特化する)。異常箇所の可視化は、general object appearance decoder によって得られた画像 I_gen と object-specific decoder によって得られた画像 I_spc をチャネル方向に結合(この操作を行うモジュールは anomaly detection module と論文中で定義されている)したデータを Unetベースのセグメンテーションモデルに入力することにより行われる。訓練は３つのステップからなる。ステップ１は、VQ codebook(画像から得られた特徴ベクトルの量子化に用いられる)、general object appearance decoder の訓練が行われる。ステップ２は、anomaly detection module と object specific appearance decoder の訓練が行われる。ここでは、量子化特徴空間を利用し生成された異常画像も用いられる。ステップ３は anomaly detection module で得られた 異常箇所のマスク画像をアップサンプリングする目的で行われる。Dream, CutPaste 等と比較し優れた性能。MVTec AD で 98.2% のAUROC。
      - 擬似的な異常画像の生成をかなり工夫した Self-Supervised Learning ベースの手法ともいえる。SSL ベースの代表的な手法である CutPaste に対しては、補助的なデータセットに依存すること・異常生成のプロセスに依存すること・分布に近い異常の生成が困難であることを問題点としてあげている。手法を正確に理解するには [VQ-VAE論文](https://arxiv.org/abs/1711.00937)は必読そう。
      ![Screenshot from 2022-09-17 08-18-15](https://user-images.githubusercontent.com/30290195/190829841-cbd3d0f1-088a-4633-9237-446ae37d54c5.png)
     </details>


## ICIP2022 Oct 16, 2022 [link](https://cmsworkshops.com/ICIP2022/papers/accepted_papers.php) TODO: add pdf links
- #### A NOVEL CONTRASTIVE LEARNING FRAMEWORK FOR SELF-SUPERVISED ANOMALY DETECTION
- #### Anomalib: A Deep Learning Library for Anomaly Detection [arXiv](https://arxiv.org/abs/2202.08341)
- #### Automatic defect segmentation by unsupervised anomaly learning [arXiv](https://arxiv.org/abs/2202.02998)
- #### Multifractal anomaly detection in images via space-scale surrogates [pdf](https://www.archives-ouvertes.fr/hal-03735492/)
- #### Object-centric and memory-guided normality reconstruction for video anomaly detection [arXiv](https://arxiv.org/abs/2203.03677)
- #### PGTNet: Prototype Guided Transfer Network for Few-shot Anomaly Localization
- #### PPT: ANOMALY DETECTION DATASET OF PRINTED PRODUCTS WITH TEMPLATES
- #### REAL-WORLD VIDEO ANOMALY DETECTION BY EXTRACTING SALIENT FEATURES
- #### Subspace Modeling for Fast Out-Of-Distribution and Anomaly Detection [arXiv](https://arxiv.org/abs/2203.10422)
- #### THE BRIO-TA DATASET: UNDERSTANDING ANOMALOUS ASSEMBLY PROCESS IN MANUFACTURING
- #### TRANSFORMER BASED SELF-CONTEXT AWARE PREDICTION FOR FEW-SHOT ANOMALY DETECTION IN VIDEOS
- #### UNSUPERVISED ANOMALY DETECTION WITH SELF-TRAINING AND KNOWLEDGE DISTILLATION
- #### EXPLORING ACTIVE LEARNING FOR SEMICONDUCTOR DEFECT SEGMENTATION
- #### Hierarchical Defect Detection based on Reinforcement Learning
- #### MLSA-UNET: END-TO-END MULTI-LEVEL SPATIAL ATTENTION GUIDED UNET FOR INDUSTRIAL DEFECT SEGMENTATION
- #### Ψ-NET IS AN EFFICIENT TINY DEFECT DETECTOR
- #### JOINT CLASSIFICATION AND OUT-OF-DISTRIBUTION DETECTION BASED ON STRUCTURED LATENT SPACE OF VARIATIONAL AUTO-ENCODERS


## IJCAI-ECAI2022 Jul 23, 2022 [link](https://ijcai-22.org/main-track-accepted-papers/)
- #### Anomaly Detection by Leveraging Incomplete Anomalous Knowledge with Anomaly-Aware Bidirectional GANs [arXiv](https://arxiv.org/abs/2204.13335)
- #### GRELEN: Multivariate Time Series Anomaly Detection from the Perspective of Graph Relational Learning [pdf](https://www.ijcai.org/proceedings/2022/0332.pdf)  
- #### Reconstruction Enhanced Multi-View Contrastive Learning for Anomaly Detection on Attributed Networks [arXiv](https://arxiv.org/abs/2205.04816)
- #### HashNWalk: Hash and Random Walk Based Anomaly Detection in Hyperedge Streams [arXiv](https://arxiv.org/abs/2204.13822)
- #### CADET: Calibrated Anomaly Detection for Mitigating Hardness Bias [pdf](https://bhooi.github.io/papers/cadet_ijcai22.pdf)
- #### Understanding and Mitigating Data Contamination in Deep Anomaly Detection: A Kernel-based Approach [pdf](https://www.ijcai.org/proceedings/2022/0322.pdf)
- #### Neural Contextual Anomaly Detection for Time Series [arXiv](https://arxiv.org/abs/2107.07702)
- #### Constrained Adaptive Projection with Pretrained Features for Anomaly Detection [arXiv](https://arxiv.org/abs/2112.02597)
- #### Raising the Bar in Graph-level Anomaly Detection [arXiv](https://arxiv.org/abs/2205.13845)
- #### Can Abnormality be Detected by Graph Neural Networks? [pdf](http://yangy.org/works/gnn/IJCAI22_Abnormality.pdf)


## ICML2022 Jul 17, 2022 [link](https://icml.cc/Conferences/2022/AcceptedPapersInitial)
- #### Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection [pdf](https://proceedings.mlr.press/v162/chen22x.html)
- #### Rethinking Graph Neural Networks for Anomaly Detection [arXiv](https://arxiv.org/abs/2205.15508)
- #### Latent Outlier Exposure for Anomaly Detection with Contaminated Data [arXiv](https://arxiv.org/abs/2202.08088)
- #### FITNESS: (Fine Tune on New and Similar Samples) to detect anomalies in streams with drift and outliers [pdf](https://abishek90.github.io/fitness_ad.pdf)



## CVPR2022 Jun 24, 2022 (final dicision: March 2, 2022) [link](https://openaccess.thecvf.com/CVPR2022?day=all)
- #### Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection [arXiv](http://arxiv.org/abs/2111.09099)
- #### Anomaly Detection via Reverse Distillation From One-Class Embedding [arXiv](http://arxiv.org/abs/2201.10703)
- #### Bayesian Nonparametric Submodular Video Partition for Robust Anomaly Detection [arXiv](http://arxiv.org/abs/2203.12840)
- #### Towards Total Recall in Industrial Anomaly Detection [arXiv](http://arxiv.org/abs/2106.08265)
- #### Catching Both Gray and Black Swans: Open-Set Supervised Anomaly Detection [arXiv](http://arxiv.org/abs/2203.14506)
- #### Deep Anomaly Discovery From Unlabeled Videos via Normality Advantage and Self-Paced Refinement [pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.html)
- #### Learning Second Order Local Anomaly for General Face Forgery Detection [pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Fei_Learning_Second_Order_Local_Anomaly_for_General_Face_Forgery_Detection_CVPR_2022_paper.html)
- #### Deep Decomposition for Stochastic Normal-Abnormal Transport [arXiv](http://arxiv.org/abs/2111.14777)
- #### Semiconductor Defect Detection by Hybrid Classical-Quantum Deep Learning [pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Semiconductor_Defect_Detection_by_Hybrid_Classical-Quantum_Deep_Learning_CVPR_2022_paper.html)
- #### Robust Outlier Detection by De-Biasing VAE Likelihoods [arXiv](http://arxiv.org/abs/2108.08760)


## ICLR2022 Apr 29, 2022 [link](https://openreview.net/group?id=ICLR.cc/2022/Conference)
- #### Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy [arXiv](https://arxiv.org/abs/2110.02642)
- #### Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series [pdf](https://openreview.net/forum?id=45L_dgP48Vd)
- #### Anomaly Detection for Tabular Data with Internal Contrastive Learning [pdf](https://openreview.net/forum?id=_hszZbt46bT)
- #### Igeood: An Information Geometry Approach to Out-of-Distribution Detection [pdf](https://openreview.net/forum?id=mfwdY3U_9ea)
- #### VOS: Learning What You Don't Know by Virtual Outlier Synthesis [arXiv](https://arxiv.org/abs/2202.01197)

## AAAI2022 Mar 1, 2022 [link](https://aaai.org/Conferences/AAAI-22/wp-content/uploads/2021/12/AAAI-22_Accepted_Paper_List_Main_Technical_Track.pdf)
- #### A Causal Inference Look at Unsupervised Video Anomaly Detection [pdf](https://www.aaai.org/AAAI22Papers/AAAI-37.LinX.pdf)
- #### Comprehensive Regularization in a Bi-Directional Predictive Network for Video Anomaly Detection [pdf](https://www.aaai.org/AAAI22Papers/AAAI-470.ChenC.pdf)
- #### Towards a Rigorous Evaluation of Time-series Anomaly Detection [arXiv](https://arxiv.org/abs/2109.05257)
- #### Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection [pdf](https://www.aaai.org/AAAI22Papers/AAAI-6637.LiS.pdf)
- #### Unsupervised Anomaly Detection by Robust Density Estimation [pdf](https://www.aaai.org/AAAI22Papers/AAAI-11219.LiuB.pdf)
- #### Transferring the Contamination Factor between Anomaly Detection Domains by Shape Similarity [pdf](https://www.aaai.org/AAAI22Papers/AAAI-12660.PeriniL.pdf)
- #### Calibrated Nonparametric Scan Statistics for Anomalous Pattern Detection in Graphs [pdf](https://www.aaai.org/AAAI22Papers/AAAI-12900.ChunpaiW.pdf)
- #### LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks [arXiv](https://arxiv.org/abs/2112.05355)

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
   - **Domain:** Image / **Dataset:** MNIST, FMNIST, CIFAR10, Fastener / **Index Terms:** Autoencoder, RotNet  <details>
      - Deep convolutional autoencoder（DCAE)が異常箇所までも再構成してしまう問題を解決するため、2ステージで学習を行う非対称DCAEを提案。まず、RotNet をEncoderとして学習し、学習済みRotNetを凍結したまま2種類のDecoderを学習(1つはloss関数にMSEを、もう1つはSSIMを使用)。2つのDecoderの結果を組み合わせるて最終的な異常スコアを得る。FMNIST, Fastenerにおいてベースライン手法より優れた性能。
     </details>
- #### Anomaly Detection via Self-organizing Map [arXiv](https://arxiv.org/abs/2107.09903)
- #### Deep Unsupervised Image Anomaly Detection: An Information Theoretic Framework [arXiv](https://arxiv.org/abs/2012.04837)
- #### Joint Anomaly Detection and Inpainting for Microscopy Images Via Deep Self-Supervised Learning [pdf](https://ieeexplore.ieee.org/abstract/document/9506454)
- #### Multi-Scale Background Suppression Anomaly Detection In Surveillance Videos [pdf](https://ieeexplore.ieee.org/document/9506580)
- #### Particle Swarm And Pattern Search Optimisation Of An Ensemble Of Face Anomaly Detectors [pdf](https://ieeexplore.ieee.org/document/9506251)
- #### SAGAN: Skip-Attention GAN For Anomaly Detection [pdf](https://ieeexplore.ieee.org/abstract/document/9506332)
   - **Domain:** Image / **Dataset:** CIFAR-10, LBOT(独自) / **Index Terms:**  GAN, Attention  <details>
      - Skip-GANomaly をベースに Attention モジュールを追加した、異常検知手法 Skip-Attention GAN (SAGAN) を提案。Attention モジュールは、異常が現れる局所的な領域に注目することを目的としている。具体的な構造は、U-Net 型のネットワークで Skip connection の前に CBAM(convolutional block attention module) が挿入されているような形になっている。EGBAD, GANomaly, Skip-GANomaly と比較し、AUC が大幅に向上。
     </details>
   
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
   - **Domain:** Image / **Dataset:** MVTec AD / **Index Terms:**  Self-Supervised Learning, Data Augmentation, ResNet, EfficientNet  <details>
      - CutPaste という画像中の長方形領域を別の場所にペーストしたような data augmentation を提案、CutPaste によって生成した擬似的な異常画像を用いて、自己教師あり学習として欠陥箇所の表現を学習する。画像の異常スコアは、(獲得した表現上での) Gaussian density estimator を用いて計算される。Localization (≒セグメンテーション)では、パッチ単位(切り取られた部分画像)で学習した表現を利用する。テスト時は、スライドしながら画像のパッチを取得していき、それぞれのパッチで異常スコアを計算していく。そうすると n×n (nは入力画像サイズより小さい)の異常マップが作成でき、それを Gaussian smoothing を用いて upsampling することで元の画像サイズの異常マップを生成する。MVTec AD データセットで当時の SoTA を達成。
169
   - アイディアはシンプルだが精度向上のための細かい詰めがすごい。
   ![Screenshot from 2022-09-17 08-20-46](https://user-images.githubusercontent.com/30290195/190829866-4a592e1d-9531-477e-a424-9f4014374f57.png)
     </details>
   
- #### Pixel-Wise Anomaly Detection in Complex Driving Scenes [arXiv](http://arxiv.org/abs/2103.05445)
- #### PANDA: Adapting Pretrained Features for Anomaly Detection and Segmentation [arXiv](http://arxiv.org/abs/2010.05903)
- #### Glancing at the Patch: Anomaly Localization With Global and Local Feature Comparison [pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Glancing_at_the_Patch_Anomaly_Localization_With_Global_and_Local_CVPR_2021_paper.pdf)
- #### Anomaly Detection in Video via Self-Supervised and Multi-Task Learning [arXiv](http://arxiv.org/abs/2011.07491)
- #### Multiresolution Knowledge Distillation for Anomaly Detection [arXiv](http://arxiv.org/abs/2011.11108)
- #### Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark [pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Yao_Joint-DetNAS_Upgrade_Your_Detector_With_NAS_Pruning_and_Dynamic_Distillation_CVPR_2021_paper.pdf)

## ICLR 2021 May 3, 2021
- #### Explainable Deep One-Class Classification [arXiv](https://arxiv.org/abs/2007.01760)
   - **Domain:** Image / **Dataset:**  Fashion-MNIST, CIFAR-10, ImageNet, MVTec AD / **Index Terms:**  one-class classification(OCC), transposed Gaussian convolution  <details>
      - 説明可能(≒可視化可能)な異常検知手法、Fully Convolutional Data Description (FCDD) を提案。Fully Convolutional Network (FCN, 全結合層を含まないCNN) と Hypersphere Classifier (HSC, 正常データと異常データを用いて分布を学習、正常データは中心近くに、異常データは中心から遠くにマッピングされる) の考え方を元にしている。FCDD は R^(c×h×w) → R^(u×v) のマッピングを学習し、ダウンサンプリングされた異常マップを生成する(元画像から小さな異常マップを生成するイメージ)。学習には異常データも用いられるが、MVTec AD のような異常データが訓練データに含まれないデータセットを用いる場合は、シミのようなノイズを正常データに埋め込むことで、擬似的な異常データを生成している。その異常マップをガウシアンカーネルを用いた転置畳み込みを用いてアップサンプリングし、最終的な異常マップを得る。AnoGAN や VAE 等のベースラインの AUC を上回っており、半教師あり(異常データを少数用いる)で学習することで AUC が向上することも確認している。
     </details>

## AAAI2021 Feb 2, 2021
- #### LREN: Low-Rank Embedded Network for Sample-Free Hyperspectral Anomaly Detection [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16536)
- #### GAN Ensemble for Anomaly Detection [arXiv](https://arxiv.org/abs/2012.07988)
  - **Domain:** Image / **Dataset:** MNIST, CIFAR-10, OCT, KDD99 / **Index Terms:** GAN, Ensemble  <details>
      - GANの学習には不安定性やモード崩壊などいくつかの欠点があるが、近年の研究でGeneratorやDiscriminatorを複数用意することでそれらの問題を解決できることが示されている。この論文では、複数のGenerator(実際にはEncoder-Decoder)とDiscriminatorを用意し、それらをアンサンブルすることで画像の異常検出問題を解決している。訓練時にはGeneratorは複数のDiscriminatorからフィードバックを受け、Discriminatorは複数のGeneratorの出力を識別する。推論時はすべてネットワーク(Encoder-Decoder⇨Discriminator)の出力の平均をとる。ベースモデルとして、f-AnoGAN・EGBAD・GANomaly・Skip-GANomalyを使用し、すべてのデータセットで単体モデルよりアンサンブルの方が優れた性能。
     </details> 
  
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

## Other important papers published in 2020 / その他2020年の重要論文
- #### Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection (ICPR2020) [arXiv](https://arxiv.org/abs/2005.14140)
   - **Domain:** Image / **Dataset:** MVTec AD / **Index Terms:**  EfficientNet, Multivariate Gaussian, Mahalanobis Distance  <details>
      - ImageNet によって事前学習された EfficientNet をそのまま利用し、その特徴表現を用いて異常検知を行う手法を提案。学習パートでは、正常画像のみを用い、EfficientNet に正常画像を与えたときに生成された特徴ベクトルを近似する多変量ガウス分布を求める。推論パートでは、テスト画像をモデルに入力して得られた特徴表現と分布の平均とのマハラノビス距離を求め、それを元に異常度を計算する。モデルの再訓練をしていないにも関わらず、95.8という高いAUROCを達成。
     </details>

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
   - **Domain:** Image / **Dataset:** MVTec AD / **Index Terms:**  Dataset  <details>
      - 画像の異常検出用のデータセット、MVTec Anomaly Detection (MVTec AD) の紹介論文。5種類のテクスチャカテゴリのデータと、10種類のオブジェクトカテゴリのデータからなり、全部で5354枚の画像が含まれる。セグメンテーション(ピクセル単位の欠陥箇所特定)タスク用に、欠陥箇所を示した grand truth 画像も提供される。AE や AnoGAN等の基本的な手法を用いた実験も行っている。
327
   - 2020~に発表された画像の異常検出を扱った論文では、このデータセットが使用されていることがかなり多い。
     </details>
   
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
  - **Domain:** image / **Dataset:** 網膜のボリュームデータ(AnoGAN の論文と同様のもの) / **Index Terms:** GAN, WGAN  <details>
      - GANベースの異常検出手法 f-AnoGAN を提案。GAN のアーキテクチャには WGAN を採用。入力画像 x からランダムノイズ z のマッピングのため、Encoder を導入。Encoder 学習の損失関数には izi_f を使用、izi_f は 入力画像 x と生成画像 x’ の差と、Discriminator の中間層の出力の差を元に計算される。AE, ALI, iterative ベースの手法と比較し、高いROCAUC。
     </details> 
  
- #### Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection (IJCNN 2019) [arXiv](https://arxiv.org/abs/1901.08954) [実装(著者)](https://github.com/samet-akcay/skip-ganomaly)
   - **Domain:** Image / **Dataset:** CIFAR10, UBA, FFOB / **Index Terms:** GAN, encoder-decoder, skip connection  <details>
      - GANomaly と同じ著者の論文。GANomaly では Generator として encoder-decoder が用いられていたが、そこに UNet スタイルの skip connection が導入されている(encoder の i 層目の出力が decoder の n-1 層目に接続されているような構造)。また、Discriminator に関する、Adversarial Loss にも変更が加えられている。AnoGAN, EGBAD, GANomaly と比較し、ほとんどのデータセットで最高の AUC。
359
   - 論文中の再構成後の画像を見ると、異常箇所まで再構成してしまっているように見える、skip connection の影響で再構成能力が高まりすぎている可能性も？
     </details>
  

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
  - **Domain:** Image / **Dataset:** MNIST, CIFAR10, UBA, FFOB / **Index Terms:** GAN  <details>
      - GANベースの異常検出手法 GANomaly を提案。Generator に encoder-decoder-encoderアーキテクチャを導入。最初の Encoder はテスト画像 x から z(ランダムノイズ)を生成、Decoder は z から訓練データ(正常画像)に近い画像 x’ を生成、2つめの Encoder は x’ から z に近いランダムノイズ z’ を生成する。 (z, z’)に関する Encoder Loss、 (x, x’)に関する Contextual Loss、Discriminator の出力(Softmaxの前)に関する Adversarial Loss、3種類の損失関数の重み付き和で Generator を訓練する。AnoGAN, EGBAD と比較し、ほぼすべてのデータセットで最高精度。
     </details>

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
  - **Domain:** image / **Dataset:** MNIST, KDD99 / **Index Terms:** GAN, Encoder  <details>
      - GANベースの異常検出手法を提案(Efficient GAN, EGBADと呼ばれている)。AnoGANではテスト画像 x に対応する z(ランダムノイズ)を勾配降下法によって更新しながら求めていたが、この論文では、x から 直接 z を生成する Encoder を導入した。AnoGAN と比較し、精度の向上、推論時間は700~900倍速くなった。
     </details>


# 2017

## important papers published in 2017 / 2017年の重要論文
- #### Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (accepted by  IPMI 2017) [arXiv](https://arxiv.org/pdf/1703.05921.pdf)
  - **Domain:** image / **Dataset:** 網膜のボリュームデータ(clinical high resolution SD-OCT volumes of the retina) / **Index Terms:** GAN  <details>
      - GANを利用した異常検出手法(この論文が初出？)AnoGANを提案。GANに正常画像のみを学習させると、Generator Gはz(ランダムノイズ)から正常画像のみを生成する。テストデータ x が異常画像である場合、x と生成された画像には差分が生まれる。x に対応する z は residural loss と discrimination loss を元に誤差逆伝搬法で更新しながら、求めていく。
     </details>

