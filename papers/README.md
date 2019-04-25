# 人脸识别：

1. **SymNets**(Domain-Symmetric Networks for Adversarial Domain Adaptation)[[MD]("Domain-Symmetric Networks for Adversarial Domain Adaptation.md"),  [PDF](pdf/Domain-Symmetric Networks for Adversarial Domain Adaptation.pdf)]
2. **SPGAN** (Weijian Deng, Liang Zheng, Qixiang Ye, Guoliang Kang, Yi Yang, Jianbin Jiao; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 994-1003) [[MD](), [PDF]("pdf/Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification.pdf")]
3. **PTGAN** (Longhui Wei, Shiliang Zhang, Wen Gao, Qi Tian; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 79-88) [[MD](), [PDF](./pdf/Person Transfer GAN to Bridge Domain Gap for Person Re-Identification.pdf)]
4. **MAR** (Unsupervised Person Re-identification by Soft Multilabel Learning) [[MD](), [PDF](./pdf/Unsupervised Person Re-identification by Soft Multilabel Learning.pdf)]
5. **ECN** (Invariance Matters - Exemplar Memory for Domain Adaptive Person Re-identification) [[MD](), [PDF](./pdf/Invariance Matters - Exemplar Memory for Domain Adaptive Person Re-identification.pdf)]

# Deblurred

- [Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring. CVPR 2017.](pdf/Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)
- [Online Video Deblurring via Dynamic Temporal Blending Network, ICCV 2017](pdf/Online Video Deblurring via Dynamic Temporal Blending Network.pdf)
- [Learning a Discriminative Prior for Blind Image Deblurring, CVPR 2018.](pdf/Li_Learning_a_Discriminative_CVPR_2018_paper.pdf)
- Single Image Dehazing via Conditional Generative Adversarial Network, CVPR 2018.
- DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks, CVPR 2018.
- Video Enhancement with Task-Oriented Flow, IJCV 2019.
- Deep Video Deblurring for Hand-Held Cameras, CVPR 2017.





## Quantitative evaluations of state-of-the-arts

|               | Blurred | Nah et al. | DeblurGAN | SRN        | SVRNN  |
| ------------- | ------- | ---------- | --------- | ---------- | ------ |
| PSNR          | 25.63   | 28.94      | 25.86     | **30.26**  | 29.19  |
| SSIM          | 0.8584  | 0.8796     | 0.8359    | **0.9342** | 0.9306 |
| Parameters(M) | -       | 11.7       | 11.6      | **8.1**    | 9.2    |
| Runtime       |         | 3.96       | **1.12**  | 3.87       | 1.57   |

