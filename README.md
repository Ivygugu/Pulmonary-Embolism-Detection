
# Pulmonary-Embolism-Detection

## 肺栓塞检测难点
1. CT扫描由数百张图像组成，需要详细检查以识别肺动脉内的凝块；
2. 扫描结果有时会受到环境的影响，如肺部运动伪影与栓子在影像中很难区分等；
3. 假阳性去除需要考虑多方面的原因；
***

## 文档整理
链接: https://pan.baidu.com/s/1o2CExesyUyuXTzCBW0W3fg
提取码: m8jk 
***

## 实验室
1. The AIMI Center: [Center for Artificial Intelligence in Medicine & Imaging](https://aimi.stanford.edu/ )
***

## Datasets 数据集
#### CT图像
| Name    | Size   | object | Link    |
| :--- | :--- | :------ | :------ |
| [1] The RSNA Pulmonary Embolism CT Dataset    | 980.24GB | 2,995,147 targets   | [Downoad](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/data?select=train.csv)           |
| [2] CT Pulmonary Angiography   | 145.13GB    | 408,856 targets  | [Downoad](https://stanfordaimi.azurewebsites.net/datasets/12c02840-2e13-42a2-b4ef-f682472d4694)                |
| [3] RadFusion: Multimodal Pulmonary Embolism Dataset   | 378.85GB    | 1,843 targets  | [Downoad](https://stanfordaimi.azurewebsites.net/datasets/3a7548a4-8f65-4ab7-85fa-3d68c9efc1bd)                |
| [4] CAD-PE   |  12.80 GB    | 该数据集是为ISBI挑战cad-pe而创建 |  [Downoad](https://ieee-dataport.org/open-access/cad-pe)    |
| [5] FUMPE   |  4.36 GB    | 马什哈德菲尔多西大学的PE数据集 |  [Downoad1](https://www.kaggle.com/datasets/andrewmvd/pulmonary-embolism-in-ct-images) [Downoad2](https://figshare.com/collections/FUMPE/4107803)    |
| Pulmonary  Embolism   |      | 来自于 Mashhad 的 Ferdowsi 大学的机器视觉实验室的开源网站 |                 |


## 经典模型
> + [PENet—a scalable deep-learning model for automated diagnosis of pulmonary embolism using volumetric CT imaging](https://www.nature.com/articles/s41746-020-0266-y)
>   - 引用：Huang S C, Kothari T, Banerjee I, et al. PENet—a scalable deep-learning model for automated diagnosis of pulmonary embolism using volumetric CT imaging[J]. NPJ digital medicine, 2020, 3(1): 1-9.
>   - 源码：https://github.com/marshuang80/penet
>   - 数据集：[3] RadFusion: Multimodal Pulmonary Embolism Dataset
***
> + [AANet: Artery-Aware Network for Pulmonary Embolism Detection in CTPA Images](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_45)
>   - 引用：Guo, Jia, et al. "AANet: artery-aware network for pulmonary embolism detection in CTPA images." Medical Image Computing and Computer Assisted Intervention–MICCAI 2022: 25th International Conference, Singapore, September 18–22, 2022, Proceedings, Part I. Cham: Springer Nature Switzerland, 2022.
>   - 源码：https://github.com/guojiajeremy/AANet
>   - 数据集：[4] CAD-PE
***
> + [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
>   - 引用：Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015: 234-241.
***
> + [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)
>   - 引用：Ren S, He K, Girshick R, et al. Faster r-cnn: Towards real-time object detection with region proposal networks[J]. Advances in neural information processing systems, 2015, 28.
***
> + [Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)
>   - 引用：He K, Gkioxari G, Dollár P, et al. Mask r-cnn[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2961-2969.
***


## 硕博论文
#### 国内硕士论文

| 论文题目    | 作者   | 大学 | 发表时间    |
| :--- | :--- | :------ | :------ |
| 基于混合域注意力机制的肺栓塞图像分割研究    | 徐欢 | 吉林大学   | 2020.5.1           |
| 基于深度学习的CTPA肺栓塞图像分割方法研究    | 温洲 | 北京化工   | 2020.6.14          |
| 基于深度学习的肺部CT图像多阶段栓塞检测方法研究    | 苏建超 | 华中科技   | 2020.8.8           |
| 基于弱标记CT影像的新冠肺炎和肺动脉栓塞识别研究    | 刘艺璇 | 华中科技   | 2021.5.20           |
| 基于改进GAC模型及深度学习的肺动脉与栓塞分割方法研究    | 刘珍宏 | 华南理工   | 2021.5.23          |
| 基于多视图加权注意力机制的CTPA肺动脉栓塞图像分割算法研究    | 鲁启洋 | 吉林大学   | 2021.6.30           |
| 基于深度学习的肺栓塞医学图像分割算法研究  | 刘硕 |  杭州电子科技  | 2022.4           |
| 基于深度学习的CTPA肺栓塞识别及评价  | 邵亚君 |  北京化工  | 2022.5         |


> + [基于混合域注意力机制的肺栓塞图像分割研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202002&filename=1020905457.nh&uniplatform=NZKPT&v=-QDNds-Ni2C7f3uTj6YYCbfY0B_mXIrfx1h1HQjF46uuOwDJ04DMcAvpsk5mgV2g)
>   - 作者：徐欢
>   - 联系方式：
>   - 数据集： Pulmonary  Embolism(肺栓塞数据集)
>   - 代码：无
>   - 总结：提出一种混合域注意力机制的肺栓塞分割方法。混合域注意力机制是指分别在空间域和通道域上添加注意力机制，这种机制使模型在学习的过程中更关注在空间和通道中的重要信息，忽略和抑制无关信息，本文对这种注意力机制的实现的方式是利用 scSE 模块对输入特征不断地进行权重的调整。
***
> + [基于深度学习的CTPA肺栓塞图像分割方法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202101&filename=1020152016.nh&uniplatform=NZKPT&v=NZmWLuuvngR1unltaImG-Y8T4yvb7KX4GPUUXSR4qeKgRL6zB-seBBPkkiIbh3MX)
>   - 作者：温洲
>   - 联系方式：wenzhouwww@126.com
>   - 数据集：中日友好医院提供的CTPA影像数据
>   - 代码：无
>   - 总结：深入分析了U-Net分割精度优于FCN的内在本质，结合ResNet和DenseNet的设计思想和基本原理，提出了一种改进U-Net结构，改进的U-Net以残差模块作为基本结构，构建了强大的特征提取网络。并引入中间特征融合模块Concat Block，保留了更多的收缩路径的中间特征，将融合后的特征以通道拼接的方式融合到扩张路径，使得网络获取了更丰富的上下文信息，进一步提升网络的分割精度。结合肺栓塞危险度评价实际场景中的需求，实现了基于Mask RCNN的肺栓塞图像分割方法，该方法能够在定位肺栓塞的位置的同时给出肺栓塞分割结果。在该方法的基础上，使用Group卷积对Mask RCNN进行改进，改进后的方法有效的减少了网络的参数量，优化了网络的推理速度，能够降低30％以上的显存和时长占用，且最终精度近乎不变。
***
> + [基于深度学习的肺部CT图像多阶段栓塞检测方法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202201&filename=1020349904.nh&uniplatform=NZKPT&v=_DZKMpDu2eH53cHBrEKMPWJaKLM4lgzsDfjd00gV17DE60VMfpGsp6Q6c5ojAVRk)
>   - 作者：苏建超
>   - 联系方式：
>   - 数据集：肺栓塞挑战赛数据集、肺栓塞 129 数据集
>   - 代码：无
>   - 总结：本文提出的肺栓塞检测方法包含三个阶段：基于 3D 卷积神经网络（3D CNN)的候选点提取、基于 2D 卷积神经网络(2D CNN)的假阳筛除和基手动静脉分离的假阳筛除。（1）3D CNN 与2DCNN 相结合，利用了3D CNN 提取三维空间信息的能力与 2D CNN 减少分类网络参数量的优势：（2）为每个候选点实现了一个新颖的图像表达，有效将3D数据降维到2D，且最大程度保留了所在血管的图像信息：（3）设计了子树分离算法提取肺血管子树，使神经网络的预测结果在子树内进行校正，获得一致的动静脉分离结果。
***
> + [基于弱标记CT影像的新冠肺炎和肺动脉栓塞识别研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1021914801.nh&uniplatform=NZKPT&v=gOSgFeEqM2G7Am5XgxSCg6uAEMJ03Ufwkx27bKe-wX6VOWDkUX4vQhIaRVm1dPnE)
>   - 作者：刘艺璇
>   - 联系方式：
>   - 数据集： 541个样本数据均来自就诊于本学校附属医院影像科的疑似患者。
>   - 代码：无
>   - 总结：本文研究了弱监督下新冠病灶区定位和轻量化网络设计的问题，通过改进3D 残差网络实现了新冠肺炎准确快速地识别，并利用分类网络中区分特征实现病灶区的定位和可视化。首先，为了减少其他组织的干扰，利用无监督的连通域分析法对肺部进行分割，并将其较好的分割结果作为真实分割注释对分割网络进行训练；然后，设计了轻量级主干网络和渐进式分类器，在保证计算成本低的同时提高新冠肺炎的分类准确性；最后，为了对分类结果进行可解释性说明，并为影像科医生提供更加直观的诊断依据，设计了一种类激活图和三维连通区域相结合的弱监督新冠肺炎病灶定位算法。
***
> + [基于改进GAC模型及深度学习的肺动脉与栓塞分割方法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202201&filename=1022004522.nh&uniplatform=NZKPT&v=O1q1Qdnf7py8WTOKTw3Bf9YxVSjnsCJkHFY2VXWrPvMdGwa1dkoR8_KttFy768cq)
>   - 作者：刘珍宏
>   - 联系方式：Corresponding author: Hongfang Yuan (yuanhf@mail.buct.edu.cn) 
>   - 数据集：中日友好医院提供的CTPA影像数据
>   - 代码：无
>   - 总结：本文提出的用于肺栓塞分割的ResD-Unet架构，它结合了残差连接、密集连接和 U-Net的优点。以U-net网络为基础，将U-net中的普通卷积块替换成残差密集块，残差密集连接将之前所有层的特征都添加到底层，实现特征重用，便于训练过程中梯度的反向传播。有效地解决了信息丢失的问题，避免了梯度消失。批处理归一化层的使用避免了过拟合，加快了训练速度。可以在构建更深的网络的同时改善网络的梯度流通。为克服DiceLoss训练不稳定的问题，结合交叉熵损失和结构相似度损失的设计思想和相关理论，本文提出了一种新的混合损失函数，充分利用三类损失函数的优点，通过混合损失函数提高目标边界的分割准确度。
***
> + [基于多视图加权注意力机制的CTPA肺动脉栓塞图像分割算法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1021893263.nh&uniplatform=NZKPT&v=Ws171w8yb6M0ZC2Vo8F-L_9Mqs0MsMUlig-rWRht3PG4hUk5sXz59Ejor0jQJQOa)
>   - 作者：鲁启洋
>   - 联系方式：Correspondence: Hui Liu: liuhuijiujiu@gmail.com // Xinzhou Xie: xinzhxie@hotmail.com 
>   - 数据集：广东省人民医院提供的CTPA影像数据、伊朗马什哈德Ferdowsi大学机器视觉实验室的开源数据集FUMPE(standing for Ferdowsi University of Mashhad's PE dataset)
>   - 代码：无
>   - 总结：针对二维分割网络对疑似肺动脉栓塞区域易出现误判，三维分割网络消耗计算资源巨大的问题，本文提出了一种基于多视图加权注意力机制的网络结构 （MWA U-Net）。该模型通过三个并联的特征提取网络，分别在三个视图上进行特征提取，通过引入注意力机制，利用自适应权重的主视图间协作以模仿临床的观察诊断，可以有效的提高肺动脉栓塞病灶的分割精度。
***
> + [基于深度学习的肺栓塞医学图像分割算法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1022086347.nh&uniplatform=NZKPT&v=g13KhaNPo0rvc-07zIUzrFKSNoXQKZKk21akKNa_KEulachyXfkWnXBJ6Ocn0aeO)
>   - 作者：刘硕
>   - 联系方式：
>   - 数据集：
>   - 代码：无
>   - 总结：
(1)本章基于在传统 U 型结构模型上得到的结果经过再次训练可以得到更好的效果这一初衷，并引用 DoubleU-Net 的基础框架，旨在肺栓塞的医学图像分割上获得更好的分割表现，采用了 DBAU-Net 结构。 
(2)在编码部分，将第一个编码器的原始结构替换为经过预训练的 VGG-19_bn 结构，VGG-19 将允许接收更深度信息，同时保证感受野的大小，从而提高网络性能，提高更好的输出分割效果。第二个编码器在传统编码结构的基础上加入了 squeeze-and-excite 模块，这使得网络在编码过程中可以更专注于重要的特征区域，进而在这个层面提升了模型的准确度。 
(3)在解码部分，和 U-Net 布局相近，在跳跃连接上由于网络层数的不同进行了修改。对于第一个 U-Net 结构中的解码器，只连接来自其本身编码器的跳跃连接，并且在其中引入了非对称特征融合模块 AFF，使不同尺度的特征信息能够得到更好的利用。然而在第二个 U-Net 结构中的解码器，则同时连接来自两个编码器的跳跃连接（前三层接收来自网络 1 的编码器的连接，所有四层接收网络2 的编码器的连接）。
(4)其次，在每个下采样阶段和上采样阶段之间，加入了深度空洞空间金字塔池化 DASPP 结构。由于肺栓塞医学图像中病灶位置变化不定，则需要在一定程度上增大感受野。DASPP 可以保证不更改原输入的大小的前提下，增大感受野，这帮助网络得到了比之前多的空间维度上的信号，并且没有给网络模型造成额外的负担。

***
> + [基于深度学习的CTPA肺栓塞识别及评价](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1022647943.nh&uniplatform=NZKPT&v=Ss4rpoRr4QxBX1NUNNLn69CB1F4fJJn25xVRA67_cU5X7lNfVu7L93HmKl8TcJ5Q)
>   - 作者：邵亚君
>   - 联系方式：
>   - 数据集：
>   - 代码：无
>   - 总结：
***

## 文献调研
### 综述论文

> + [Survey on deep learning for pulmonary medical imaging](https://link.springer.com/article/10.1007/s11684-019-0726-4)
>   - 年份期刊：2020/Frontiers of medicine
>   - 引用次数：48
>   - 引用：Ma J, Song Y, Tian X, et al. Survey on deep learning for pulmonary medical imaging[J]. Frontiers of medicine, 2020, 14(4): 450-469.
>   - 总结：
***


### 待读论文列表
| 名称  | 作者   | 年份 | 引用次数 | 期刊 | 源码 | 数据集 |
| :--- | :---  | :------ | :------ |:------ | :------ |:------ |
| [Attention Based CNN-LSTM Network for Pulmonary Embolism Prediction on Chest Computed Tomography Pulmonary Angiograms](https://link.springer.com/chapter/10.1007/978-3-030-87234-2_34) | Sudhir Suman, Gagandeep Singh, Nicole Sakla |  2021|  2| MICCAI |
| AANet: Artery-Aware Network for Pulmonary Embolism Detection in CTPA Images | Jia Guo, Xinglong Liu, Yinan Chen, Shaoting Zhang |  2022|   | MICCAI |
| [RadFusion: Benchmarking Performance and Fairness for Multimodal Pulmonary Embolism Detection from CT and EHR](https://arxiv.org/abs/2111.11665) | Yuyin Zhou, Shih-Cheng Huang, Jason Alan Fries, Alaa Youssef |  2021|  | arXiv |   | from Stanford University Medical Center (SUMC) |
| [Multimodal fusion with deep neural networks for leveraging CT imaging and electronic health record: a case-study in pulmonary embolism detection](https://www.nature.com/articles/s41598-020-78888-w) | Shih‑Cheng Huang, Anuj Pareek |  2020|  28| Scientific reports |


### 已读论文总结

> + [Deep Learning for Pulmonary Embolism Detection: Tackling the RSNA 2020 AI Challenge.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8489447/)
>   - 引用：Pan I. Deep Learning for Pulmonary Embolism Detection: Tackling the RSNA 2020 AI Challenge[J]. Radiology: Artificial Intelligence, 2021, 3(5).
>   - 总结：
***
> + [Use of machine learning to develop and evaluate models using preoperative and intraoperative data to identify risks of postoperative complications ](https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2777894)
>   - 引用：Xue B, Li D, Lu C, et al. Use of machine learning to develop and evaluate models using preoperative and intraoperative data to identify risks of postoperative complications[J]. JAMA network open, 2021, 4(3): e212240-e212240.
>   - 总结：
***
> + [Automated Deep Learning Analysis for Quality Improvement of CT Pulmonary Angiography](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8980873/)
>   - 引用：Hahn L D, Hall K, Alebdi T, et al. Automated deep learning analysis for quality improvement of CT pulmonary angiography[J]. Radiology: Artificial Intelligence, 2022, 4(2).
>   - 总结：
***
> + [PENet—a scalable deep-learning model for automated diagnosis of pulmonary embolism using volumetric CT imaging](https://www.nature.com/articles/s41746-020-0266-y)
>   - 引用：Huang S C, Kothari T, Banerjee I, et al. PENet—a scalable deep-learning model for automated diagnosis of pulmonary embolism using volumetric CT imaging[J]. NPJ digital medicine, 2020, 3(1): 1-9.
***
> + [Attention Based CNN-LSTM Network for Pulmonary Embolism Prediction on Chest Computed Tomography Pulmonary Angiograms](https://link.springer.com/chapter/10.1007/978-3-030-87234-2_34)
>   - 引用：Suman S, Singh G, Sakla N, et al. Attention Based CNN-LSTM Network for Pulmonary Embolism Prediction on Chest Computed Tomography Pulmonary Angiograms[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2021: 356-366.
***
> + [Development and Performance of the Pulmonary Embolism Result Forecast Model (PERFORM) for Computed Tomography Clinical Decision Support]()
>   - 引用：Banerjee I, Sofela M, Yang J, et al. Development and performance of the pulmonary embolism result forecast model (PERFORM) for computed tomography clinical decision support[J]. JAMA network open, 2019, 2(8): e198719-e198719.
***
> + [Automated detection of pulmonary embolism from CT-angiograms using deep learning]()
>   - 引用：Huhtanen H, Nyman M, Mohsen T, et al. Automated detection of pulmonary embolism from CT-angiograms using deep learning[J]. BMC Medical Imaging, 2022, 22(1): 1-10.
***
> + [Automated detection of pulmonary embolism in CT pulmonary angiograms using an AI-powered algorithm]()
>   - 引用：Weikert T, Winkel D J, Bremerich J, et al. Automated detection of pulmonary embolism in CT pulmonary angiograms using an AI-powered algorithm[J]. European Radiology, 2020, 30(12): 6545-6553.
***
> + [RadFusion: Benchmarking Performance and Fairness for Multimodal Pulmonary Embolism Detection from CT and EHR]()
>   - 引用：Zhou Y, Huang S C, Fries J A, et al. Radfusion: Benchmarking performance and fairness for multimodal pulmonary embolism detection from ct and ehr[J]. arXiv preprint arXiv:2111.11665, 2021.
***
> + []()
>   - 引用：
***
> + []()
>   - 引用：
***





