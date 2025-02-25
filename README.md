
# Pulmonary-Embolism-Detection

### 肺栓塞检测难点
1. CT扫描由数百张图像组成，需要详细检查以识别肺动脉内的凝块；
2. 扫描结果有时会受到环境的影响，如肺部运动伪影与栓子在影像中很难区分等；
3. 假阳性去除需要考虑多方面的原因；
***

### 文档整理
链接: https://pan.baidu.com/s/1o2CExesyUyuXTzCBW0W3fg
提取码: m8jk 
***

### 2020 RSNA Pulmonary Embolism Detection Challenge 相关信息
1. [竞赛Overview](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/overview)
2. [竞赛成果汇总](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-pe-detection-challenge-2020)
3. [1st place solution with code](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/194145)
4. [3D CNN for rv_lv_ratio](https://www.kaggle.com/code/pedromatos14/3d-cnn-for-rv-lv-ratio-lt-1/notebook)
***

###  肺栓塞工程化产品 
1. [UCD School of Medicine](https://www.aidoc.com/blog/clinical_study/automated-detection-of-pulmonary-embolism-in-ct-pulmonary-angiograms-using-an-ai-powered-algorithm/)
   [/1](https://www.aidoc.com/blog/clinical_study/assessment-of-artificial-intelligence-technology-for-pulmonary-embolism-detection/)
   [/2](https://www.aidoc.com/blog/clinical_study/ai-powered-detection-of-pulmonary-embolism-in-ct-pulmonary-angiograms-a-validation-study-of-the-diagnostic-performance-of-prototype-algorithms/)
   [/3](https://www.aidoc.com/solutions/pe-care-coordination/)
2. [Mount Sinai pilot study](https://www.newswise.com/articles/could-ekgs-help-doctors-use-ai-to-detect-pulmonary-embolisms)
3. [TechTarget xtelligent HEALTHCARE MEDIA](https://healthitanalytics.com/news/artificial-intelligence-detectspulmonary-embolisms)
4. [Duke Institute](https://dihi.org/project/machine-learning-for-early-identification-and-management-of-pulmonary-embolism/)
5. [Stanford University, Stanford](https://grantome.com/grant/NIH/R01-LM012966-01)
6. [DIHI](https://dihi.org/project/machine-learning-for-early-identification-and-management-of-pulmonary-embolism/)
7. [Newswise](https://www.newswise.com/articles/could-ekgs-help-doctors-use-ai-to-detect-pulmonary-embolisms)
***

### 相关实验室
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


## 经典论文

#### 1. 深度学习模型
| 名称  | 任务   | 年份 | 引用次数 | 期刊/会议  | 源码 | 数据集 |
| :--- | :---  | :------ | :------ |:------ | :------ |:------ |
| [PECon: Contrastive Pretraining to Enhance Feature Alignment Between CT and EHR Data for Improved Pulmonary Embolism Diagnosis](https://arxiv.org/pdf/2308.14050.pdf) | 分类 |  2023|   | International Workshop on Machine Learning in Medical Imaging | https://github.com/BioMedIA-MBZUAI/PECon | [3] RadFusion: Multimodal |
| [AANet: Artery-Aware Network for Pulmonary Embolism Detection in CTPA Images](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_45) | 分割 |  2022|   7| MICCAI | https://github.com/guojiajeremy/AANet | LUNA16, [4] CAD-PE, [5]FUMPE |
| [A multitask deep learning approach for pulmonary embolism detection and identification](https://www.nature.com/articles/s41598-022-16976-9) | 分类 |  2022|   11| Scientific Reports |  | [1] The RSNA Pulmonary Embolism CT Dataset |
| [Detecting Pulmonary Embolism using Deep Neural Networks](http://www.ijpe-online.com/EN/Y2021/V17/I3/322) | 分类 |  2021| 11 | International Journal of Performability Engineering |  |  |
| [PENet—a scalable deep-learning model for automated diagnosis of pulmonary embolism using volumetric CT imaging](https://www.nature.com/articles/s41746-020-0266-y) | 分类 |  2020 | 86  | NPJ digital medicine | https://github.com/marshuang80/penet | [3] RadFusion: Multimodal Pulmonary Embolism Dataset |
| [A two-stage convolutional neural network for pulmonary embolism detection from CTPA images](https://ieeexplore.ieee.org/abstract/document/8746218) | 分类 |  2018|   37| IEEE Access |  | [4] CAD-PE |
| [Artificial intelligence models may predict pulmonary embolism risk](https://www.healio.com/news/pulmonology/20190812/artificial-intelligence-models-may-predict-pulmonary-embolism-risk) | 分类 | 2019 |  |  |  |  |


#### 2. CNN-LSTM模型
| 名称  |  任务  | 年份 | 引用次数 | 期刊/会议  | 源码 | 数据集 |
| :--- | :---  | :------ | :------ |:------ | :------ |:------ |
| [Automated detection of pulmonary embolism from CT-angiograms using deep learning](https://link.springer.com/article/10.1186/s12880-022-00763-z) | 分割 |  2022|   19| BMC Medical Imaging | https://github.com/turku-rad-ai/pe-detection |  |
| [Attention Based CNN-LSTM Network for Pulmonary Embolism Prediction on Chest Computed Tomography Pulmonary Angiograms](https://link.springer.com/chapter/10.1007/978-3-030-87234-2_34) | 分类 |  2021|  9| MICCAI |


#### 3. 多模态融合模型
| 名称  | 任务  | 年份 | 引用次数 | 期刊/会议 | 源码 | 数据集 |
| :--- | :---  | :------ | :------ |:------ | :------ |:------ |
| [Multimodal Diagnosis for Pulmonary Embolism from EHR Data and CT Images](https://ieeexplore.ieee.org/abstract/document/9871041) | 分类 |  2022|   1| EMBC |  |  |
| [RadFusion: Benchmarking Performance and Fairness for Multimodal Pulmonary Embolism Detection from CT and EHR](https://arxiv.org/abs/2111.11665) | 分类 |  2021|  17| arXiv |  https://github.com/marshuang80/pe_fusion | [3] RadFusion: Multimodal Pulmonary Embolism Dataset |
| [Dual-energy CT for pulmonary embolism: current and evolving clinical applications](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8390816/) | 分类 |  2021|   18 | Korean Journal of Radiology |  |  |
| [Multimodal fusion with deep neural networks for leveraging CT imaging and electronic health record: a case-study in pulmonary embolism detection](https://www.nature.com/articles/s41598-020-78888-w) | 分类 |  2020|  102| Scientific reports | https://github.com/marshuang80/pe_fusion | [3] RadFusion: Multimodal Pulmonary Embolism Dataset |
| [Development and Performance of the Pulmonary Embolism Result Forecast Model (PERFORM) for Computed Tomography Clinical Decision Support](https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2747483) | 分类 |  2019|   50 | JAMA |   | SHC, Duke |


#### 4. Multi-view模型
| 名称  | 任务  | 年份 | 引用次数 | 期刊/会议 | 源码 | 数据集 |
| :--- | :---  | :------ | :------ |:------ | :------ |:------ |
| [Multi-View Coupled Self-Attention Network for Pulmonary Nodules Classification](https://openaccess.thecvf.com/content/ACCV2022/html/Zhu_Multi-View_Coupled_Self-Attention_Network_for_Pulmonary_Nodules_Classification_ACCV_2022_paper.html) | 分类  | 2022 | 4 |ACCV | https://github.com/ahukui/MVCs | |
| [A Novel Framework for Accurate and Non-Invasive Pulmonary Nodule Diagnosis by Integrating Texture and Contour Descriptors](https://ieeexplore.ieee.org/abstract/document/9433830) | 分类  | 2021 | 2 | ISBI |  | |
| [Richer fusion network for breast cancer classification based on multimodal data](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01340-6) |  分类 | 2021 | 30 | BMC Medical Informatics and Decision Making |  |  |


#### 5. Cross-modal模型
| 名称  | 任务  | 年份 | 引用次数 | 期刊/会议 | 源码 | 数据集 |
| :--- | :---  | :------ | :------ |:------ | :------ |:------ |
| [CMAFGAN: A Cross-Modal Attention Fusion based Generative Adversarial Network for attribute word-to-face synthesis](https://www.sciencedirect.com/science/article/pii/S0950705122008863) |   | 2022 | 4 | Knowledge-Based Systems |  |  |


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
>   - 数据集： Pulmonary  Embolism(肺栓塞数据集)
>   - 总结：提出一种混合域注意力机制的肺栓塞分割方法。混合域注意力机制是指分别在空间域和通道域上添加注意力机制，这种机制使模型在学习的过程中更关注在空间和通道中的重要信息，忽略和抑制无关信息，本文对这种注意力机制的实现的方式是利用 scSE 模块对输入特征不断地进行权重的调整。
***
> + [基于深度学习的CTPA肺栓塞图像分割方法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202101&filename=1020152016.nh&uniplatform=NZKPT&v=NZmWLuuvngR1unltaImG-Y8T4yvb7KX4GPUUXSR4qeKgRL6zB-seBBPkkiIbh3MX)
>   - 作者：温洲
>   - 联系方式：wenzhouwww@126.com
>   - 数据集：中日友好医院提供的CTPA影像数据
>   - 总结：深入分析了U-Net分割精度优于FCN的内在本质，结合ResNet和DenseNet的设计思想和基本原理，提出了一种改进U-Net结构，改进的U-Net以残差模块作为基本结构，构建了强大的特征提取网络。并引入中间特征融合模块Concat Block，保留了更多的收缩路径的中间特征，将融合后的特征以通道拼接的方式融合到扩张路径，使得网络获取了更丰富的上下文信息，进一步提升网络的分割精度。结合肺栓塞危险度评价实际场景中的需求，实现了基于Mask RCNN的肺栓塞图像分割方法，该方法能够在定位肺栓塞的位置的同时给出肺栓塞分割结果。在该方法的基础上，使用Group卷积对Mask RCNN进行改进，改进后的方法有效的减少了网络的参数量，优化了网络的推理速度，能够降低30％以上的显存和时长占用，且最终精度近乎不变。
***
> + [基于深度学习的肺部CT图像多阶段栓塞检测方法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202201&filename=1020349904.nh&uniplatform=NZKPT&v=_DZKMpDu2eH53cHBrEKMPWJaKLM4lgzsDfjd00gV17DE60VMfpGsp6Q6c5ojAVRk)
>   - 作者：苏建超
>   - 联系方式：
>   - 数据集：肺栓塞挑战赛数据集、肺栓塞 129 数据集
>   - 总结：本文提出的肺栓塞检测方法包含三个阶段：基于 3D 卷积神经网络（3D CNN)的候选点提取、基于 2D 卷积神经网络(2D CNN)的假阳筛除和基手动静脉分离的假阳筛除。（1）3D CNN 与2DCNN 相结合，利用了3D CNN 提取三维空间信息的能力与 2D CNN 减少分类网络参数量的优势：（2）为每个候选点实现了一个新颖的图像表达，有效将3D数据降维到2D，且最大程度保留了所在血管的图像信息：（3）设计了子树分离算法提取肺血管子树，使神经网络的预测结果在子树内进行校正，获得一致的动静脉分离结果。
***
> + [基于弱标记CT影像的新冠肺炎和肺动脉栓塞识别研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1021914801.nh&uniplatform=NZKPT&v=gOSgFeEqM2G7Am5XgxSCg6uAEMJ03Ufwkx27bKe-wX6VOWDkUX4vQhIaRVm1dPnE)
>   - 作者：刘艺璇
>   - 联系方式：
>   - 数据集： 541个样本数据均来自就诊于本学校附属医院影像科的疑似患者。
>   - 总结：本文研究了弱监督下新冠病灶区定位和轻量化网络设计的问题，通过改进3D 残差网络实现了新冠肺炎准确快速地识别，并利用分类网络中区分特征实现病灶区的定位和可视化。首先，为了减少其他组织的干扰，利用无监督的连通域分析法对肺部进行分割，并将其较好的分割结果作为真实分割注释对分割网络进行训练；然后，设计了轻量级主干网络和渐进式分类器，在保证计算成本低的同时提高新冠肺炎的分类准确性；最后，为了对分类结果进行可解释性说明，并为影像科医生提供更加直观的诊断依据，设计了一种类激活图和三维连通区域相结合的弱监督新冠肺炎病灶定位算法。
***
> + [基于改进GAC模型及深度学习的肺动脉与栓塞分割方法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202201&filename=1022004522.nh&uniplatform=NZKPT&v=O1q1Qdnf7py8WTOKTw3Bf9YxVSjnsCJkHFY2VXWrPvMdGwa1dkoR8_KttFy768cq)
>   - 作者：刘珍宏
>   - 联系方式：Corresponding author: Hongfang Yuan (yuanhf@mail.buct.edu.cn) 
>   - 数据集：中日友好医院提供的CTPA影像数据
>   - 总结：本文提出的用于肺栓塞分割的ResD-Unet架构，它结合了残差连接、密集连接和 U-Net的优点。以U-net网络为基础，将U-net中的普通卷积块替换成残差密集块，残差密集连接将之前所有层的特征都添加到底层，实现特征重用，便于训练过程中梯度的反向传播。有效地解决了信息丢失的问题，避免了梯度消失。批处理归一化层的使用避免了过拟合，加快了训练速度。可以在构建更深的网络的同时改善网络的梯度流通。为克服DiceLoss训练不稳定的问题，结合交叉熵损失和结构相似度损失的设计思想和相关理论，本文提出了一种新的混合损失函数，充分利用三类损失函数的优点，通过混合损失函数提高目标边界的分割准确度。
***
> + [基于多视图加权注意力机制的CTPA肺动脉栓塞图像分割算法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1021893263.nh&uniplatform=NZKPT&v=Ws171w8yb6M0ZC2Vo8F-L_9Mqs0MsMUlig-rWRht3PG4hUk5sXz59Ejor0jQJQOa)
>   - 作者：鲁启洋
>   - 联系方式：Correspondence: Hui Liu: liuhuijiujiu@gmail.com // Xinzhou Xie: xinzhxie@hotmail.com 
>   - 数据集：广东省人民医院提供的CTPA影像数据、伊朗马什哈德Ferdowsi大学机器视觉实验室的开源数据集FUMPE(standing for Ferdowsi University of Mashhad's PE dataset)
>   - 总结：针对二维分割网络对疑似肺动脉栓塞区域易出现误判，三维分割网络消耗计算资源巨大的问题，本文提出了一种基于多视图加权注意力机制的网络结构 （MWA U-Net）。该模型通过三个并联的特征提取网络，分别在三个视图上进行特征提取，通过引入注意力机制，利用自适应权重的主视图间协作以模仿临床的观察诊断，可以有效的提高肺动脉栓塞病灶的分割精度。
***
> + [基于深度学习的肺栓塞医学图像分割算法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1022086347.nh&uniplatform=NZKPT&v=g13KhaNPo0rvc-07zIUzrFKSNoXQKZKk21akKNa_KEulachyXfkWnXBJ6Ocn0aeO)
>   - 作者：刘硕
>   - 数据集：[5] FUMPE
>   - 总结：引用 DoubleU-Net 的基础框架，在编码部分，将第一个编码器的原始结构替换为经过预训练的 VGG-19_bn 结构，VGG-19 将允许接收更深度信息，同时保证感受野的大小。第二个编码器在传统编码结构的基础上加入了 squeeze-and-excite 模块，这使得网络在编码过程中可以更专注于重要的特征区域。 在解码部分，在跳跃连接上由于网络层数的不同进行了修改。对于第一个 U-Net 结构中的解码器，只连接来自其本身编码器的跳跃连接，并且在其中引入了非对称特征融合模块 AFF，使不同尺度的特征信息能够得到更好的利用。然而在第二个 U-Net 结构中的解码器，则同时连接来自两个编码器的跳跃连接。在每个下采样阶段和上采样阶段之间，加入了深度空洞空间金字塔池化 DASPP 结构。DASPP 可以保证不更改原输入的大小的前提下，增大感受野，这帮助网络得到了比之前多的空间维度上的信号，并且没有给网络模型造成额外的负担。
***


## 文献调研
### 综述论文

> + [Survey on deep learning for pulmonary medical imaging](https://link.springer.com/article/10.1007/s11684-019-0726-4)
>   - 年份/期刊：2020/Frontiers of medicine
>   - 引用次数：48
>   - 引用：Ma J, Song Y, Tian X, et al. Survey on deep learning for pulmonary medical imaging[J]. Frontiers of medicine, 2020, 14(4): 450-469.
***
> + [Computer Aided Detection for Pulmonary Embolism Challenge (CAD-PE)](https://arxiv.org/abs/2003.13440)
>   - 年份/期刊：2020/arXiv
>   - 引用次数：13
>   - 引用：González G, Jimenez-Carretero D, Rodríguez-López S, et al. Computer aided detection for pulmonary embolism challenge (cad-pe)[J]. arXiv preprint arXiv:2003.13440, 2020.
***
> + [Seeking an Optimal Approach for Computer-Aided Pulmonary Embolism Detection](https://link.springer.com/chapter/10.1007/978-3-030-87589-3_71)
>   - 年份/期刊：2021/International Workshop on Machine Learning in Medical Imaging
>   - 引用次数：11
>   - 引用：Islam N U, Gehlot S, Zhou Z, et al. Seeking an Optimal Approach for Computer-Aided Pulmonary Embolism Detection[C]//International Workshop on Machine Learning in Medical Imaging. Springer, Cham, 2021: 692-702.
***


### 已读论文列表

> + [Deep Learning for Pulmonary Embolism Detection: Tackling the RSNA 2020 AI Challenge.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8489447/)
>   - 引用：Pan I. Deep Learning for Pulmonary Embolism Detection: Tackling the RSNA 2020 AI Challenge[J]. Radiology: Artificial Intelligence, 2021, 3(5).
***
> + [Use of machine learning to develop and evaluate models using preoperative and intraoperative data to identify risks of postoperative complications ](https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2777894)
>   - 引用：Xue B, Li D, Lu C, et al. Use of machine learning to develop and evaluate models using preoperative and intraoperative data to identify risks of postoperative complications[J]. JAMA network open, 2021, 4(3): e212240-e212240.
***
> + [Automated Deep Learning Analysis for Quality Improvement of CT Pulmonary Angiography](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8980873/)
>   - 引用：Hahn L D, Hall K, Alebdi T, et al. Automated deep learning analysis for quality improvement of CT pulmonary angiography[J]. Radiology: Artificial Intelligence, 2022, 4(2).
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


### 已接收的论文

> + [PE-MVCNet:  Multi-view and Cross-modal Fusion Network for Pulmonary Embolism Prediction](https://arxiv.org/abs/2402.17187)



