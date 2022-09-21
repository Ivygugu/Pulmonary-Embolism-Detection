
# Pulmonary-Embolism-Detection

## 实验室
1. 国防科技大学电子科学学院 ATR 重点实验室
2. [Center for Artificial Intelligence in Medicine & Imaging](https://aimi.stanford.edu/ )
***

## 经典模型
> + [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)
>   - 引用：Ren S, He K, Girshick R, et al. Faster r-cnn: Towards real-time object detection with region proposal networks[J]. Advances in neural information processing systems, 2015, 28.

***
> + [Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)
>   - 引用：He K, Gkioxari G, Dollár P, et al. Mask r-cnn[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2961-2969.

***
> + [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640 )
>   - 引用：Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.
>   - [代码](https://github.com/hizhangp/yolo_tensorflow.git )
***
> + [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242 )
>   - 引用：Redmon J, Farhadi A. YOLO9000: better, faster, stronger[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7263-7271.
>   - [代码](https://github.com/allanzelener/YAD2K  )
***
> + [Yolov3: An incremental improvement](https://arxiv.53yu.com/pdf/1804.02767.pdf)
>   - 引用：Redmon J, Farhadi A. Yolov3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.
>   - [代码](https://github.com/eriklindernoren/PyTorch-YOLOv3)
***
> + [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)
>   - 引用：Bochkovskiy A, Wang C Y, Liao H Y M. Yolov4: Optimal speed and accuracy of object detection[J]. arXiv preprint arXiv:2004.10934, 2020.
>   - [代码](https://github.com/bubbliiiing/yolov4-pytorch)
***
> + [YOLOv5](https://github.com/xuanzhangyang/yolov5)
>   - 无论文，有[代码](https://github.com/xuanzhangyang/yolov5)
***
> + [YOLOv6](https://github.com/meituan/YOLOv6)
>   - 无论文，有[代码](https://github.com/meituan/YOLOv6)
***
> + [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
>   - 引用：Wang C Y, Bochkovskiy A, Liao H Y M. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors[J]. arXiv preprint arXiv:2207.02696, 2022.
>   - [代码](https://github.com/WongKinYiu/yolov7)
***
> + [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)
>   - 引用：Ge Z, Liu S, Wang F, et al. Yolox: Exceeding yolo series in 2021[J]. arXiv preprint arXiv:2107.08430, 2021.
>   - [代码](https://github.com/Megvii-BaseDetection/YOLOX)
***


## 硕士论文
### 国内硕士论文
> + [基于混合域注意力机制的肺栓塞图像分割研究](、)
>   - 作者：徐欢
>   - 联系方式：
>   - 数据集： Pulmonary  Embolism(肺栓塞数据集)
>   - 代码：无
>   - 总结：基于开源的肺栓塞数据集，提出了一种混合域注意力机制的肺栓塞分割方法。混合域注意力机制是指分别在空间域和通道域上添加注意力机制，这种机制使模型在学习的过程中更关注在空间和通道中的重要信息，忽略和抑制无关信息，本文对这种注意力机制的实现的方式是利用 scSE 模块对输入特征不断地进行权重的调整。为了验证 scSE 的在本实验中的有效性，选用 U-Net 为基础网络，加入scSE对 U-Net 改进，实验结果表明改进后的网络在肺栓塞数据集上取得了 0.837 的Dice 值，相比于原网络的结果提高了 3.4%，证明了混合域注意力机制能够有效提升基础网络对肺栓塞分割的准确率。
***
> + [基于深度学习的CTPA肺栓塞图像分割方法研究](、)
>   - 作者：温洲
>   - 联系方式：wenzhouwww@126.com
>   - 数据集：中日友好医院提供的CTPA影像数据
>   - 代码：无
>   - 总结：深入分析了U-Net分割精度优于FCN的内在本质，结合ResNet和DenseNet的设计思想和基本原理，提出了一种改进U-Net结构，改进的U-Net以残差模块作为基本结构，构建了强大的特征提取网络。并引入中间特征融合模块Concat Block，保留了更多的收缩路径的中间特征，将融合后的特征以通道拼接的方式融合到扩张路径，使得网络获取了更丰富的上下文信息，进一步提升网络的分割精度。结合肺栓塞危险度评价实际场景中的需求，实现了基于Mask RCNN的肺栓塞图像分割方法，该方法能够在定位肺栓塞的位置的同时给出肺栓塞分割结果。在该方法的基础上，使用Group卷积对Mask RCNN进行改进，改进后的方法有效的减少了网络的参数量，优化了网络的推理速度，能够降低30％以上的显存和时长占用，且最终精度近乎不变。
***
> + [基于深度学习的肺部CT图像多阶段栓塞检测方法研究](、)
>   - 作者：苏建超
>   - 联系方式：
>   - 数据集：肺栓塞挑战赛数据集、肺栓塞 129 数据集
>   - 代码：无
>   - 总结：本文提出的肺栓塞检测方法包含三个阶段：基于 3D 卷积神经网络（3D CNN)的候选点提取、基于 2D 卷积神经网络(2D CNN)的假阳筛除和基手动静脉分离的假阳筛除。（1）3D CNN 与2DCNN 相结合，利用了3D CNN 提取三维空间信息的能力与 2D CNN 减少分类网络参数量的优势：（2）为每个候选点实现了一个新颖的图像表达，有效将3D数据降维到2D，且最大程度保留了所在血管的图像信息：（3）设计了子树分离算法提取肺血管子树，使神经网络的预测结果在子树内进行校正，获得一致的动静脉分离结果。
***
> + [基于弱标记CT影像的新冠肺炎和肺动脉栓塞识别研究](、)
>   - 作者：刘艺璇
>   - 联系方式：
>   - 数据集： 541个样本数据均来自就诊于本学校附属医院影像科的疑似患者。
>   - 代码：无
>   - 总结：本文研究了弱监督下新冠病灶区定位和轻量化网络设计的问题，通过改进3D 残差网络实现了新冠肺炎准确快速地识别，并利用分类网络中区分特征实现病灶区的定位和可视化。首先，为了减少其他组织的干扰，利用无监督的连通域分析法对肺部进行分割，并将其较好的分割结果作为真实分割注释对分割网络进行训练；然后，设计了轻量级主干网络和渐进式分类器，在保证计算成本低的同时提高新冠肺炎的分类准确性；最后，为了对分类结果进行可解释性说明，并为影像科医生提供更加直观的诊断依据，设计了一种类激活图和三维连通区域相结合的弱监督新冠肺炎病灶定位算法。
***
> + [基于改进GAC模型及深度学习的肺动脉与栓塞分割方法研究](、)
>   - 作者：刘珍宏
>   - 联系方式：Corresponding author: Hongfang Yuan (yuanhf@mail.buct.edu.cn) 
>   - 数据集：中日友好医院提供的CTPA影像数据
>   - 代码：无
>   - 总结：本文提出的用于肺栓塞分割的ResD-Unet架构，它结合了残差连接、密集连接和 U-Net的优点。以U-net网络为基础，将U-net中的普通卷积块替换成残差密集块，残差密集连接将之前所有层的特征都添加到底层，实现特征重用，便于训练过程中梯度的反向传播。有效地解决了信息丢失的问题，避免了梯度消失。批处理归一化层的使用避免了过拟合，加快了训练速度。可以在构建更深的网络的同时改善网络的梯度流通。为克服DiceLoss训练不稳定的问题，结合交叉熵损失和结构相似度损失的设计思想和相关理论，本文提出了一种新的混合损失函数，充分利用三类损失函数的优点，通过混合损失函数提高目标边界的分割准确度。
***
> + [基于多视图加权注意力机制的CTPA肺动脉栓塞图像分割算法研究](、)
>   - 作者：鲁启洋
>   - 联系方式：Correspondence: Hui Liu: liuhuijiujiu@gmail.com // Xinzhou Xie: xinzhxie@hotmail.com 
>   - 数据集：广东省人民医院提供的CTPA影像数据、伊朗马什哈德Ferdowsi大学机器视觉实验室的开源数据集FUMPE(standing for Ferdowsi University of Mashhad's PE dataset)
>   - 代码：无
>   - 总结：针对二维分割网络对疑似肺动脉栓塞区域易出现误判，三维分割网络消耗计算资源巨大的问题，本文提出了一种基于多视图加权注意力机制的网络结构 （MWA U-Net）。该模型通过三个并联的特征提取网络，分别在三个视图上进行特征提取，通过引入注意力机制，利用自适应权重的主视图间协作以模仿临床的观察诊断，可以有效的提高肺动脉栓塞病灶的分割精度。
***



## 文献调研
### 综述论文

#### 红外目标检测

> + [红外弱小目标检测算法综述](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2020&filename=ZZDZ202002001&uniplatform=NZKPT&v=PTtAkTKP5eakp3HJd1vkEK5niqiNzoWMB8wKL-rdCQ3cvekJJ3sA9fTwP9DlOcwP)
>   - 年份期刊：2020/中国图象图形学报
>   - 引用次数：14
>   - 引用：[1]李俊宏,张萍,王晓玮,黄世泽.红外弱小目标检测算法综述[J].中国图象图形学报,2020,25(09):1739-1753.
>   - 总结：本文主要论述国内外红外弱小目标检测的研究成果和现状，分析典型的基于单帧和基于序列的红外弱小目标检测方法，分析各种算法的利弊，比较算法的性能，提出问题，最后对发展进行展望。（详见思维导图）
***
> + [红外单帧图像弱小目标检测技术综述](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2019&filename=JGDJ201908001&uniplatform=NZKPT&v=J88b_sjCpUn7yOC0QIvdQ6R-2cKd24pTJlVQ0E5e4Eaf524JycUdSLuRv4BPBSJB)
>   - 年份期刊：2018/激光与光电子学进展
>   - 引用次数：49
>   引用：[1]王好贤,董衡,周志权.红外单帧图像弱小目标检测技术综述[J].激光与光电子学进展,2019,56(08):9-22.
***

#### 目标检测
> + [ 基于深度卷积神经网络的目标检测研究综述]( https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2020&filename=GXJM202005019&uniplatform=NZKPT&v=XWfRRRl2GbXDw8C4NakqMkkocpNF_RyFM4q-wwk2UVODVh8JhoabrY1phyRgU3RH)
>   - 年份期刊：2020/光学精密工程
>   - 引用次数：111
>   引用：[1]范丽丽,赵宏伟,赵浩宇,胡黄水,王振.基于深度卷积神经网络的目标检测研究综述[J].光学精密工程,2020,28(05):1152-1164.
***
> + [ 基于深度学习的目标检测研究综述]( https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDAUTO&filename=XDXK202211020&uniplatform=NZKPT&v=guNZIliFumhqyhehOoq-pOR1AuRLdQ5NNCdQXPEt-RGytvdVQbKzP3felP6uESN8)
>   - 年份期刊：2022/现代信息科技
>   - 引用次数：/
>   引用：[1]谷永立,宗欣欣.基于深度学习的目标检测研究综述[J].现代信息科技,2022,6(11):76-81.DOI:10.19850/j.cnki.2096-4706.2022.011.020.
***
> + [基于双阶段目标检测算法研究综述 ]( )
>   - 年份期刊：2021/中国计算机用户协会网络应用分会2021年第二十五届网络新技术与应用年会论文集
>   - 引用次数：/
>   引用：[1]贺宇哲,何宁,张人,晏康,于海港. 基于双阶段目标检测算法研究综述[C]//.中国计算机用户协会网络应用分会2021年第二十五届网络新技术与应用年会论文集.,2021:182-186.DOI:10.26914/c.cnkihy.2021.047821.
***
> + [ 基于遥感图像的船舶目标检测方法综述](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2020&filename=DATE202009021&uniplatform=NZKPT&v=l61Q0_VBN9hQtm8V6PQAPFI7Wc92VQcYEruLe3vN1iQnRryLGAlZV-C2zI0pvQ5k )
>   - 年份期刊：2020/电讯技术
>   - 引用次数：14
>   引用：[1]王伟.基于遥感图像的船舶目标检测方法综述[J].电讯技术,2020,60(09):1126-1132.
***
> + [ A Survey of Deep Learning-based Object Detection]( )
>   - 年份期刊：2019/IEEE access
>   - 引用次数：681
>   引用：Jiao L, Zhang F, Liu F, et al. A survey of deep learning-based object detection[J]. IEEE access, 2019, 7: 128837-128868.
***
> + [A Survey of Modern Deep Learning based Object Detection Models ](https://www.sciencedirect.com/science/article/pii/S1051200422001312 )
>   - 年份期刊：2022/Digital Signal Processing
>   - 引用次数：88
>   引用：Zaidi S S A, Ansari M S, Aslam A, et al. A survey of modern deep learning based object detection models[J]. Digital Signal Processing, 2022: 103514.
***
> + [ A Survey of Modern Object Detection Literature using Deep Learning]( https://arxiv.org/abs/1808.07256)
>   - 年份期刊：2018
>   - 引用次数：30
>   引用：Chahal K S, Dey K. A survey of modern object detection literature using deep learning[J]. arXiv preprint arXiv:1808.07256, 2018.
***
> + [A Survey on Deep Domain Adaptation and Tiny Object Detection Challenges, Techniques and Datasets ]( https://arxiv.org/abs/2107.07927)
>   - 年份期刊：2021
>   - 引用次数：1
>   引用：Muzammul M, Li X. A Survey on Deep Domain Adaptation and Tiny Object Detection Challenges, Techniques and Datasets[J]. arXiv preprint arXiv:2107.07927, 2021.
***
> + [ A Survey on Object Detection in Optical Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0924271616300144 )
>   - 年份期刊：2016
>   - 引用次数：1011
>   引用：Cheng G, Han J. A survey on object detection in optical remote sensing images[J]. ISPRS Journal of Photogrammetry and Remote sensing, 2016, 117: 11-28.
***
> + [ Salient Object Detection in the Deep Learning Era: An In-depth Survey]( https://ieeexplore.ieee.org/abstract/document/9320524)
>   - 年份期刊：2021/IEEE Transactions on Pattern Analysis and Machine Intelligence
>   - 引用次数：377
>   引用：Wang W, Lai Q, Fu H, et al. Salient object detection in the deep learning era: An in-depth survey[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 44(6): 3239-3259.
***
> + [ Deep Learning for Generic Object Detection: A Survey]( https://link.springer.com/article/10.1007/s11263-019-01247-4)
>   - 年份期刊：2020
>   - 引用次数：1700
>   引用：Liu L, Ouyang W, Wang X, et al. Deep learning for generic object detection: A survey[J]. International journal of computer vision, 2020, 128(2): 261-318.
***
> + [Deep Learning for UAV-based Object Detection and Tracking: A Survey ]( https://ieeexplore.ieee.org/abstract/document/9604009)
>   - 年份期刊：2021/
>   - 引用次数：21
>   引用：Wu X, Li W, Hong D, et al. Deep learning for unmanned aerial vehicle-based object detection and tracking: a survey[J]. IEEE Geoscience and Remote Sensing Magazine, 2021, 10(1): 91-124.
***

#### 红外图像处理
> + [ 非制冷红外图像降噪算法综述]( https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=HWJS202106008&uniplatform=NZKPT&v=yJDsWGcr4R8iYOQc8oZrT8xV56pW5K40zmJtnYUZvq7a2MU9-g4qDCyHcnFFT-ZJ)
>   - 年份期刊：2021/红外技术
>   - 引用次数：2
>   引用：王加, 周永康, 李泽民, 等. 非制冷红外图像降噪算法综述[J]. 红外技术, 2021, 43(6): 557-565.
***
> + [红外图像边缘检测算法综述]( https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=HWJS202103001&uniplatform=NZKPT&v=yJDsWGcr4R_YPT0P5y4X7OqmqIVIYD1jCtxAlnJ_yT1_Ea87YNTQHyLXqd2VB0Wq)
>   - 年份期刊：2021/红外技术
>   - 引用次数：9
>   引用：[1]何谦,刘伯运.红外图像边缘检测算法综述[J].红外技术,2021,43(03):199-207.
***
> + [ 红外图像降噪与增强技术综述](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2016&filename=WXDG201610001&uniplatform=NZKPT&v=goAoTKz4zcD2gvcVIcKuU3XZqkyHDP7NuM-7eRRS5hJO73v2gHI3Qf8U9amCghgA )
>   - 年份期刊：2016/无线电工程
>   - 引用次数：25
>   引用：[1]王洋,潘志斌.红外图像降噪与增强技术综述[J].无线电工程,2016,46(10):1-7+28.
***
> + [红外图像质量的提升技术综述 ]( https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2019&filename=HWJS201910009&uniplatform=NZKPT&v=ydpihIHwBCiSPBdNWok_Uip9m89Acpy5e9MMVWInll1v5q3gH_IBK0RQSAPlqAKd )
>   - 年份期刊：2019/红外技术
>   - 引用次数：16
>   引用：[1]凡遵林,管乃洋,王之元,苏龙飞.红外图像质量的提升技术综述[J].红外技术,2019,41(10):941-946.
***

#### 注意力机制
> + [ 注意力机制综述](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=JSJY2021S1002&uniplatform=NZKPT&v=5lc3RO-EUUBWu3fSJdcCaIlIGfqocgrazfJPGxFxW75igXkCecBI4_4WuK8ghFDY )
>   - 年份期刊：2021/计算机应用
>   - 引用次数：81
>   引用：[1]任欢,王旭光.注意力机制综述[J].计算机应用,2021,41(S1):1-6.
***
> + [ 卷积神经网络中的注意力机制综述]( https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=JSGG202120007&uniplatform=NZKPT&v=3Qinehu7XHKNZM5rzwWQIUmDT-4gFKy7FWkqmx_0R6r-c9bZw7YyqU7h1b3OAHia)
>   - 年份期刊：2021/计算机工程与应用
>   - 引用次数：28
>   引用：[1]张宸嘉,朱磊,俞璐.卷积神经网络中的注意力机制综述[J].计算机工程与应用,2021,57(20):64-72.
***
> + [深度学习推荐模型中的注意力机制研究综述](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2022&filename=JSGG202209001&uniplatform=NZKPT&v=ZhiUyUsnOHBTM4hKrlREm25vTX7BpvO-J9lucdsUBOtYA_GcM3llntAI9Ohg6wht)
>   - 年份期刊：2022/计算机工程与应用
>   - 引用次数：2
>   引用：[1]高广尚.深度学习推荐模型中的注意力机制研究综述[J].计算机工程与应用,2022,58(09):9-18.
***

### 已读论文总结

> + [基于多尺度特征融合的红外弱小目标检测](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CPFD&dbname=CPFDLAST2022&filename=ZGZN202110001001&uniplatform=NZKPT&v=yTXq56-lHUARleZolmCPC_wu5aSCr8WBIiYMVMsS0Y3ZFylOMRl1IQ_K-rmFqTpb99SccLOfFxY%3d)
>   - 引用：[1]孙召进,王国刚,刘云鹏. 基于多尺度特征融合的红外弱小目标检测[C]//.2021中国自动化大会论文集.[出版者不详],2021:2-7.DOI:10.26914/c.cnkihy.2021.053472.
>   - 总结：    该论文针对在红外弱小目标检测过程中经典的深度学习方法存在的检测效果较差、虚警率高等问题，提出一种基于YOLOv3的增强多尺度特征融合算法，有效的应用于红外弱小目标的检测。通过对YOLOv3在网络结构改进残差单元数量，将底层特征与输出特征进行融合以此提升对小尺度目标的检测能力。同时改进方法采用高斯损失函数，增加最大池化层，进而降低弱小目标的检测虚警率。实验结果表明，所提出的改进方法在地/空背景下红外图像弱小飞机目标检测跟踪数据集与YOLOv3算法对比，在准确率上提升2%，在召回率上提升5%，整体AP值提升7.67%。
***
> + [Automated Image Data Preprocessing with DeepReinforcement Learning](https://arxiv.org/abs/1806.05886)
>   - 引用：Minh T N, Sinn M, Lam H T, et al. Automated image data preprocessing with deep reinforcement learning[J]. arXiv preprint arXiv:1806.05886, 2018.
>   - 总结：提出了一种通过元学习实现数据预处理自动化的方法。然而，他们的方法只关注结构化数据进行有限的相对简单的标准化、离散化预处理技术。此外，这种情况下的预处理不会处理单个数据实例，而是应用于整个数据集。
***
> + [Oriented RepPoints for Aerial Object Detection](https://arxiv.53yu.com/pdf/2105.11111.pdf)
>   - 代码：https://github.com/LiWentomng/OrientedRepPoints
>   - 引用：Li W, Chen Y, Hu K, et al. Oriented reppoints for aerial object detection[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 1829-1838.
>   - 总结：提出了一种面向航空图像的目标检测器（为面向代表点），它引入了不同方向、形状和姿态的自适应点表示，也引入了一个有效的空间约束（惩罚函数）。论文提出三个有向转换函数、有效的自适应点评估与样本分配(APAA)方案和空间约束方法，不仅实现了 具有精确定向的精确航空检测，而且还捕获了任意定向航空 实例的底层几何结构。
***
> + [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/pdf/2202.06934)
>   - 引用：Akyon, Fatih Cagatay, Sinan Onur Altinuc, and Alptekin Temizel. "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." arXiv preprint arXiv:2202.06934 (2022)
>   - 总结：本文提出了一个名为切片辅助超推理（SAHI）的开源框架，与Detectron2、MMDetection和YOLOv5模型集成，它为小目标检测提供了一个通用的切片辅助推理和微调管道。这个技术是通用的，它可以应用在任何可用的物体探测器上，而无需任何微调。
***
> + [YOLO-Z: Improving small object detection in YOLOv5 for autonomous vehicles](https://arxiv.org/pdf/2112.11798.pdf)
>   - 引用：Benjumea A, Teeti I, Cuzzolin F, et al. YOLO-Z: Improving small object detection in YOLOv5 for autonomous vehicles[J]. arXiv preprint arXiv:2112.11798, 2021.
***

> + [Dense Nested Attention Network for Infrared Small Target Detection](https://arxiv.org/pdf/2106.00487.pdf?ref=https://githubhelp.com)
>   - 引用：Li B, Xiao C, Wang L, et al. Dense nested attention network for infrared small target detection[J]. arXiv preprint arXiv:2106.00487, 2021.
>   - 总结：现有的基于CNN的方法不能直接应用于红外小目标，因为在其网络中汇集层可能会导致深层目标的丢失。本文提出了一种密集嵌套注意力网络（DNANet），设计了一个密集嵌套交互模块（DNIM），以实现高级和低级特征之间的渐进交互，通过在非负离子显微镜中的重复交互作用，可以维持深层中的红外小目标。基于DNIM，进一步提出了级联通道和空间注意模块（CSAM），以自适应增强多级特征。开发了一个红外小目标数据集（即NUDT-SIRST），并提出了一组评估指标来进行全面的性能评估。
***
> + [Infrared Target Detection in Cluttered Environments by Maximization of a Target to Clutter Ratio (TCR) Metric Using a Convolutional Neural Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9199539)
>   - 引用：McIntosh B, Venkataramanan S, Mahalanobis A. Infrared target detection in cluttered environments by maximization of a target to clutter ratio (TCR) metric using a convolutional neural network[J]. IEEE Transactions on Aerospace and Electronic Systems, 2020, 57(1): 485-496.
>   - 总结：本文定义了一个TCR度量（TCR度量指网络输出对目标和杂波的响应所产生的能量的比率），通过解析推导出同时表示目标并将其与杂波区分开来的最佳特征向量，此度量最大限度地表示目标，同时最小化杂波的影响，这些特征向量被用作特征提取的第一层的滤波器。然后使用TCR度量的修改版本作为成本函数对网络的其余部分进行培训。将这种混合结构称为TCR网络进行目标检测，效果优。
***
> + [Slim-neck by GSConv: A better design paradigm of detector architectures for autonomous vehicles](https://arxiv.org/abs/2206.02424)
>   - 引用：Li, Hulin, et al. "Slim-neck by GSConv: A better design paradigm of detector architectures for autonomous vehicles." arXiv preprint arXiv:2206.02424 (2022).
>   - 总结：为了使 DSC 的输出尽可能接近 SC，引入了一种新方法 GSConv 来代替 SC 操作，使卷积计算的输出尽可能接近 SC，同时降低计算成本；
***
> + [MPANET: MULTI-PATCH ATTENTION FOR INFRARED SMALL TARGET OBJECT DETECTION](https://arxiv.org/abs/2206.02120)
>   - 引用：Wang A, Li W, Wu X, et al. MPANet: Multi-Patch Attention For Infrared Small Target object Detection[J]. arXiv preprint arXiv:2206.02120, 2022.
>   - 总结：本文提出了一种基于轴向注意力编码器和多尺度补丁分支（MSPB）结构的多补丁注意力MPANet网络，设计了一种轴向注意力改进编码器架构，以突出小目标的有效特征并抑制背景噪声，而无需任何分类主干。并在SIRST数据集上的大量实验表明，与现有方法相比，所提出的MPANet具有优越的性能和有效性。
***
> + [A Multi-task Framework for Infrared Small  Target Detection and Segmentation](https://arxiv.org/abs/2206.06923)
>   - 引用：Chen, Yuhang, et al. "A Multi-task Framework for Infrared Small Target Detection and Segmentation." arXiv preprint arXiv:2206.06923 (2022).
>   - 总结：本文提出了一种新颖的端到端红外小目标检测与分割框架：分支使用UNet作为目标检测骨干，改进的CenterNet作为目标检测头。利用UNet作为主干维护分辨率和语义信息，通过附加一个简单而有效的无锚检测或分割头部，模型可以达到比其他先进的方法更高的检测精度。
***
> + [A lightweight and accurate YOLO-like network for small target detection in Aerial Imagery](https://arxiv.org/abs/2204.02325)
>   - 引用：Betti A. A lightweight and accurate YOLO-like network for small target detection in Aerial Imagery[J]. arXiv preprint arXiv:2204.02325, 2022.
>   - 总结：提出了两种新颖的类yolo架构，专门设计来满足小目标检测的要求:YOLO-L和YOLO-S。提出了一个小型、简单、快速和高效的网络YOLO-S。它采用了一个较小的特征提取器Darknet20和单一的细粒度输出尺度，通过上采样，利用残差连接和特征融合，加强特征融合。
***
> + [QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html)
>   - 引用：Yang C, Huang Z, Wang N. QueryDet: Cascaded sparse query for accelerating high-resolution small object detection[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 13668-13677.
>   - 总结：本文提出了QueryDet，是基于级联稀疏查询（CSQ）的方法，利用低分辨率特征图预测小目标的粗位置，然后在粗位置的稀疏引导下利用高分辨率特征计算精确的检测结果。
***

## Datasets 数据集
### 自然图像
| Name    | Size   | object | Link    |
| :--- | :--- | :------ | :------ |
| The RSNA Pulmonary Embolism CT Dataset    | 980.24GB | 2,995,147 targets   | [Downoad](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/data?select=train.csv)           |
| CT Pulmonary Angiography   | 145.13GB    | 408,856 targets  | [Downoad](https://stanfordaimi.azurewebsites.net/datasets/12c02840-2e13-42a2-b4ef-f682472d4694)                |


