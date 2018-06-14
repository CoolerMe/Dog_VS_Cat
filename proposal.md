# 毕业项目<[猫狗大战](https://github.com/nd009/capstone/tree/master/dog_vs_cat)>开题报告

## 项目背景
 2013年 Kaggle举行了[Cat Vs Dog](https://www.kaggle.com/c/dogs-vs-cats)比赛.随着时间过去,机器学习领域已经发生了太多变化,尤其是在深度学习和图像识别方面,Google的Tensorflow更是在其中大放异彩.几年过去了,猫狗大战也变得不那么困难.识别率从几年前的82.7%[1]飙升到[98.914%[2].而30s内人对猫和狗的识别率能达到99.6%,超过计算机.所以我也想尝试下,看能否训练一个识别准确率比较高的模型来加深对神经网络和深度学习的理解.

##  问题描述
输入一张彩色照片,输出是狗的概率.目前有一个方案是结合支持向量机和从图片中得到的颜色纹理来识别图像.问题通过将输入的图片,转换为狗的概率来实现量化输出.

##  输入数据 
* 训练集:25000张已经被标记好的猫和狗的图片,是[Asirra](http://research.microsoft.com/en-us/um/redmond/projects/asirra/)的子数据集.而Assira是和[Petfinder.com](http://www.petfinder.com/)(全球最大的为流浪猫狗找到收养家庭的网站)进行和合作的,数据集来自全美几千个动物收容所并由人工区分的300万多张猫狗的照片.在这数据集中,最大的图片尺寸约为1050  $\times$  702,大小为90k左右.最小的约为60 $\times$  39,大小为2k左右,其中绝大部分的图片尺寸小于 500 $\times$ 500,且横向的照片居多.绝大部分图片是正常的猫狗,环境为常见的生活场景.剩下的图片被错误标注,如网站logo,花草等.所以会利用Keras已有的一些预加载模型来排除一些错误图片.所有输入模型的照片都会被缩放到224 $\times$ 224的大小.在训练的时候,大约有10% -20%的图片被随机分配为验证集.
* 测试集:12500张未被标记成猫和狗的图片,将测试好的结果上传来查看模型的成绩.
* [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/):扩充数据集,可取出其中一部分照片,来检验模型的预测的准确率.

##  解决办法 
模仿InceptionV3和使用Keras搭建深度神经网络,然后进行训练.同时根据验证和训练集损失曲线和参考Keras已有的几个模型慢慢调整模型复杂度和超参数来实现更复杂准确率更高的模型.然后将训练的模型保存为hdf5文件.

##  基准模型 
目前已有几个比较知名的模型,且当前Keras中已经预制了几个实现好的预训练的模型:VGGNet[3] ,ResNet[4],Inception v3[5],InceptionResNetV2[6],DenseNet[7],Xception[8], NASNet[9].
* 选用的基准模型是Inception V3,InceptionV3使用非对称卷积来代替常规卷积层,将省下的计算能力来加深网络深度[10].模型大小92M,在ImageNet的数据集上Top1-准确率0.788,Top-5准确率0.944,参数数量23,851,784,深度159[11].
* 选用基准阈值是Kaggle排行榜[5%](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/leaderboard),也就是测试集的Logloss小于0.05040.


##  评估指标 
$$
LogLoss= -\frac{1}{n}\sum\limits_{k=1}^n[y_i \log(\hat{y_i})-(1-y_i)\log(1-\hat{y_i})]
$$
目前使用的函数是对数损失函数,对数损失越小表明模型表现越好.全部预测成功损失就是0.用到了极大似然估计的思想[12],预测正确的概率越高,其损失值应该是越小,因此再加个负号取反.
*   n 是测试集中样本数量
*   $\hat{y}_i$  :预测图片是中是一条狗的可能性,1为狗,0是猫
*   $y_i$ :0表示图片中实际是猫,1为狗
*   log() :底数为e的对数函数

##  设计大纲
* 加载数据,使用Keras已经预训练好的模型ResNet,DenseNet对训练集进行预测,清除明显的错误数据.
* 将训练集10%-20%随机划分为验证集,使用Keras的ImageDataGenerator对图片进行预处理,模仿InceptionV3搭建模型
* 在模型的callback中保存模型,在fit的History回调中获取训练和验证集loss的历史记录,并做出曲线图.
* 根据曲线图,调整模型参数,如增加删减层,修改全连接层大小,修改学习率,dropout,ImageDataGenerator的参数,batch_size,epoch等参数.
* 使用扩充数据集的部分图片进行预测,分析预测错误的原因,修改模型
* 重复上面几个步骤,直到达到模型阈值,也就自己设定的目标.
* 保存模型.
* 输出预测数据到csv文件,上传Kaggle,得到实际成绩.
* 尝试应用到Android App上.
* 输出报告


## 引用部分
[1] [Machine Learning Attacks Against the Asirra CAPTCHA](http://xenon.stanford.edu/~pgolle/papers/dogcat.pdf)
[2] [Can we beat the state of the art from 2013 with only 0.046% of training examples?](https://medium.com/@radekosmulski/can-we-beat-the-state-of-the-art-from-2013-with-only-0-046-of-training-examples-yes-we-can-18be24b8615f)
[3] Very Deep Convolutional Networks for Large-Scale Image Recognition | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) |
[4] Deep Residual Learning for Image Recognition | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
[5] Rethinking the Inception Architecture for Computer Vision|  [arXiv:1512.00567](https://arxiv.org/abs/1512.00567) |
[6] Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning  |  [arXiv:1602.07261](https://arxiv.org/abs/1602.07261) |
[7] Densely Connected Convolutional Networks |  [arXiv:1608.06993](https://arxiv.org/abs/1608.06993) |
[8]Xception: Deep Learning with Depthwise Separable Convolutions  |  [arXiv:1610.02357](https://arxiv.org/abs/1610.02357)  |
[9]Learning Transferable Architectures for Scalable Image Recognition |  [arXiv:1707.07012](https://arxiv.org/abs/1707.07012) |
[10][深度学习卷积神经网络——经典网络GoogLeNet(Inception V3)网络的搭建与实现](https://blog.csdn.net/loveliuzz/article/details/79135583)
[11]模型概览 [https://keras.io/zh/applications/](https://keras.io/zh/applications/)
[12][浅析机器学习中各种损失函数及其含义](https://blog.csdn.net/qq547276542/article/details/77980042)


