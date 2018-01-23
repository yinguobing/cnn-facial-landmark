# cnn-facial-landmark

Facial landmark detection based on convolution neural network.

The model is build with TensorFlow, the training code is provided so you can train your own model with your own datasets.

Here are some sample gifs extracted from video file showing the detection result compared with Dlib. The result of CNN is on the right side.

![](https://github.com/yinguobing/cnn-facial-landmark/blob/master/demo01.gif)

![](https://github.com/yinguobing/cnn-facial-landmark/blob/master/demo02.gif)

![](https://github.com/yinguobing/cnn-facial-landmark/blob/master/demo03.gif)

## Background
This repo is a part of my deep learning series posts. For all the posts please refer to the following links.

### 第一篇：基于深度学习的人脸特征点检测 - 背景

为什么我决定采用深度学习实现面部特征点检测。[阅读全文](https://yinguobing.com/facial-landmark-localization-by-deep-learning-background/)

### 第二篇：基于深度学习的人脸特征点检测 - 数据与方法

解决问题所需的数据来源与对应的方法。[阅读全文](https://yinguobing.com/facial-landmark-localization-by-deep-learning-data-and-algorithm/)

### 第三篇：基于深度学习的人脸特征点检测 - 数据集整理

从互联网获取的数据大多数情况下不是开箱即用的，这意味着我们需要对数据进行初步的整理，例如统计数据量、去除不需要的文件、必要的格式转换等。[阅读全文](https://yinguobing.com/facial-landmark-localization-by-deep-learning-data-collate/)

### 第四篇：基于深度学习的人脸特征点检测 - 数据预处理
如何使用Python从22万张图片中提取检测人脸特征点的可用样本。[阅读全文](https://yinguobing.com/facial-landmark-localization-by-deep-learning-preprocessing/)

### 第五篇：基于深度学习的人脸特征点检测 - 生成TFRecord文件
将面部区域的图片与特征点位置一起打包成TensorFlow可用的TFRecord文件。[阅读全文](https://yinguobing.com/facial-landmark-localization-by-deep-learning-tfrecord/)

### 第六篇：基于深度学习的人脸特征点检测 - 网络模型构建
如何使用TensorFlow构建一个属于你自己的神经网络模型。[阅读全文](https://yinguobing.com/facial-landmark-localization-by-deep-learning-network-model/)

### 第七篇：基于深度学习的人脸特征点检测-模型导出与应用
使用Estimator API时，导出适用于推演的网络模型的正确方法。[阅读全文](https://yinguobing.com/facial-landmark-localization-by-deep-learning-save-model-application/)

### 第八篇：基于深度学习的人脸特征点检测-移植到iPhone
如何通过CoreML在iPhone应用中使用TensorFlow模型。[阅读全文](https://yinguobing.com/facial-landmark-localization-by-deep-learning-port-to-ios-with-coreml/)

![dl-posts](https://yinguobing.com/content/images/2018/01/dl-posts.jpg)
