# 猫狗大战 (Dogs vs Cats)

## 项目概述

## 项目指南

1. 克隆存储库并进入到项目文件夹。

   ```
   git clone https://github.com/SimonLeeGit/udacity_project.git
   cd udacity_project/dogs_vs_cats/
   ```

2. 默认所需数据集没有包含在项目目录中，可以通过kaggle命令进行下载，只需要在dogs_vs_cats目录下执行如下命令。（需要安装kaggle环境，详情请参考[https://github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api)）
   
   ```
   kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
   ```
   
   如果需要从kaggle上下载，可以通过链接[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)下载，下载后解压到dogs_vs_cats目录下，如下所示：
   
   > dogs_vs_cats/     
   >> bottleneck_features.ipynb  
   >> README.md  
   >> requirements/  
   >> **sample_submission.csv**  
   >> **test.zip**  
   >> **train.zip**  
   >> transfer_learning.ipynb
       

3. 配置安装环境及相关的Python依赖包。

   对于 __Linux__：

   ```
   conda env create -f requirements/linux.yml
   conda activate dogs_vs_cats
   KERAS_BACKEND=tensorflow python -c "from keras import backend"
   ```

   对于 __Mac/OSX__：

   ```
   conda env create -f requirements/mac.yml
   conda activate dogs_vs_cats
   KERAS_BACKEND=tensorflow python -c "from keras import backend"
   ```

   对于 __Windows__：

   ```
   conda env create -f requirements/windows.yml
   conda activate dogs_vs_cats
   set KERAS_BACKEND=tensorflow
   python -c "from keras import backend"
   ```

4. 打开notebook，打开**bottleneck_features.ipynb**，以提取迁移学习的bottleneck特征。

   ```
   jupyter notebook bottleneck_features.ipynb
   ```

5. 打开notebook，打开**transfer_learning.ipynb**，使用提取的bottleneck进行迁移学习。

   ```
   jupyter notebook transfer_learning.ipynb
   ```
   
6. 提交预测结果到kaggle官网中，执行如下命令。

   ```
   kaggle competitions submit -c dogs-vs-cats-redux-kernels-edition -f submission.csv -m "Message"
   ```
