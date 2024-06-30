# Neural-Network -Final-task2
 CNN 和 Transformer模型在cifar-100数据集上训练并测试

### 文件和目录说明

- `main.py`：主程序文件，包含模型训练和评估的主要流程。
- `data/cifar100.py`：包含 CIFAR-100 数据集的加载和预处理代码。
- `models/cnn.py`：定义 ResNet18 模型的结构。
- `models/transformer.py`：定义 Vision Transformer 模型的结构。
- `utils/augmentations.py`：包含数据增强方法，如 CutMix。
- `utils/scheduler.py`：包含学习率调度器。
- `utils/train_eval.py`：包含训练和评估函数。

请自行下载 CIFAR-100 数据集并解压到 data 目录下。
解压后，确保数据集文件夹结构如下：

       
```kotlin
data
│  cifar100.py
│
└─cifar-100-python
        file1
        file2
        ...
```
