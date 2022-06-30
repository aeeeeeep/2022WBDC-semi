# wbdc2022-preliminary-a268bc75faf4491d95f20675d69406b2

## 环境依赖

| 软件         | 软件要求     |
| ------------ | ------------ |
| python 版本  | 3.8.10       |
| CUDA 版本    | 11.3         |
| PyTorch 版本 | 1.10.0       |
| 操作系统     | Ubuntu 20.04 |

## 代码结构

```
.
├── data
├── inference.sh	# 模型测试脚本 
├── init.sh	# 初始化脚本 
├── README.md
├── requirements.txt # pip 依赖包
├── src
│   ├── category_id_map.py	# category_id 和一级、二级分类的映射
│   ├── config
│   │   └── config.json	# 预训练网络配置
│   ├── config.py	# 配置文件
│   ├── data_helper.py	# 数据预处理模块
│   ├── inference.py	# 生成提交文件的代码
│   ├── main_fgm.py	# 使用fgm对抗方法训练模型的入口
│   ├── main_pgd.py	# 使用pgd对抗方法训练模型的入口
│   ├── model.py	# 算法模型
│   ├── third_party	# 开源的第三方代码文件
│   │   ├── ema.py
│   │   ├── fgm.py
│   │   ├── focal_loss.py
│   │   ├── masklm.py
│   │   └── pgd.py
│   ├── util.py	# util函数
│   └── weights	# 权重目录
└── train.sh	# 模型训练脚本
```

## 运行流程说明

```shell
bash init.sh	# 安装需要的 python 包
bash train.sh	# 开始训练
bash inference.sh	# 模型融合和预测结果
```

## 算法模型介绍

多模态模型结构与参数量和 Bert 一致
layer=12, hidden_size=768, num_attention_heads=12
frame_input 通过 frame_dense 降维为 768 维，与 text_embedding 拼接
其输入为[CLS] Video [SEP] Text [SEP]，encoder 后得到特征
对 encoder 后的特征接 mean_pooling 和全连接层降维得到了比较好的效果

### 数据预处理

对于提供的 title，osr，acr 前后各取64

### 模型训练

bert 初始化权重来自于在中文语料预训练过的开源模型，没有使用提供的 unlabel 数据集进行预训练。
https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main
超参：batch_size=64, epoch=5, learning_rate=5e-5
模式在训练了 5 tuijianepoch 后会有过拟合现象，4 ~ 5 epoch 效果最佳。

(1) Fast Gradient Method
为了使 loss 更小，通过在 embedding 上添加对抗扰动，生成对抗代码，通过训练，准确率提高了 0.7 个百分点。

(2)  Projected Gradient Descent
相比于普通的 FGM 仅做一次迭代，PGD 做多次迭代，每次迭代都会将扰动投射到规定范围内，但通过实验，PGD 的 mean_f1 下降慢于 FGM，但在 5 epoch 后的准确率高于 FGM 0.2个百分点，两种方法作融合后准确率又会提高 0.2 百分点。

(3) Exponential Moving Average
以指数式递减加权的移动平均，衰减率为 0.999，控制模型的更新速度。

(4) label smooth
使用 label smooth 正则化技巧，提高模型的泛化性能和准确率，设置 label_smoothing=0.1，可以提高 0.5 个百分点。

(5) K-折交叉验证
训练集中使用了 90% 的 label 数据，为 9w，验证集使用了 10%，为 1w
经过实验，10折交叉验证比5折交叉验证准确率高 0.3 个百分点。

### 模型融合

模型都使用了 bert 这种结构，通过对 2 个模型， 10 折交叉验证后的 20 个权重的预测结果进行 mean 和 argmax 操作，得到最终的预测结果，经实验，准确率提升了 1 个百分点。

## 模型初赛B榜在线结果

得分: *0.686206*

## 开源预训练模型

https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main
