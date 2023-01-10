## 训练时间

预训练: 8h

微调: 8h

## 环境依赖

| 软件         | 软件要求         |
| ------------ |--------------|
| python 版本  | 3.8.10       |
| CUDA 版本    | 11.4         |
| PyTorch 版本 | 1.10.0       |

## 代码结构

```
.
├── Dockerfile                          # 复赛代码提交镜像的 Dockerfile 文件
├── inference.sh                        # 模型测试脚本
├── init.sh                             # 初始化脚本
├── opensource_models                   # 第三方开源预训练权重
│   ├── chinese-roberta-wwm-ext
│   └── swin_tiny_patch4_window7_224.pth
├── pretrain                            # 预训练权重
├── README.md
├── requirements.txt                    # pip 依赖包
├── save                                # 微调权重
├── src                                 # 源代码
│   ├── category_id_map.py
│   ├── config_bert.json                # 配置文件
│   ├── config.json                     # 配置文件
│   ├── config.py                       # 配置文件
│   ├── data_helper.py
│   ├── evaluate.py
│   ├── extract_feature.py
│   ├── inference.py                    # 生成提交文件的代码
│   ├── lxmert_main.py                  # 微调源代码
│   ├── lxmert_model_pretrain.py        # 预训练源代码
│   ├── lxmert_model.py                 # 微调模型源代码
│   ├── lxmert_pretrain.py              # 预训练模型源代码
│   └── utils                           # 功能函数目录
│       ├── ema.py
│       ├── fgm.py
│       ├── file_utils.py
│       ├── masklm.py
│       ├── modeling.py
│       ├── optimization.py
│       ├── pgd.py
│       ├── swin.py
│       └── util.py
└── train.sh                            # 模型训练脚本
```

## 运行流程说明

```shell
bash init.sh        # 安装需要的 python 包
bash train.sh       # 模型训练脚本
bash inference.sh   # 模型测试脚本
```

## 算法描述

### 简介

多模态模型结构参考自 [LXMERT](https://github.com/airsplay/lxmert)
模型的输入分为视频特征和文本特征, 视频特征是视频帧使用`swin_tiny`提取特征后, 再通过`frame_dense`降维为 768 维, 文本特征是将预处理后的文本信息经过`embedding`后得到文本信息处理部分与参数量和 Bert 一致 `layer=12, hidden_size=768, num_attention_heads=12`。

最后将处理后的文本特征和视频特征经过5层`Cross-Modality Encoder`后得到三个多模态输出。

![figure1](https://github.com/aeeeeeep/wbdc2022-semi/blob/main/picture/figure1.png)

### 数据预处理

使用`tdidf`对`title + osr + acr`拼接后的字符串提取特征, 再与`title`相加后前后各取64, 得到`tfidf_str_title`
将`osr`, `acr`前后各取64, 最后`tfidf_str_title + osr + acr`得到文本信息

由于模型性能优化不足, 在限制时间的前提下, 只能平均步长, 使用8帧训练

### 预训练

预训练采用了`Mask language model`, `Image Text Matched` 两个任务, `bert` 初始化权重来自于在中文语料预训练过的开源模型

(1) `Mask language model` 任务

与常见的自然语言处理`mlm`预训练方法相同, 对`text`随机 15% 进行 `mask`, 预测`mask`词。

多模态场景下, 结合视频的信息预测 `mask` 词, 可以有效融合多模态信息。

(2) `Image Text Matching` 任务

为了判断模型将文本信息与视觉信息的对齐效果, 训练阶段对于一个`batch`中的视频帧, 对后50%逆序, 模型中的分类器需要判断文本与视频帧是否匹配。

(3) 多任务联合训练

预训练任务的`loss`采用了上述两个任务`loss`的加权和

`L = L(mlm) + L(itm) * 0.3`

超参: `batch_size=50, num_workers=15, learning_rate=5e-5, max_epochs=5`

预训练更多的`epoch`对效果提升比较大, 对下游任务`finetune`效果提升显著, 但由于失误, 时间有限，固只训练一个`epoch`

### 微调

`bert`初始化权重来自于在中文语料预训练过的开源模型, 使用提供的`unlabel`数据集进行预训练。

https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main

模型在训练了 3 epoch 后会有过拟合现象, 3 ~ 4 epoch 效果最佳。
微调使用的以下`trick`:

(1) `Fast Gradient Method`

为了使`loss`更小, 通过在`embedding`上添加对抗扰动, 生成对抗代码, 通过训练, 准确率提高了 0.7 个百分点。

(2) `Exponential Moving Average`

以指数式递减加权的移动平均, 衰减率为 0.999, 控制模型的更新速度。

(3) `label smooth`

使用 label smooth 正则化技巧, 提高模型的泛化性能和准确率, 设置 label_smoothing=0.1, 可以提高 0.5 个百分点。

(4) `Resample`

将200分类中类别数量少于 100 的样本 *5, 少于 500 的样本 *3, 少于 1000 的样本 *2, 可以提高 0.3 百分点。

超参: `batch_size=26, num_workers=13, learning_rate=4e-5, max_epochs=5`

## 模型复赛B榜在线结果

得分: *0.678031*

排名: 72

## 开源预训练模型

swin_tiny_patch4_window7_224: 比赛方提供

chinese-roberta-wwm-ext: https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main

[![Security Status](https://www.murphysec.com/platform3/v3/badge/1610998808142446592.svg?t=1)](https://www.murphysec.com/accept?code=b1f8cb8fda1f9d090cd7a257f034ac05&type=1&from=2&t=2)
