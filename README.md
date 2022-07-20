## 工作清单

### Finish

* FGM
* EMA
* text_input: title + ocr_text + asr_text
* label smooth
* Resample
* 双流 albert
* 双流 lxmert
* torch 2 onnx 2 trt fp16
* tfidf
* AdamW 换 BertAdam

### TODO

* 预训练: mlm + itm
* 调参

## [2022中国高校计算机大赛-微信大数据挑战赛](https://algo.weixin.qq.com/)

### 赛题介绍

多模态短视频分类是视频理解领域的基础技术，在安全审核、推荐运营、内容搜索等领域有着非常广泛的应用。
微信视频号每天有海量的短视频创作，我们需要用算法对这些视频分类。分类体系由产品预先定义。
我们从线上抽样真实的视频号数据，并提供视频的标题、抽帧、ASR、OCR等多模态信息，以及部分人工标注，要求参赛队伍基于这些数据，训练视频分类模型。
赛题的主要挑战包括：分类的分布不均衡，无标注数据多而有标注数据少，模态缺失，层次分类等。

大赛官方网站：https://algo.weixin.qq.com/

### 代码介绍

大部分核心代码、函数与初赛保持一致：

- [category_id_map.py](category_id_map.py) 是category_id 和一级、二级分类的映射
- [config.py](config.py) 是配置文件
- [data_helper.py](data_helper.py) 是数据预处理模块
- [evaluate.py](evaluate.py) 是线上评测代码示例
- [inference.py](inference.py) 是生成提交文件的示例代码
- [main.py](main.py) 是训练模型的入口
- [model.py](model.py) 是baseline模型
- [util.py](utils/util.py) 是util函数

区别在于，复赛不再提供预提取好的视觉特征，因此需要选手在训练、测试中自行提取视觉特征。在代码中，增加了如下文件：

- [swin.py](utils/swin.py) 初赛所用的 Swin-Transformer （tiny） 视觉特征提取器
- [extract_feature.py](extract_feature.py) 提取视觉特征的函数（仅供参考，实际是端到端的训练，并未单独提取特征）

在 `data_helper.py` 以及 `model.py` 中，将直接使用 swin 模型来端到端地提取视觉特征。
注意，由于端到端的视觉特征提取非常耗费计算资源，因此，将输入的视频帧从初赛 baseline 的32帧改为8帧。

复赛中使用 docker 容器来提交测评。因此本代码中提供了 Dockerfile 与入口脚本 start.sh 的示例：

- [start.sh](start.sh) 运行入口脚本
- [Dockerfile](dockerfile) Docker 构建脚本示例


### Docker 提交

（注：Docker 相关命令需要 sudo 权限来运行。）

1. 获取队伍的访问凭证。

  选手可以通过 https://console.cloud.tencent.com/tcr/token 来创建一个临时、或者永久的访问凭证。
  
  获取账号密码之后，使用
  ```
  sudo docker login tione-wxdsj.tencentcloudcr.com --username （账户id） --password (访问凭证)
  ```
  命令来登录。若成功，应该能看到 Login successfully 的字样。
  

2. 通过预定义好的 Dockerfile 创建 docker image

  ```bash
  sudo docker build -t tione-wxdsj.tencentcloudcr.com/team-xx/challenge:v1.0 .
  ```

  * 注1：`xx` 为你的队伍 ID，后面的 `challenge:v1.0` 可自行命名。
  * 注2：请勿将训练数据打包进 docker 容器中，这样会导致容器太大。在推理测试的时候，测试数据会自动挂载到 /opt/ml/input/ 目录下。
  * 注3：请确保入口工作目录（WORKDIR）为 /opt/ml/wxcode 
  

3. 推送 docker image 至 服务器
  
  使用 docker push 命令，将本地的构建好的镜像推送到队伍的镜像仓库中，并在官网提交 image 的名字。
  
  ```bash
  sudo docker push tione-wxdsj.tencentcloudcr.com/team-xx/challenge:v1.0
  ```

### 预热阶段验证代码是否可以运行

在预热阶段，平台提供的计算资源是 CPU 机器，且不开放正式的比赛数据。因此选手无法完成完整的训练流程。

但是可以通过少量的 demo 数据，来熟悉使用环境，并检查自己的代码是否可以运行。

例如，在 baseline 代码中，我们可以指定训练数据的路径为 demo ，并将 batch-size 设置为 1 来检验代码：

```
python main.py \
    --train_annotation ../demo_data/annotations/semi_demo.json \
    --train_zip_frames ../demo_data/zip_frames/demo/ \
    --batch_size 1 \
    --val_batch_size 1 \
    --print_steps 1
```

