# 第71题方案二：基于多任务学习的雾天目标检测框架导读

> 说明：本文是课程设计用的中文导读和论文要点翻译，不是论文全文逐字翻译。目标是帮助理解方案二的框架、术语和可落地实现方式。

## 1. 方案二一句话理解

方案二不是“先把雾图变清晰，再送进 YOLO”，而是让一个模型同时学习两件事：

- 主任务：检测目标在哪里、属于什么类别。
- 辅助任务：从雾图中恢复更清晰的图像或更干净的特征。

本科生可以这样理解：  
原始 YOLO 像一个只负责考试答题的学生；多任务 YOLO 像一个学生一边做检测题，一边被要求解释“这张图哪里被雾影响了、原本可能长什么样”。辅助任务会迫使模型学到更抗雾的特征，最后服务于检测。

## 2. 你负责的“框架和大体结构”到底是什么

你主要负责三件事：

1. 输入和输出怎么走  
   输入仍然是雾天图像。输出至少包括 YOLO 检测结果；训练时还可以额外输出去雾图或去雾特征。

2. 网络怎么分叉  
   Backbone/Neck 先提取共享特征，然后一条路走检测头，另一条路走去雾辅助头。

3. 推理阶段保留什么  
   最稳妥做法是：训练时使用去雾辅助分支，推理时只用检测分支。这样不会明显拖慢检测速度。

推荐最小可行结构：

```text
雾图 image
  |
YOLO Backbone + Neck  提取多尺度特征 P3/P4/P5
  |                         |
  |                         +--> 去雾辅助头：输出 clean-like image 或 dehaze feature
  |
  +--> Detect Head：输出 bbox / class / confidence
```

训练时总损失可以写成：

```text
L_total = L_det + lambda * L_dehaze
```

其中 `L_det` 是 YOLO 原本检测损失，`L_dehaze` 是去雾辅助损失，`lambda` 由负责调参和损失函数的同学处理。

## 3. 关键术语通俗解释

### 多任务学习

让一个模型同时学习多个相关任务。目标不是“每个任务都做到最强”，而是让辅助任务帮主任务学到更好的特征。

在本题中，检测是主任务，去雾是辅助任务。

### Backbone

特征提取主干。可以理解为模型的“眼睛”和“初级视觉皮层”，负责把原图变成不同层次的特征。

YOLOv8 中，`Conv`、`C2f`、`SPPF` 等模块组成 Backbone。

### Neck

特征融合部分。它把深层语义特征和浅层细节特征融合起来。可以理解为“把远景轮廓和近处细节拼起来”。

YOLOv8 中的上采样、Concat、C2f 等组成 Neck。

### Head

任务输出头。检测头负责输出框和类别；你要加的去雾辅助头负责输出复原图或辅助特征。

### 特征

不是肉眼看到的图，而是神经网络内部的一堆数字矩阵。浅层特征更像边缘、纹理、颜色；深层特征更像车、人、路牌等语义。

### 端到端训练

输入图像进来，检测和去雾的损失一起反向传播，模型参数一起更新。中间不需要手工拆成很多独立步骤。

### 共享参数

两个任务共用一部分网络层。例如检测和去雾共用 YOLO Backbone。好处是省显存，也能让两个任务互相影响。

### 辅助分支

主干网络旁边接出来的一条小路。它不是最终目的，而是训练时给模型多一个学习信号。

## 4. 在你当前 YOLOv8 代码里的落点

你的代码路径：

```text
C:/Python/project/ultralytics-yolov8-official
```

重点文件：

- `ultralytics/models/v8/yolov8.yaml`  
  定义 YOLOv8 的 Backbone、Neck、Detect Head。后续可以复制一份 `yolov8-dehaze.yaml`，在 Detect 外再接一个辅助头。

- `ultralytics/nn/modules.py`  
  定义基础模块，如 `Conv`、`C2f`、`Detect`、`Segment`。你要新增去雾头，可以先在这里写一个 `DehazeHead`。

- `ultralytics/nn/tasks.py`  
  按 yaml 搭网络。新增模块后，需要让 `parse_model()` 能识别你的 `DehazeHead`。

- `ultralytics/yolo/v8/detect/train.py`  
  训练逻辑和检测损失在这里。之后要把模型输出拆成检测输出和去雾输出，再交给损失函数。

## 5. 推荐实现路线

### 第一步：只跑原始 YOLOv8

目标：确认数据集、标签、训练、验证能跑通。

建议命令：

```bash
yolo detect train model=yolov8n.pt data=datasets/VOC_hazy/VOC_hazy.yaml imgsz=640 epochs=50 batch=8
```

如果显存不够，先把 batch 改成 4。

### 第二步：复制配置文件

复制：

```text
ultralytics/models/v8/yolov8.yaml
```

新建：

```text
ultralytics/models/v8/yolov8-dehaze.yaml
```

先不要改太多，只保留原检测结构，确认新 yaml 能正常加载。

### 第三步：新增轻量去雾头

先做一个非常简单的辅助头，不要追求图像复原特别漂亮。

输入可以来自 YOLO Neck 的 P3 特征，也就是 `yolov8.yaml` 里第 15 层附近的高分辨率特征。原因是 P3 分辨率较高，更适合恢复图像细节。

最小结构可以是：

```text
P3 feature
  -> Conv
  -> Upsample x2
  -> Conv
  -> Upsample x2
  -> Conv
  -> 输出 3 通道图像
```

### 第四步：训练时输出两个结果

模型 forward 训练时返回：

```text
detect_preds, dehaze_img
```

检测损失继续用原 YOLO 逻辑。去雾损失由搭档负责，例如 L1、MSE、SSIM 或它们的组合。

### 第五步：推理阶段只看检测结果

训练完成后，推理时可以丢弃或忽略去雾输出，只保留检测框。报告里可以写：辅助分支主要用于训练阶段增强特征鲁棒性。

## 6. 论文中文导读

### 6.1 AOD-Net: All-In-One Dehazing Network

来源：ICCV 2017, CVF Open Access  
链接：https://openaccess.thecvf.com/content_iccv_2017/html/Li_AOD-Net_All-In-One_Dehazing_ICCV_2017_paper.html

中文要点：

AOD-Net 是一个轻量级 CNN 去雾模型。它的关键思想是：不要分别估计透射率和大气光，而是把大气散射模型重新整理，让网络直接从雾图生成清晰图。论文还把 AOD-Net 接到 Faster R-CNN 前面，证明去雾模块可以改善雾图目标检测。

对你的项目的价值：

- 适合作为方案一 Baseline。
- 证明“去雾模块可以嵌入检测管线”这个思路有经典依据。
- 但它更偏级联式：先去雾，再检测；不是你主攻的共享 Backbone 多任务结构。

### 6.2 DSNet: Joint Semantic Learning for Object Detection in Inclement Weather Conditions

来源：IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021  
链接：https://ieeexplore.ieee.org/document/9022905/

中文要点：

DSNet 是非常贴近方案二的论文。它提出双子网结构：一个检测子网，一个复原子网。两个子网共享部分特征，训练时同时学习可见性增强、目标分类和目标定位。它的核心观点是：复原子网产生的“干净特征”能够帮助检测子网在恶劣天气下更好地分类和定位。

对你的项目的价值：

- 可以作为方案二的理论核心参考。
- 你的 YOLO 多任务结构可以写成 DSNet 思路的 YOLOv8 版本。
- 它强调“增强特征服务检测”，而不是只追求图像好看。

### 6.3 ODFC-YOLO: Multi-Task Learning for UAV Aerial Object Detection in Foggy Weather Condition

来源：Remote Sensing, 2023  
链接：https://www.mdpi.com/2072-4292/15/18/4617

中文要点：

ODFC-YOLO 是一个面向雾天目标检测的多任务 YOLO 框架。它包含检测子网和去雾子网，两个任务端到端联合训练。论文中特别重要的一点是：去雾子网只在训练阶段参与，推理阶段不作为检测输入，这样可以保持较快检测速度。

对你的项目的价值：

- 和你当前方案最接近。
- 可以借鉴“训练时有去雾辅助，推理时只检测”的写法。
- 你可以不用复现它的复杂 CSP-Decoder、GCEE、CAALoss，只实现课程设计可控的轻量版。

### 6.4 Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions

来源：AAAI 2022 / arXiv  
链接：https://arxiv.org/abs/2112.08088

中文要点：

IA-YOLO 不直接做传统去雾，而是在 YOLO 前加入可微图像处理模块，让模型根据每张图像自动选择增强参数。它和 YOLOv3 端到端联合训练，使图像增强为了检测任务服务。

对你的项目的价值：

- 说明增强模块可以和 YOLO 联合训练。
- 如果老师问“为什么不先固定去雾再检测”，可以用这篇解释：检测友好的增强应该和检测任务一起优化。
- 它更像“自适应预处理 + YOLO”，不是最标准的共享 Backbone 多任务结构。

### 6.5 DENet / DE-YOLO: Detection-driven Enhancement Network

来源：ACCV 2022, CVF Open Access  
链接：https://openaccess.thecvf.com/content/ACCV2022/html/Qin_DENet_Detection-driven_Enhancement_Network_for_Object_Detection_under_Adverse_Weather_ACCV_2022_paper.html

中文要点：

DENet 把图像分解成低频和高频信息，再做检测驱动的增强，最后和 YOLO 形成 DE-YOLO。它强调增强模块不能破坏对检测有用的潜在特征。

对你的项目的价值：

- 可以支持你的报告观点：去雾/增强不是越强越好，而是要检测友好。
- 如果后续做加分项，可以加入“特征门控”或“检测友好增强”的概念。

### 6.6 Detection-Friendly Dehazing: Object Detection in Real-World Hazy Scenes

来源：IEEE TPAMI, 2023  
链接：https://doi.org/10.1109/TPAMI.2023.3234976

中文要点：

这篇论文关注真实雾天场景中的检测友好去雾。它的重点不是单纯追求 PSNR/SSIM，而是让去雾结果更有利于下游目标检测。

对你的项目的价值：

- 是高水平期刊 TPAMI，适合放在相关工作里。
- 可以支撑你的核心论点：本项目评价重点应该是 mAP、Precision、Recall，而不是只看去雾图像是否漂亮。

## 7. 你的课程设计可以怎么表述创新点

建议写法：

本文以 YOLOv8 为基础检测器，设计一种面向雾天场景的轻量多任务检测框架。模型在共享 Backbone/Neck 特征的基础上，同时接入检测头与去雾辅助头。训练阶段通过去雾辅助监督约束共享特征，使模型学习更抗雾、更稳定的目标表征；推理阶段保留检测分支，从而兼顾检测精度和实时性。

如果后续加一点自己的改进，可以写：

在基础多任务结构上，引入轻量特征反馈模块，将去雾分支学习到的雾退化信息转化为注意力权重，对检测特征进行重标定，从而增强模型对低对比度目标和远距离目标的响应。

## 8. 下一步建议

你现在最该做的是：不要先改损失函数，也不要先追复杂论文结构。先把框架分三步跑通：

1. 原始 YOLOv8n/s 在 VOC_hazy 上训练和验证。
2. 新增 `DehazeHead`，确认模型能 forward，输出检测结果和去雾图。
3. 接入最简单的辅助损失，让训练不报错，再交给搭档调参。

只要这三步完成，你负责的“大体框架”就立住了。
