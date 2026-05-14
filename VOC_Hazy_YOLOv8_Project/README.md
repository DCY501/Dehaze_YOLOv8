# VOC Hazy YOLOv8 Object Detection Project

本项目是基于 YOLOv8 的雾天图像目标检测课程设计项目，主要用于 VOC_hazy 数据集上的目标检测实验。

## Project Structure

- `configs/`: 数据集配置文件和模型配置文件
- `tools/prepare_voc_hazy.py`: VOC2007 雾天数据集生成脚本
- `train.py`: 训练脚本
- `predict.py`: 预测脚本
- `examples/`: 少量示例图片
- `docs/`: 课程设计说明文档
- `src/`: 课程设计中涉及的关键源码备份

## Environment

建议使用 Anaconda 创建环境：

```bash
conda create -n yolo8hazy python=3.9
conda activate yolo8hazy
pip install -r requirements.txt
```

## Prepare Dataset

本仓库不包含完整 VOC 数据集。可以运行以下命令自动下载 VOC2007，并生成带雾版本：

```bash
python tools/prepare_voc_hazy.py --root datasets
```

生成后，数据集配置文件 `configs/VOC_hazy.yaml` 会指向：

```text
datasets/VOC_hazy
```

## Training

默认使用 YOLOv8n 预训练权重训练：

```bash
python train.py
```

如果需要使用自定义模型结构，可以把 `train.py` 中的模型加载方式改为：

```python
model = YOLO("configs/yolov8-dehaze.yaml")
```

## Prediction

将训练得到的 `best.pt` 放在项目根目录后运行：

```bash
python predict.py
```

## Notes

由于数据集、训练结果和模型权重文件较大，本仓库不包含 `datasets/`、`runs/` 和 `.pt` 权重文件。
