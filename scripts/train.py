"""
YOLOv8 雾天检测训练脚本
支持模式：基线训练(baseline) / 多任务训练(dehaze) / 可视化(visualize)

用法示例：
    # 1. 基线训练（原始 YOLOv8，检测单任务）
    python scripts/train.py --mode baseline --epochs 50 --name baseline_50e

    # 2. 多任务训练（YOLOv8 + DehazeHead，检测+去雾）
    python scripts/train.py --mode dehaze --epochs 50 --dehaze 0.05 --name dehaze_50e

    # 3. 快速验证（1 epoch，结构是否能跑通）
    python scripts/train.py --mode dehaze --epochs 1 --batch 4 --workers 0 --name test

    # 4. 去雾三联图可视化
    python scripts/train.py --mode visualize --weights runs/detect/train6/weights/best.pt --out-dir runs/dehaze_vis/train6
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml

# 将项目根目录加入路径，确保能导入本地 ultralytics
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def check_cuda():
    """检查 CUDA 和 GPU 状态"""
    print("=" * 55)
    print("环境检查")
    print("=" * 55)
    print(f"PyTorch 版本 : {torch.__version__}")
    print(f"CUDA 可用    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前 GPU     : {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本    : {torch.version.cuda}")
    else:
        print("警告: 未检测到 GPU，将使用 CPU 训练（极慢）")
    print("=" * 55)


def train_baseline(args):
    """
    基线训练：原始 YOLOv8 检测模型
    用于和去雾辅助分支做对比实验
    """
    print("\n>>> [模式: 基线训练] 原始 YOLOv8 检测")
    print(f"    模型 : {args.model}")
    print(f"    数据 : {args.data}")
    print(f"    Epoch: {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print(f"    预训练: {args.pretrained}\n")

    model = YOLO(args.model)
    # 修复：从 YAML 创建模型时，pretrained 参数不会被自动使用，
    # 必须显式调用 model.load() 才能加载预训练权重。
    if args.pretrained and str(args.pretrained).lower() not in ('false', 'none', '0', ''):
        print(f">>> 正在加载预训练权重: {args.pretrained}")
        model.load(args.pretrained)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        pretrained=args.pretrained,
        exist_ok=True,
        project=args.project,
        name=args.name,
        resume=args.resume,
        cache=args.cache,
        save_period=args.save_period,
    )
    print(f"\n>>> 基线训练完成，结果保存在: {args.project}/{args.name}")


def train_dehaze(args):
    """
    多任务训练：YOLOv8 + DehazeHead 辅助分支
    同时优化检测 loss 和去雾 L1 loss
    """
    print("\n>>> [模式: 多任务训练] YOLOv8-Dehaze")
    print(f"    模型   : {args.model}")
    print(f"    数据   : {args.data}")
    print(f"    Epoch  : {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print(f"    预训练 : {args.pretrained}")
    print(f"    Dehaze : {args.dehaze} (去雾辅助损失权重)\n")

    model = YOLO(args.model)
    # 修复：从 YAML 创建模型时，pretrained 参数不会被自动使用，
    # 必须显式调用 model.load() 才能加载预训练权重。
    if args.pretrained and str(args.pretrained).lower() not in ('false', 'none', '0', ''):
        print(f">>> 正在加载预训练权重: {args.pretrained}")
        model.load(args.pretrained)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        pretrained=args.pretrained,
        dehaze=args.dehaze,
        exist_ok=True,
        project=args.project,
        name=args.name,
        resume=args.resume,
        cache=args.cache,
        save_period=args.save_period,
    )
    print(f"\n>>> 多任务训练完成，结果保存在: {args.project}/{args.name}")


def run_visualize(args):
    """
    去雾三联图可视化
    输出: Hazy input | Dehaze output | Clean target
    """
    print("\n>>> [模式: 可视化] 生成去雾三联图")
    print(f"    权重   : {args.weights}")
    print(f"    数据   : {args.data}")
    print(f"    输出   : {args.out_dir}\n")

    vis_script = ROOT / "tools" / "visualize_dehaze_triplets.py"
    if not vis_script.exists():
        print(f"错误: 找不到可视化脚本 {vis_script}")
        print("请确认 tools/visualize_dehaze_triplets.py 存在")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(vis_script),
        "--weights", args.weights,
        "--data", args.data,
        "--split", args.split,
        "--imgsz", str(args.imgsz),
        "--num", str(args.num),
        "--device", str(args.device),
        "--out-dir", args.out_dir,
    ]
    subprocess.run(cmd, check=True)
    print(f"\n>>> 可视化完成，结果保存在: {args.out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 雾天目标检测训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
常用命令示例:
  # 基线训练 50 epoch
  python scripts/train.py --mode baseline --epochs 50 --name baseline_yolov8n

  # 多任务训练 50 epoch（dehaze=0.05）
  python scripts/train.py --mode dehaze --epochs 50 --dehaze 0.05 --name dehaze_005

  # 快速结构验证（1 epoch，无预训练）
  python scripts/train.py --mode dehaze --epochs 1 --batch 4 --workers 0 --pretrained False --name sanity_check

  # 可视化（默认读取 train6 权重）
  python scripts/train.py --mode visualize --weights runs/detect/train6/weights/best.pt --out-dir runs/dehaze_vis/train6
        """
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "dehaze", "visualize"],
        default="dehaze",
        help="训练/可视化模式: baseline=原始YOLOv8基线, dehaze=多任务(检测+去雾, 默认), visualize=三联图可视化"
    )

    # ---------- 通用参数 ----------
    parser.add_argument("--model", default="ultralytics/models/v8/yolov8-dehaze.yaml",
                        help="模型配置文件或权重路径 (默认: yolov8-dehaze.yaml)")
    parser.add_argument("--data", default="datasets/VOC_hazy/VOC_hazy.yaml",
                        help="数据集 YAML 路径 (默认: datasets/VOC_hazy/VOC_hazy.yaml)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch", type=int, default=8,
                        help="每批次图像数 (默认: 8，显存小可改 4)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸 (默认: 640)")
    parser.add_argument("--workers", type=int, default=0,
                        help="数据加载线程数 (Windows 建议 0，默认: 0)")
    parser.add_argument("--device", default="0",
                        help="计算设备: 0=GPU0, 1=GPU1, cpu=CPU (默认: 0)")
    parser.add_argument("--pretrained", default="yolov8n.pt",
                        help="预训练权重 (默认: yolov8n.pt; 不想加载预训练则填 False)")
    parser.add_argument("--project", default="runs/detect",
                        help="训练结果保存目录 (默认: runs/detect)")
    parser.add_argument("--name", default=None,
                        help="实验名称，结果保存在 project/name 下 (默认: 自动生成)")

    # ---------- Dehaze 多任务专用 ----------
    parser.add_argument("--dehaze", type=float, default=0.05,
                        help="去雾辅助损失权重 (默认: 0.05; 0=关闭去雾分支)")

    # ---------- 快速验证模式 ----------
    parser.add_argument("--quick", action="store_true",
                        help="快速验证模式: epochs=3, batch=4, workers=0, 自动命名")

    # ---------- 可视化专用 ----------
    parser.add_argument("--weights", default="runs/detect/train6/weights/best.pt",
                        help="用于推理可视化的权重路径")
    parser.add_argument("--split", default="val",
                        help="可视化使用的数据集划分 (默认: val)")
    parser.add_argument("--num", type=int, default=12,
                        help="可视化样本数量 (默认: 12)")
    parser.add_argument("--out-dir", default="runs/dehaze_vis/train6",
                        help="可视化输出目录 (默认: runs/dehaze_vis/train6)")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断的 checkpoint 恢复训练 (last.pt)")
    parser.add_argument("--cache", default=False, nargs="?", const=True,
                        help="缓存图片到内存(True)或硬盘('disk')，默认不缓存")
    parser.add_argument("--save-period", type=int, default=-1,
                        help="每隔 N 个 epoch 保存一次 checkpoint (-1=只保存 last/best，默认)")

    args = parser.parse_args()

    # 快速模式覆盖参数
    if args.quick:
        args.epochs = 3
        args.batch = 4
        args.workers = 0
        if args.name is None:
            args.name = f"quick_{args.mode}"
        print(f">>> 快速验证模式: epochs={args.epochs}, batch={args.batch}, workers={args.workers}")

    return args


def main():
    args = parse_args()
    check_cuda()

    # 断点恢复: 自动定位 last.pt，并从 args.yaml 恢复原始训练配置
    if args.resume:
        if args.name is None:
            print("错误: 恢复训练时必须指定 --name，例如: --name baseline_5beta_subset30_50e")
            sys.exit(1)
        run_dir = Path(args.project) / args.name
        last_pt = run_dir / "weights" / "last.pt"
        args_yaml = run_dir / "args.yaml"
        if not last_pt.exists():
            print(f"错误: 找不到 checkpoint: {last_pt}")
            print("可能原因: 训练尚未完成过至少 1 个完整 epoch，last.pt 还未生成。")
            print("请确认 --name 和 --project 是否正确，或该实验是否已跑完 1 个 epoch 以上。")
            sys.exit(1)
        print(f">>> 断点恢复模式: 从 {last_pt} 继续训练")
        args.model = str(last_pt)

        # 从 args.yaml 恢复原始训练配置，避免使用脚本默认值覆盖
        if args_yaml.exists():
            print(f">>> 读取原始训练配置: {args_yaml}")
            with open(args_yaml, "r", encoding="utf-8") as f:
                saved_args = yaml.safe_load(f) or {}
            # 关键参数：恢复训练必须保持一致
            restore_map = {
                "data": "data",
                "epochs": "epochs",
                "batch": "batch",
                "imgsz": "imgsz",
                "workers": "workers",
                "pretrained": "pretrained",
                "dehaze": "dehaze",
                "cache": "cache",
                "save_period": "save_period",
            }
            for saved_key, arg_key in restore_map.items():
                if saved_key in saved_args:
                    old_val = getattr(args, arg_key)
                    new_val = saved_args[saved_key]
                    setattr(args, arg_key, new_val)
                    if old_val != new_val:
                        print(f"    恢复 {arg_key}: {old_val} -> {new_val}")
        else:
            print(f"警告: 找不到 {args_yaml}，将使用当前命令行参数继续训练")

    if args.mode == "baseline":
        # 基线模式默认用原始 yolov8n.pt，而非 dehaze yaml
        if not args.resume and args.model == "ultralytics/models/v8/yolov8-dehaze.yaml":
            print("提示: 基线训练自动切换为原始 yolov8n.yaml")
            args.model = "ultralytics/models/v8/yolov8.yaml"
        train_baseline(args)

    elif args.mode == "dehaze":
        train_dehaze(args)

    elif args.mode == "visualize":
        run_visualize(args)


if __name__ == "__main__":
    main()
