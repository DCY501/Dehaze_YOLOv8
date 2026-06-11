import argparse
import math
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import yaml
from tqdm import tqdm


VOC_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# 5 种代表性 beta（轻雾到浓雾）
BETAS = [0.60, 0.80, 1.00, 1.35, 1.80]

# 默认子集采样参数
SUBSET_SEED = 42

URLS = {
    "VOCtrainval_06-Nov-2007.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip",
    "VOCtest_06-Nov-2007.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtest_06-Nov-2007.zip",
}


def download_file(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"exists: {dst.name}")
        return

    print(f"downloading: {url}")
    with urllib.request.urlopen(url) as response, dst.open("wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(src: Path, dst: Path) -> None:
    marker = dst / f".{src.stem}.done"
    if marker.exists():
        print(f"extracted: {src.name}")
        return

    print(f"extracting: {src.name}")
    with ZipFile(src) as zf:
        zf.extractall(dst)
    marker.write_text("ok", encoding="utf-8")


def convert_box(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_label(voc_root: Path, year: str, image_id: str, out_file: Path) -> None:
    xml_path = voc_root / f"VOC{year}" / "Annotations" / f"{image_id}.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        difficult = int(obj.find("difficult").text)
        if cls not in VOC_NAMES or difficult == 1:
            continue
        xmlbox = obj.find("bndbox")
        box = [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")]
        xywh = convert_box((width, height), box)
        lines.append(" ".join(str(v) for v in (VOC_NAMES.index(cls), *xywh)))

    out_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def has_valid_labels(voc_root: Path, year: str, image_id: str) -> bool:
    """检查图像是否包含至少一个非 difficult 的有效标注目标。"""
    xml_path = voc_root / f"VOC{year}" / "Annotations" / f"{image_id}.xml"
    if not xml_path.exists():
        return False
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.iter("object"):
        cls = obj.find("name").text
        difficult = int(obj.find("difficult").text)
        if cls in VOC_NAMES and difficult != 1:
            return True
    return False


def add_haze(image: np.ndarray, beta: float = 1.35, atmosphere: float = 0.82) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    center_x, center_y = w * 0.5, h * 0.58
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    depth = distance / (math.sqrt(center_x**2 + center_y**2) + 1e-6)
    transmission = np.exp(-beta * depth).astype(np.float32)
    transmission = cv2.GaussianBlur(transmission, (0, 0), sigmaX=25, sigmaY=25)
    transmission = transmission[..., None]

    img = image.astype(np.float32) / 255.0
    hazy = img * transmission + atmosphere * (1.0 - transmission)
    return np.clip(hazy * 255.0, 0, 255).astype(np.uint8)


def build_split_multi_beta(
    voc_root: Path,
    out_root: Path,
    year: str,
    split: str,
    image_split: str,
    betas: list,
) -> None:
    """
    为多 beta 生成完整的雾图、清晰图副本和标签。
    （子集 txt 由单独的 generate_subset_txts() 生成）
    """
    ids_file = voc_root / f"VOC{year}" / "ImageSets" / "Main" / f"{image_split}.txt"
    image_ids = ids_file.read_text(encoding="utf-8").strip().split()

    # 过滤无有效标注的图像（学 DR-YOLO）
    valid_ids = [img_id for img_id in image_ids if has_valid_labels(voc_root, year, img_id)]
    filtered = len(image_ids) - len(valid_ids)
    if filtered > 0:
        print(f"[{split}] 过滤了 {filtered} 张无有效标注的图像，剩余 {len(valid_ids)} 张")

    img_out = out_root / "images" / split
    clean_out = out_root / "clean" / split
    label_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    # 为每个 beta 生成雾图 + 清晰图副本 + 标签
    for image_id in tqdm(valid_ids, desc=f"{split}"):
        src_img = voc_root / f"VOC{year}" / "JPEGImages" / f"{image_id}.jpg"
        image = cv2.imread(str(src_img))
        if image is None:
            raise FileNotFoundError(src_img)

        # 先转换一次标签（所有 beta 共用同一份标签）
        label_cache = label_out / f"{image_id}.txt"
        if not label_cache.exists():
            convert_label(voc_root, year, image_id, label_cache)

        for beta in betas:
            suffix = f"_beta{beta:.2f}"
            dst_img = img_out / f"{image_id}{suffix}.jpg"
            dst_clean = clean_out / f"{image_id}{suffix}.jpg"
            dst_label = label_out / f"{image_id}{suffix}.txt"

            if not dst_img.exists():
                hazy = add_haze(image, beta=beta)
                cv2.imwrite(str(dst_img), hazy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if not dst_clean.exists():
                shutil.copy2(src_img, dst_clean)

            if not dst_label.exists():
                shutil.copy2(label_cache, dst_label)


def generate_subset_txts(
    out_root: Path,
    split: str,
    betas: list,
    subset_ratios: list,
    subset_seed: int = 42,
) -> None:
    """
    为指定 split 生成多个比例的确定性子集 txt 文件。

    Args:
        out_root: 数据集输出根目录
        split: 'train' / 'val' / 'test'
        betas: beta 列表
        subset_ratios: 子集比例列表，如 [0.30, 0.50]
        subset_seed: 随机种子（默认 42）
    """
    img_out = out_root / "images" / split
    if not img_out.exists():
        print(f"警告: {img_out} 不存在，跳过子集生成")
        return

    for subset_ratio in subset_ratios:
        if not (0 < subset_ratio < 1.0):
            print(f"警告: 跳过无效子集比例 {subset_ratio}")
            continue

        all_selected = []
        for beta in betas:
            suffix = f"_beta{beta:.2f}.jpg"
            beta_images = sorted([
                "./" + str(f.relative_to(out_root)).replace("\\", "/")
                for f in img_out.iterdir()
                if f.name.endswith(suffix)
            ])
            if not beta_images:
                print(f"警告: beta={beta} 下未找到图像")
                continue

            # 确定性随机: 30% 保持原 seed，其他比例额外加偏移避免冲突
            if abs(subset_ratio - 0.30) < 1e-6:
                seed_offset = 0
            else:
                seed_offset = int(subset_ratio * 1000)
            rng = random.Random(subset_seed + int(beta * 100) + seed_offset)
            k = max(1, int(len(beta_images) * subset_ratio))
            selected = sorted(rng.sample(beta_images, k=k))
            all_selected.extend(selected)

            # 写入单个 beta 的子集 txt（调试用）
            beta_txt = out_root / f"{split}_beta{beta:.2f}_subset{int(subset_ratio*100)}.txt"
            with beta_txt.open("w", encoding="utf-8") as f:
                f.write("\n".join(selected) + "\n")
            print(f"[{split}] beta={beta:.2f} 子集({int(subset_ratio*100)}%)已生成: {beta_txt} ({k} 张)")

        # 合并所有 beta 的子集为总训练集
        all_selected = sorted(set(all_selected))
        subset_txt = out_root / f"{split}_5beta_subset{int(subset_ratio*100)}.txt"
        with subset_txt.open("w", encoding="utf-8") as f:
            f.write("\n".join(all_selected) + "\n")
        print(f"[{split}] 5-beta 合并子集({int(subset_ratio*100)}%)已生成: {subset_txt} ({len(all_selected)} 张)")


def write_yaml(out_root: Path, subset_ratio: float = None, suffix: str = "") -> None:
    """生成数据集 YAML 配置文件。"""
    if subset_ratio and subset_ratio > 0 and subset_ratio < 1.0:
        train_path = f"train_5beta_subset{int(subset_ratio*100)}.txt"
    else:
        train_path = "images/train"

    data = {
        "path": str(out_root).replace("\\", "/"),
        "train": train_path,
        "val": "images/val",      # val 保持全量，用于稳定评估
        "test": "images/test",
        "clean": "clean",
        "names": {i: name for i, name in enumerate(VOC_NAMES)},
    }
    yaml_name = out_root / f"VOC_hazy{suffix}.yaml"
    with yaml_name.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"YAML 已生成: {yaml_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VOC2007_hazy in YOLO detection format.")
    parser.add_argument("--root", default="datasets", help="Dataset parent directory.")
    parser.add_argument("--out-dir", default="VOC_hazy_5beta", help="输出目录名 (默认: VOC_hazy_5beta)")
    parser.add_argument("--beta", type=float, default=None, help="单 beta 模式 (兼容旧版). 若指定则忽略多 beta 列表.")
    parser.add_argument("--betas", nargs="+", type=float, default=None,
                        help=f"多 beta 列表 (默认: {BETAS})")
    parser.add_argument("--subset-ratios", nargs="+", type=float, default=[0.30],
                        help="子集采样比例列表，如 30 50 表示生成 30%% 和 50%% 两个子集 (默认: 30)")
    parser.add_argument("--subset-seed", type=int, default=SUBSET_SEED,
                        help=f"子集采样随机种子 (默认: {SUBSET_SEED})")
    parser.add_argument("--keep-raw-zips", action="store_true", help="Keep downloaded zip files after extraction.")
    args = parser.parse_args()

    # beta 参数解析
    if args.beta is not None:
        betas = [args.beta]
        print(f">>> 单 beta 模式: {betas}")
    elif args.betas is not None:
        betas = args.betas
        print(f">>> 多 beta 模式: {betas}")
    else:
        betas = BETAS
        print(f">>> 默认多 beta 模式: {betas}")

    # 子集比例解析（支持输入 30 50 这样的整数）
    subset_ratios = []
    for r in args.subset_ratios:
        ratio = r / 100.0 if r > 1.0 else r  # 30 -> 0.30
        subset_ratios.append(ratio)
    print(f">>> 子集比例: {[f'{r*100:.0f}%' for r in subset_ratios]}")

    root = Path(args.root).resolve()
    raw = root / "_raw_voc2007"
    out_root = root / args.out_dir
    raw.mkdir(parents=True, exist_ok=True)

    voc_root = raw / "VOCdevkit"
    if not voc_root.exists():
        for name, url in URLS.items():
            zip_path = raw / name
            download_file(url, zip_path)
            extract_zip(zip_path, raw)
            if not args.keep_raw_zips:
                zip_path.unlink(missing_ok=True)
    else:
        print(f">>> 检测到已存在原始 VOC 数据: {voc_root}，跳过下载")

    # 1. 生成完整雾图（所有 beta）
    build_split_multi_beta(voc_root, out_root, "2007", "train", "train", betas)
    build_split_multi_beta(voc_root, out_root, "2007", "val", "val", betas)
    build_split_multi_beta(voc_root, out_root, "2007", "test", "test", betas)

    # 2. 生成指定比例的子集 txt（独立函数，支持多比例）
    generate_subset_txts(out_root, "train", betas, subset_ratios, subset_seed=args.subset_seed)

    # 3. 生成 YAML 配置文件（默认指向 30%，额外生成其他比例的 YAML）
    # 默认 YAML (30%)
    write_yaml(out_root, subset_ratio=0.30, suffix="")
    # 其他比例单独 YAML
    for ratio in subset_ratios:
        if abs(ratio - 0.30) > 1e-6:
            write_yaml(out_root, subset_ratio=ratio, suffix=f"_subset{int(ratio*100)}")

    print(f"\nDone: {out_root}")
    print(f"YAML (默认 30%): {out_root / 'VOC_hazy.yaml'}")
    for ratio in subset_ratios:
        if abs(ratio - 0.30) > 1e-6:
            print(f"YAML ({int(ratio*100)}%): {out_root / f'VOC_hazy_subset{int(ratio*100)}.yaml'}")
    if not args.keep_raw_zips:
        print("Zip files removed after extraction to save disk space.")


if __name__ == "__main__":
    main()
