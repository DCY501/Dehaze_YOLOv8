from ultralytics import YOLO


def main() -> None:
    # Use "configs/yolov8-dehaze.yaml" here if you want to train from the custom model structure.
    model = YOLO("yolov8n.pt")

    model.train(
        data="configs/VOC_hazy.yaml",
        imgsz=512,
        epochs=50,
        batch=8,
        device=0,
        workers=4,
        project="runs",
        name="voc_hazy_experiment",
        exist_ok=True,
        lr0=0.001,
        augment=True,
    )


if __name__ == "__main__":
    main()
