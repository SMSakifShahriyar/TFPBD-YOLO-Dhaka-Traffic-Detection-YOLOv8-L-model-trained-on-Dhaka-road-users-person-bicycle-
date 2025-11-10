from ultralytics import YOLO

def main():
    model = YOLO("yolov8l.pt")  # pretrained YOLOv8-L

    model.train(
        data="data.yaml",         
        imgsz=896,
        batch=16,             
        device=0,                 
        workers=2,              

        cache="disk",
        rect=True,
        mosaic=0,
        mixup=0,
        copy_paste=0,
        plots=False,

        epochs=60,
        patience=20,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        cos_lr=True,

        project="runs",
        name="tfpbd_yolov8l_896_b16_local",
    )

if __name__ == "__main__":
    main()
