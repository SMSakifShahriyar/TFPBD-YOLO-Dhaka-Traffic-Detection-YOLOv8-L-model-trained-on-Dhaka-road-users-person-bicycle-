from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(r"D:\yolo_exports\best_v8l_896.pt")  

SOURCE = Path(r"D:\TFPBD_YOLO\images\val") 

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Update MODEL_PATH in this script.")

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(SOURCE),
        imgsz=896,
        conf=0.25,
        save=True,
        project="runs",
        name="tfpbd_v8l_896_infer",
    )

    print(f"Saved predictions to: {Path('runs') / 'tfpbd_v8l_896_infer'}")

if __name__ == "__main__":
    main()
