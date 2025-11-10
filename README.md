
```markdown
# TFPBD YOLO – Traffic Detection in Dhaka

Custom YOLOv8-L model trained to detect common Dhaka road users:
**person, bicycle, rickshaw, cng, car, van, bus, truck**.

The goal is a practical detector that understands local vehicle types (e.g., rickshaw, CNG) in busy street scenes.



## Why this project?

Most open datasets don’t cover Dhaka-specific classes well.  
This project packages a reproducible YOLOv8 pipeline for **training, validation, inference, and export**, plus a ready-to-use model trained on real Dhaka traffic data.

**Weights:** `best_v8l_896.pt` (add via GitHub Release)  
**Dataset:** [Mendeley Traffic Dataset](https://data.mendeley.com/datasets/h8bfgtdp2r/4)



## Dataset & config

The dataset follows the standard YOLO structure:

```

TFPBD_YOLO/
├─ images/
│  ├─ train/  ├─ val/  └─ test/
└─ labels/
├─ train/  ├─ val/  └─ test/

````

Example `data.yaml`:
```yaml
train: /workspace/TFPBD_YOLO/images/train
val:   /workspace/TFPBD_YOLO/images/val
test:  /workspace/TFPBD_YOLO/images/test
names:
  0: person
  1: bicycle
  2: rickshaw
  3: cng
  4: car
  5: van
  6: bus
  7: truck
````

---

## Setup

```bash
# Python 3.9–3.11 recommended
pip install --upgrade ultralytics opencv-python numpy
```

If using GPU, make sure your PyTorch version matches CUDA.

---

## Training

Final training command used:

```bash
yolo detect train \
  model=yolov8l.pt data=/workspace/TFPBD_YOLO/data.yaml \
  imgsz=896 batch=16 device=0 workers=6 \
  cache=disk rect=True mosaic=0 mixup=0 copy_paste=0 plots=False \
  epochs=60 patience=20 optimizer=AdamW lr0=0.002 lrf=0.01 weight_decay=0.0005 cos_lr=True \
  project=/workspace/runs name=tfpbd_yolov8l_896_b16_runpod
```

---

## Results (Validation, imgsz=896)

Overall:

* **mAP@50:** 0.615
* **mAP@50–95:** 0.374
* **Precision:** 0.580
* **Recall:** 0.582

Per-class AP@50:

| Class    | AP@50 |
| -------- | ----- |
| person   | 0.661 |
| bicycle  | 0.738 |
| rickshaw | 0.715 |
| cng      | 0.791 |
| car      | 0.868 |
| van      | 0.306 |
| bus      | 0.773 |
| truck    | 0.072 |

> Metrics are from validating the final `best.pt` on the validation split (2,400 images, 30,191 objects).

---

## Inference

Run prediction on images or videos:

```bash
python infer_tfpbd_yolo.py \
  --weights best_v8l_896.pt \
  --source path/to/images_or_video
```

Or directly with the YOLO CLI:

```bash
yolo detect predict model=best_v8l_896.pt source=path/to/images imgsz=896
```

Predictions are saved under `runs/detect/predict*`.

---

## Re-evaluate the model

To recompute metrics on your validation or test split:

```bash
yolo detect val \
  model=best_v8l_896.pt \
  data=data.yaml \
  imgsz=896 \
  project=runs name=val_v8l_896
```

Results (mAP, PR curve, confusion matrix) are stored in:

```
runs/detect/val_v8l_896/
```

---

