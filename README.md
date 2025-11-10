# TFPBD YOLO – Traffic Detection in Dhaka

Custom YOLOv8-L model trained to detect common Dhaka road users:
**person, bicycle, rickshaw, cng, car, van, bus, truck**.

The goal is a practical detector that understands local vehicle types (e.g., rickshaw, CNG) in busy street scenes.

---

## Why this project?

Most open datasets don’t cover Dhaka-specific classes well.  
This project packages a reproducible YOLOv8 pipeline for **training, validation, inference, and export**, plus a ready-to-use model trained on real Dhaka traffic data.

**Weights:** `best_v8l_896.pt` (add via GitHub Release)  
**Dataset:** [Mendeley Traffic Dataset](https://data.mendeley.com/datasets/h8bfgtdp2r/4)

---

## Dataset & config

The dataset follows the standard YOLO structure:

```text
TFPBD_YOLO/
├─ images/
│  ├─ train/  ├─ val/  └─ test/
└─ labels/
   ├─ train/  ├─ val/  └─ test/
