```markdown
# TFPBD YOLO – Traffic Detection in Dhaka

This project uses a YOLOv8 **object detector** on Dhaka road scenes to detect 8 road-user classes:

- person, bicycle, rickshaw, CNG, car, van, bus, truck

The model was originally trained on a rented GPU (RTX 4090), and the final weights were downloaded to Windows as `best_v8l_896.pt`.  
This repo keeps the **code and configuration**, but not the big dataset or weights.

---

## Dataset config

On my machine, the dataset lives at `D:\TFPBD_YOLO`.  
The YOLO `data.yaml` used during training is:

```yaml
train: D:/TFPBD_YOLO/images/train
val:   D:/TFPBD_YOLO/images/val
test:  D:/TFPBD_YOLO/images/test
names:
  0: person
  1: bicycle
  2: rickshaw
  3: cng
  4: car
  5: van
  6: bus
  7: truck