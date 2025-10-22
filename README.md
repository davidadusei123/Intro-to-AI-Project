# YOLOv8 on Kaggle **PCB Defect dataset** (norbertelter/pcb-defect-dataset)

This starter kit trains a YOLOv8 model (Ultralytics) on the Kaggle PCB Defect dataset:
- Dataset: https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset
- Classes (6): `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`

> According to the Kaggle page, the dataset already uses **YOLO annotations** (TXT).  
> You can train right away once the images/labels are arranged into `train/`, `val/`, and optionally `test/` folders.

## Quick Start

### 0) Create and activate a clean environment (recommended)
```bash
# one example with conda; use any environment tool you like
conda create -n pcb-yolov8 python=3.10 -y
conda activate pcb-yolov8
```

### 1) Install dependencies
```bash
pip install ultralytics==8.* kaggle==1.* scikit-learn==1.* opencv-python tqdm
```

### 2) Authenticate Kaggle (first time only)
- Put your Kaggle API token file at `~/.kaggle/kaggle.json`  
  (Create/download from https://www.kaggle.com/settings/account)
- Or set env vars:
```bash
export KAGGLE_USERNAME=<your_username>
export KAGGLE_KEY=<your_key>
```

### 3) Run the end-to-end script
```bash
python train_pcb_yolov8.py \
  --model yolov8n.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch 16 \
  --conf 0.001 \
  --device 0
```

This will:
1. Download and extract the dataset from Kaggle into `data/raw/pcb-defect-dataset`.
2. Auto-detect `images/` and `labels/` layout; if not already split, it will **split** into `data/pcb/splits/{train,val,test}`.
3. Generate a `pcb.yaml` pointing to those splits.
4. Launch Ultralytics training, validate, and export to ONNX.
5. Save results under `runs/detect/pcb_yolov8*`.

### 4) Inference examples
After training, you can run:
```bash
# Single image
yolo task=detect mode=predict model=runs/detect/pcb_yolov8n/weights/best.pt source=some.jpg imgsz=640

# Or via Python (see bottom of train_pcb_yolov8.py)
```

### 5) Tips
- Start with `yolov8n.pt` (nano) for quick experiments, then try `yolov8s.pt` / `yolov8m.pt` for higher accuracy.
- Keep `imgsz=640` initially (dataset was often prepared around 600–640). Later tune to 640–1024 if VRAM allows.
- Monitor training with `tensorboard --logdir runs` if you have TensorBoard installed.

---

## Repo Layout (after running)
```
pcb_yolov8_starter/
  pcb.yaml
  train_pcb_yolov8.py
  data/
    raw/pcb-defect-dataset/        # raw Kaggle extraction
    pcb/splits/                     # auto-created YOLO structure if needed
      images/train/ ...             # symlinks or copies
      images/val/ ...
      images/test/ ...
      labels/train/ ...
      labels/val/ ...
      labels/test/ ...
  runs/                              # Ultralytics outputs
```

## Data config (`pcb.yaml`)
This points YOLOv8 to your `train/val/test` folders and defines the class names (order matters).
You generally won't need to edit it unless your paths change.

---

## Troubleshooting
- **Kaggle auth error**: ensure `~/.kaggle/kaggle.json` exists with correct permissions (`chmod 600`) or env vars are set.
- **No labels found**: make sure the dataset folders actually contain `labels/*.txt`. If the dataset root has a different structure,
  update `--data_root` or the auto-detection logic in the script.
- **CUDA errors**: lower `--batch`, use `--device cpu`, or try `yolov8n.pt`.
- **Class order mismatch**: confirm your labels use the same class indices as `names` in `pcb.yaml`.

Happy training!