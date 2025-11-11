# CrowdHuman YOLOv8 — Detection-based Crowd Counting

This project converts CrowdHuman `.odgt` annotations into YOLO format and trains YOLOv8.
You can choose to use **full-body** (`fbox`) or **head** (`hbox`) boxes.
Counting is simply the number of detections per image.

## Steps
1) **Convert** CrowdHuman to YOLO
```bash
python scripts/convert_crowdhuman_to_yolo.py   --odgt /path/annotation_train.odgt   --images-root /path/CrowdHuman_train/Images   --out data/crowdhuman_yolo --split train --box-type fbox

python scripts/convert_crowdhuman_to_yolo.py   --odgt /path/annotation_val.odgt   --images-root /path/CrowdHuman_val/Images   --out data/crowdhuman_yolo --split val --box-type fbox
```

2) **Train**
```bash
pip install -r requirements.txt
python train.py --data crowdhuman_yolo.yaml --epochs 50 --imgsz 800 --batch 16
```

3) **Evaluate**
```bash
python scripts/eval.py --weights runs/detect/train/weights/best.pt --data crowdhuman_yolo.yaml --conf 0.25
```

4) **Streamlit demo**
```bash
streamlit run app.py
```

## Docker
```bash
make docker-build
make docker-run  # open http://localhost:8501
```
