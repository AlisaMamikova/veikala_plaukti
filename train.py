import argparse, os
from ultralytics import YOLO

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def pick_device(d):
    if d: 
        return d
    try:
        import torch
        if torch.cuda.is_available(): 
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): 
            return "mps"
    except Exception:
        pass
    return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="crowdhuman_yolo.yaml")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print("Using device:", device)

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device)

if __name__ == "__main__":
    main()
