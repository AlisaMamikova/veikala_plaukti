import json, argparse
from pathlib import Path
from PIL import Image

def load_odgt_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)

def box_to_yolo(x, y, w, h, W, H):
    cx = (x + w/2) / W
    cy = (y + h/2) / H
    return cx, cy, w / W, h / H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--odgt', required=True)
    ap.add_argument('--images_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', choices=['train','val'], required=True)
    ap.add_argument('--box-type', choices=['fbox','hbox'], default='fbox')
    args = ap.parse_args()

    images_root = Path(args.images_root)
    out_images = Path(args.out) / 'images' / args.split
    out_labels = Path(args.out) / 'labels' / args.split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0
    for rec in load_odgt_lines(args.odgt):
        img_rel = rec['ID']
        img_path = Path.joinpath(images_root, f"{img_rel}.jpg") 
        if not img_path.exists():
            skipped += 1
            continue
        try:
            W, H = Image.open(img_path).size
        except Exception:
            skipped += 1
            continue

        stem = img_path.stem
        img_out = out_images / img_path.name
        lbl_out = out_labels / f"{stem}.txt"

        if not img_out.exists():
            with open(img_path, 'rb') as fi, open(img_out, 'wb') as fo:
                fo.write(fi.read())

        lines = []
        for g in rec.get('gtboxes', []):
            tag = g.get('tag', 'person')
            if tag == 'mask':
                continue
            extra = g.get('extra', {})
            head_attr = g.get('head_attr', {})
            if extra.get('ignore', 0) == 1:
                continue
            if args.box_type == 'hbox' and head_attr.get('ignore', 0) == 1:
                continue
            bbox = g.get(args.box_type, None)
            if not bbox:
                continue
            x, y, w, h = bbox
            x = max(0.0, min(float(x), W - 1))
            y = max(0.0, min(float(y), H - 1))
            w = max(1.0, min(float(w), W - x))
            h = max(1.0, min(float(h), H - y))
            cx, cy, nw, nh = box_to_yolo(x, y, w, h, W, H)
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(lbl_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        kept += 1

    print(f"Converted: {kept}, skipped: {skipped}")
    print(f"Images -> {out_images}")
    print(f"Labels -> {out_labels}")

if __name__ == '__main__':
    main()
