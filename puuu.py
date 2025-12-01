import os, yaml, sys
cfg = yaml.safe_load(open("plaukti_yolo.yaml"))
p = cfg["path"]
train = os.path.join(p, cfg["train"])
val = os.path.join(p, cfg["val"])
print("train exists:", os.path.isdir(train), " — ", train)
print("val exists:  ", os.path.isdir(val), " — ", val)
print("train files:", len([f for f in os.listdir(train) if f.lower().endswith('.jpg')]) )
print("val files:  ", len([f for f in os.listdir(val) if f.lower().endswith('.jpg')]) )