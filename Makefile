.PHONY: setup train eval app docker-build docker-run

setup:
	python -m pip install -r requirements.txt

train:
	python scripts/train.py --data plaukti_yolo.yaml --epochs 50 --imgsz 800 --batch 16

eval:
	python scripts/eval.py --weights runs/detect/train/weights/best.pt --data plaukti_yolo.yaml --imgsz 800 --conf 0.25

app:
	streamlit run app.py --server.port=8501 --server.address=0.0.0.0

docker-build:
	docker build -t $(shell basename $(PWD)):latest .

docker-run:
	docker run --rm -it -p 8501:8501 -v $(PWD)/data:/app/data -v $(PWD)/runs:/app/runs $(shell basename $(PWD)):latest
