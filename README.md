🔎 Change Detection & 🧭 Space Item Detection & 🧪 Space Classification
(TinyChangeUNet · YOLOv8 · PyTorch/timm)
<p align="center"> <img src="https://img.shields.io/badge/python-3.10%2B-1f6feb"> <img src="https://img.shields.io/badge/pytorch-2.x-EE4C2C"> <img src="https://img.shields.io/badge/timm-MobileNetV3-ffc107"> <img src="https://img.shields.io/badge/ultralytics-YOLOv8-00b894"> <img src="https://img.shields.io/badge/repro-seed%3D42-8957e5"> </p>

본 저장소는 실내 공간 데이터를 대상으로 하는 세 가지 작업을 포함합니다.
Change Detection — YOLO 라벨 박스로 합성한 before/after 쌍 + 변경 GT(0/255) 생성, 경량 TinyChangeUNet(MobileNetV3 encoder) 학습/평가
Space Item Detection — 실내 사진에서 에어컨/거울/피아노 등 YOLOv8 탐지
Space Classification — 공간/물품의 단일 이미지 분류(PyTorch/timm 기반, 스크립트: space_classification.py)

목차
환경 준비
폴더 구조
모듈 A — Change Detection
모듈 B — Space Item Detection
모듈 C — Space Classification
재현성 & 라이선스
문의
환경 준비

# 공통 의존성 (로컬/콜랩 공통)
pip install torch torchvision timm ultralytics opencv-python numpy matplotlib tqdm scikit-learn
노트북 파일: change_detection.ipynb, space_item_detection.ipynb
분류 스크립트: space_classification.py
폴더 구조
# A) Change Detection (합성 데이터)
pairs_out_cd/
  train/{before_images, after_images, labels}
  val/{before_images, after_images, labels}
  test/{before_images, after_images, labels}
meta/pairs_{train,val}.json

# B) Space Item Detection (YOLO 형식)
space_data/
  images/{train,val,test}/*.jpg|png
  labels/{train,val,test}/*.txt        # YOLO txt (cls cx cy w h)
  space.yaml                           # 데이터 구성/클래스 정의

# C) Space Classification (클래스별 폴더 구조)
space_cls/
  train/<class_name>/*.jpg|png
  val/<class_name>/*.jpg|png
  test/<class_name>/*.jpg|png
마스크는 0/255 바이너리 PNG (리사이즈 시 nearest 권장).
YOLO 라벨은 cls cx cy w h 정규화 형식.
모듈 A — Change Detection
노트북: change_detection.ipynb
모델: TinyChangeUNet (MobileNetV3 Small encoder)
입력: before(3)·after(3)·diff(1) → 7채널
✨ 핵심
합성 데이터 구축: 가림/블러/픽셀화/인페인트/이동으로 after + GT(0/255) 생성
모델: 7ch → 1×1 conv 축소 → MobileNetV3 encoder → 경량 decoder → 1ch logit
학습 루프: AMP(FP16), Cosine+Warmup, EMA 검증/저장, pos_weight 자동 추정
평가: 검증 threshold sweep으로 최적 th → 테스트 mIoU/F1 + PNG 저장
flowchart LR
    A[Before (3ch)] ---|concat| R[Reduce 1x1 Conv → 3ch]
    B[After (3ch)]  ---|concat| R
    D[abs(Before-After) (1ch)] ---|concat| R
    R --> E[MobileNetV3 Encoder]
    E -->|multi-scale| Dec[Tiny Decoder]
    Dec --> H[Head 1x1 Conv]
    H --> M[Logit → Sigmoid → Binary Mask (0/255)]
퀵스타트
PairDataset2In 정의 → 2) TinyChangeUNet 정의 → 3) Train/Eval 루프 실행(IMG_SIZE=256, BATCH=8, LR=3e-4, EPOCHS=40) → 4) Self-contained TEST Eval로 재시작 후 평가
결과 카드(예시)
Split	Best-th	mIoU	F1	AMP	EMA
Val	0.18	0.41	0.51	✅	✅
Test	0.18	0.40	0.50	✅	✅
모듈 B — Space Item Detection
노트북: space_item_detection.ipynb
모델: YOLOv8 (COCO 사전학습 → 커스텀 클래스 파인튜닝)
데이터 설정 (space.yaml 예시)
path: ./space_data
train: images/train
val: images/val
test: images/test

names:
  0: air_conditioner
  1: mirror
  2: piano
  # ...
학습/검증/테스트
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # n/s/m/l/x 중 선택
model.train(data="space_data/space.yaml", imgsz=640, epochs=100, batch=16, seed=42)
model.val(data="space_data/space.yaml", imgsz=640, split="val")
model.predict(source="space_data/images/test", imgsz=640, save=True, conf=0.25)
결과 카드(예시)
Split	mAP@0.5	mAP@0.5:0.95	이미지크기	모델
Val	0.xx	0.xx	640	YOLOv8n
Test	0.xx	0.xx	640	YOLOv8n
모듈 C — Space Classification
스크립트: space_classification.py
목적: 공간/물품 이미지를 클래스 단위로 분류(예: living_room, study_room, kitchen 또는 air_conditioner, mirror, …)
데이터 포맷
클래스별 폴더 구조(이미 위 “폴더 구조” 참고):
space_cls/
  train/<class_name>/*.jpg|png
  val/<class_name>/*.jpg|png
  test/<class_name>/*.jpg|png
예시 실행
※ 스크립트 인자명은 구현에 따라 다를 수 있습니다. python space_classification.py -h 로 확인하세요.
# 학습
python space_classification.py \
  --data space_cls \
  --img-size 224 \
  --batch-size 32 \
  --epochs 50 \
  --model resnet18 \
  --lr 3e-4 \
  --seed 42 \
  --save ckpt_space_cls.pt

# 평가(가중치 로드 후 test 성능 리포트)
python space_classification.py \
  --data space_cls \
  --img-size 224 \
  --batch-size 32 \
  --weights ckpt_space_cls.pt \
  --eval
권장 사항
전이학습: timm 사전학습 백본(resnet/efficientnet/mobilevit 등) 사용 시 수렴/정확도 유리
클래스 불균형: WeightedRandomSampler 또는 class_weight(loss)에 반영
로깅: Top-1 Acc, macro F1, Confusion Matrix 저장
결과 카드(예시)
Split	Top-1 Acc	Macro F1	ImgSize	Backbone
Val	0.xx	0.xx	224	resnet18
Test	0.xx	0.xx	224	resnet18
<details> <summary><b>Confusion Matrix 시각화(예시)</b></summary>
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
plt.tight_layout(); plt.savefig("confusion_matrix.png")
</details>
