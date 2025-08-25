🔍 Change Detection · 🧭 Space Item Detection · 🧪 Space Classification
TinyChangeUNet · YOLOv8 · PyTorch/timm
<p align="center"> <img src="https://img.shields.io/badge/python-3.10%2B-1f6feb"> <img src="https://img.shields.io/badge/pytorch-2.x-EE4C2C"> <img src="https://img.shields.io/badge/timm-MobileNetV3-ffc107"> <img src="https://img.shields.io/badge/ultralytics-YOLOv8-00b894"> <img src="https://img.shields.io/badge/repro-seed%3D42-8957e5"> </p> <p align="center"> 실내 공간 데이터를 대상으로 <b>변화 감지</b>, <b>물품 탐지</b>, <b>분류</b>를 아우르는 파이프라인입니다. </p> <p align="center"> <a href="#-10초-퀵스타트">🚀 빠른 시작</a> · <a href="#-폴더-구조">📁 폴더 구조</a> · <a href="#모듈-a--change-detection">A. Change Detection</a> · <a href="#모듈-b--space-item-detection">B. Item Detection</a> · <a href="#모듈-c--space-classification">C. Classification</a> </p>
## ✨ 무엇이 들어있나요?

- **합성 Change 데이터 구축**: 가림/블러/픽셀화/인페인트/이동으로 after + **GT(0/255)** 자동 생성
- **경량 TinyChangeUNet**: `before(3)+after(3)+diff(1)=7ch → 1×1 conv → MobileNetV3 encoder → 얕은 decoder`
- **안정 학습 루프**: AMP(FP16), Cosine+Warmup, **EMA** 검증/저장, `pos_weight` 자동 추정
- **평가 루틴**: 검증 **threshold sweep**으로 최적 `th` 선택 → 테스트 **mIoU/F1** & PNG 저장
- **YOLOv8 탐지**: 공간 내 물품(에어컨/거울/피아노 등) 커스텀 탐지
- **분류 스크립트**: `timm` 백본으로 공간/물품 **단일 이미지 분류**

🚀 10초 퀵스타트
pip install torch torchvision timm ultralytics opencv-python numpy matplotlib tqdm scikit-learn
노트북: change_detection.ipynb, space_item_detection.ipynb · 분류 스크립트: space_classification.py
📁 폴더 구조
# A) Change Detection (합성 데이터)
pairs_out_cd/
  train/{before_images, after_images, labels}
  val/{before_images, after_images, labels}
  test/{before_images, after_images, labels}
meta/pairs_{train,val}.json   # 통계/경로 메타

# B) Space Item Detection (YOLO 형식)
space_data/
  images/{train,val,test}/*.jpg|png
  labels/{train,val,test}/*.txt      # YOLO: cls cx cy w h
  space.yaml                         # 데이터 설정

# C) Space Classification (폴더-클래스)
space_cls/
  train/<class>/*.jpg|png
  val/<class>/*.jpg|png
  test/<class>/*.jpg|png
마스크는 0/255 바이너리 PNG, 리사이즈 시 nearest 권장.
모듈 A — Change Detection
노트북: change_detection.ipynb · 모델: TinyChangeUNet(MobileNetV3 Small) · 입력: 7채널(before+after+diff)
flowchart LR
  A[Before (3ch)] ---|concat| R[Reduce 1x1 Conv → 3ch]
  B[After (3ch)]  ---|concat| R
  D[abs(Before-After) (1ch)] ---|concat| R
  R --> E[MobileNetV3 Encoder]
  E -->|multi-scale| Dec[Tiny Decoder]
  Dec --> H[Head 1x1 Conv]
  H --> M[Sigmoid → Binary Mask (0/255)]
사용 절차
PairDataset2In 정의 → 2) TinyChangeUNet 정의 → 3) Train/Eval 실행(IMG_SIZE=256, BATCH=8, LR=3e-4, EPOCHS=40)
Self-contained TEST Eval: 재시작 후 체크포인트 로드 → 검증 sweep → 테스트 mIoU/F1 + ./test_preds/*.png
결과 카드(예시)
Split	Best-th	mIoU	F1	AMP	EMA
Val	0.18	0.41	0.51	✅	✅
Test	0.18	0.40	0.50	✅	✅
<details> <summary>🎨 오버레이(After 위 반투명) 코드</summary>
def overlay(rgb, mask, color=(0,255,255), alpha=0.35):
    import numpy as np
    m = (mask > 127).astype(np.uint8)
    tint = np.ones_like(rgb, dtype=np.uint8)*np.array(color, dtype=np.uint8)
    over = (rgb*(1-alpha) + tint*alpha).astype(np.uint8)
    out = rgb.copy(); out[m>0] = over[m>0]
    return out
</details>
모듈 B — Space Item Detection
노트북: space_item_detection.ipynb · 모델: YOLOv8(사전학습 → 커스텀 파인튜닝)
데이터 설정(space.yaml)
path: ./space_data
train: images/train
val: images/val
test: images/test
names:
  0: air_conditioner
  1: mirror
  2: piano
  # ...
학습/평가
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="space_data/space.yaml", imgsz=640, epochs=100, batch=16, seed=42)
model.val(data="space_data/space.yaml", imgsz=640, split="val")
model.predict(source="space_data/images/test", imgsz=640, conf=0.25, save=True)
모듈 C — Space Classification
스크립트: space_classification.py · 목적: 공간/물품 단일 이미지 분류
예시 실행
# 학습
python space_classification.py \
  --data space_cls --img-size 224 --batch-size 32 --epochs 50 \
  --model resnet18 --lr 3e-4 --seed 42 --save ckpt_space_cls.pt

# 평가
python space_classification.py \
  --data space_cls --img-size 224 --batch-size 32 \
  --weights ckpt_space_cls.pt --eval
권장: timm 백본 전이학습, 불균형 시 class_weight/WeightedRandomSampler, Confusion Matrix 로그
🔁 재현성 & 라이선스
공통 시드: 42 · EMA 가중치 기준 검증/저장
강한 결정론 옵션: torch.use_deterministic_algorithms(True), cudnn.benchmark=False
라이선스: 루트의 LICENSE, DATA_LICENSE 참고 (민감 데이터 비의도 사용 금지 권장)
