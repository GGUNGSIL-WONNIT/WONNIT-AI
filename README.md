🔍 Change Detection · 🧭 Space Item Detection · 🧪 Space Classification
TinyChangeUNet · YOLOv8 · PyTorch/timm
<p align="center"> <img src="https://img.shields.io/badge/python-3.10%2B-1f6feb"> <img src="https://img.shields.io/badge/pytorch-2.x-EE4C2C"> <img src="https://img.shields.io/badge/timm-MobileNetV3-ffc107"> <img src="https://img.shields.io/badge/ultralytics-YOLOv8-00b894"> <img src="https://img.shields.io/badge/repro-seed%3D42-8957e5"> </p> <p align="center"> 실내 공간 데이터를 대상으로 <b>변화 감지</b>, <b>물품 탐지</b>, <b>분류</b>를 아우르는 파이프라인입니다. </p> <p align="center"> <a href="#-폴더-구조">📁 폴더 구조</a> · <a href="#모듈-a--change-detection">A. Change Detection</a> · <a href="#모듈-b--space-item-detection">B. Item Detection</a> · <a href="#모듈-c--space-classification">C. Classification</a> </p>

---

## 무엇이 들어있나요?

- 합성 Change 데이터 구축: 가림/블러/픽셀화/인페인트/이동으로 after + GT(0/255) 자동 생성
- 경량 TinyChangeUNet: before(3) + after(3) + diff(1) = 7ch → 1x1 conv → MobileNetV3 encoder → 얕은 decoder
- 안정 학습 루프: AMP(FP16), Cosine+Warmup, EMA 검증/저장, pos_weight 자동 추정
- 평가 루틴: 검증 threshold sweep으로 최적 th 선택 → 테스트 mIoU/F1 및 예측 PNG 저장
- YOLOv8 탐지: 공간 내 물품(에어컨, 거울, 피아노 등) 커스텀 탐지
- 분류 스크립트: timm 백본으로 공간/물품 단일 이미지 분류

---
## 📁 폴더 구조
A) Change Detection (합성 데이터)
pairs_out_cd/
  train/{before_images, after_images, labels}
  val/{before_images, after_images, labels}
  test/{before_images, after_images, labels}
meta/pairs_{train,val}.json

B) Space Item Detection (YOLO 형식)
space_data/
  images/{train,val,test}/*.jpg|png
  labels/{train,val,test}/*.txt       # YOLO: cls cx cy w h
  space.yaml

C) Space Classification (클래스별 폴더)
space_cls/
  train/<class>/*.jpg|png
  val/<class>/*.jpg|png
  test/<class>/*.jpg|png

## A) Change Detection (합성 데이터)
데이터: before/after/label 쌍 (label은 0/255 변경 마스크)
모델: TinyChangeUNet (입력 7ch = before 3 + after 3 + diff 1)
학습: AMP, Cosine+Warmup, EMA 저장, pos_weight 자동 추정
평가: 검증 threshold sweep → best th로 테스트 mIoU/F1, ./test_preds/*.png 저장
파이프라인 다이어그램(머메이드 지원 환경에서 표시됨):
flowchart LR
  Bf[Before (3ch)] ---|concat| R[Reduce 1x1 Conv -> 3ch]
  Af[After  (3ch)] ---|concat| R
  Df[abs(Before-After) (1ch)] ---|concat| R
  R --> E[MobileNetV3 Encoder]
  E --> D[Light Decoder]
  D --> H[Head 1x1 Conv]
  H --> M[Sigmoid -> Binary Mask (0/255)]


## B) Space Item Detection (YOLO 형식)
모델: YOLOv8(사전학습 → 커스텀 파인튜닝)
데이터: space_data/(YOLO txt: cls cx cy w h)
출력: val mAP, test 예측 이미지 저장
# space.yaml (예시)
path: ./space_data
train: images/train
val: images/val
test: images/test
names:
  0: air_conditioner
  1: mirror
  2: piano


##C) Space Classification (폴더-클래스)
모델: timm 백본(resnet/efficientnet 등) 전이학습
데이터: space_cls/{train,val,test}/<class>/*.jpg|png
지표: Top-1 Acc, macro F1, Confusion Matrix
# 학습
python space_classification.py \
  --data space_cls --img-size 224 --batch-size 32 --epochs 50 \
  --model resnet18 --lr 3e-4 --seed 42 --save ckpt_space_cls.pt

# 평가
python space_classification.py \
  --data space_cls --img-size 224 --batch-size 32 \
  --weights ckpt_space_cls.pt --eval
