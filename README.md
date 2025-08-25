# 🏷️ Scene Classification · 🧭 Space Item Detection · 🔍 Change Detection
MobileNetV2 (torchvision) · YOLOv8n · TinyChangeUNet (MobileNetV3 encoder)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-1f6feb">
  <img src="https://img.shields.io/badge/pytorch-2.x-EE4C2C">
  <img src="https://img.shields.io/badge/classifier-MobileNetV2(torchvision)-4ea1ff">
  <img src="https://img.shields.io/badge/detector-YOLOv8n-00b894">
  <img src="https://img.shields.io/badge/change-TinyChangeUNet(MobileNetV3)-ffd24d">
  <img src="https://img.shields.io/badge/repro-seed%3D42-8957e5">
</p>

<p align="center">
  실내 공간 데이터를 대상으로 <b>장소 분류</b> → <b>공간 아이템 탐지</b> → <b>변화 감지</b>를 하나의 경량 파이프라인으로 제공합니다.
</p>

---

- [폴더 구조](#-폴더-구조)
- [A) Scene Classification (장소 분류)](#a-scene-classification-장소-분류)
- [B) Space Item Detection (아이템 탐지)](#b-space-item-detection-아이템-탐지)
- [C) Change Detection (전후 변화 감지)](#c-change-detection-전후-변화-감지)
- [재현성](#-재현성)

---

## 📁 폴더 구조

```text
A) Scene Classification (ImageFolder)
space_cls/
  train/<class>/*.jpg|png
  val/<class>/*.jpg|png
  test/<class>/*.jpg|png

B) Space Item Detection (YOLO)
space_data/
  images/{train,val,test}/*.jpg|png
  labels/{train,val,test}/*.txt      # YOLO: cls cx cy w h
  space.yaml

C) Change Detection (합성 before/after/GT)
pairs_out_cd/
  train/{before_images,after_images,labels}
  val/{before_images,after_images,labels}
  test/{before_images,after_images,labels}
meta/pairs_{train,val}.json
```

---

## A) Scene Classification (장소 분류)
    
**모델**: MobileNetV2 *(torchvision, ImageNet 사전학습 → 파인튜닝)*  
**입력**: 224×224  
**타깃 클래스(5)**: `creative_studio`, `dance_studio`, `music_rehearsal_room`, `small_theater_gallery`, `study_room`  
**데이터 구축**: 클래스당 ~50장 *(1차 크롤링 → 2차 수작업 정제)* 후 **train:val=8:2** 분할, ImageFolder 포맷

```bash
# 학습 스크립트
python train_mobilenet.py
# 산출물: mobilenetv2.pth, class_names.txt
```

**테스트 성능(클래스별 정확도)**  
| Class | Correct / Total | Acc. |
|---|---:|---:|
| creative_studio | 10 / 10 | **100.0%** |
| dance_studio | 8 / 10 | **80.0%** |
| music_rehearsal_room | 7 / 10 | **70.0%** |
| small_theater_gallery | 8 / 10 | **80.0%** |
| study_room | 10 / 10 | **100.0%** |

<img width="1290" height="998" alt="matrix" src="https://github.com/user-attachments/assets/31a65d1a-5148-461b-a993-5ef87e921895" />

**Overall Acc**: 43/50 = **86.0%** · **Macro Acc**: **86.0%**  
> 주요 오분류: `dance_studio → small_theater_gallery` 2건, `music_rehearsal_room → study_room` 2건 등.



---
## B) Space Item Detection (아이템 탐지)

**모델**: YOLOv8n *(Ultralytics, COCO 사전학습 → 커스텀 파인튜닝)*  
**목적**: 실내 사진에서 공간 아이템(13종)을 탐지합니다.

### 데이터셋
- **클래스(13종)**  
  `air_conditioner, chair, desk, drum, microphone, mirror, monitor, piano, projector, speaker, spotlight, stage, whiteboard`
- **구축**: 장소 데이터에서 **자동 박스 라벨링** 파이프라인으로 초안 생성 → **수작업 보정**  
- **형식**: YOLO 형식 (`images/`, `labels/*.txt` ; 각 txt: `cls cx cy w h`)

```text
라벨 통계 (히스토그램)
[dataset_trainval]
  0 air_conditioner : 1026   7 piano       : 1265
  1 chair           : 2058   8 projector   : 1026
  2 desk            : 4199   9 speaker     : 1900
  3 drum            : 3444  10 spotlight   : 6136
  4 microphone      : 1126  11 stage       : 1012
  5 mirror          : 1185  12 whiteboard  : 1504
  6 monitor         : 1391

[dataset_test]
  0 air_conditioner :   7   7 piano       :  22
  1 chair           :  94   8 projector   :  10
  2 desk            :  69   9 speaker     :  22
  3 drum            :  13  10 spotlight   : 142
  4 microphone      :  15  11 stage       :   8
  5 mirror          :  44  12 whiteboard  :  13
  6 monitor         :  14
```

**YOLO 데이터 설정 예시(`space.yaml`)**
```yaml
path: ./space_data
train: images/train
val: images/val
test: images/test
names:
  0: air_conditioner
  1: chair
  2: desk
  3: drum
  4: microphone
  5: mirror
  6: monitor
  7: piano
  8: projector
  9: speaker
 10: spotlight
 11: stage
 12: whiteboard
```

**학습/평가/추론**
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="space_data/space.yaml", imgsz=640, epochs=80, batch=16, seed=42)
model.val(data="space_data/space.yaml", imgsz=640, split="val")
model.predict(source="space_data/images/test", imgsz=640, conf=0.25, save=True)
```

**성능 요약**  
- 80 epochs 완료, best/last 가중치 저장 (`runs/detect/space_no_leak/weights/best.pt`)  
- **Val**: `mAP50=0.981`, `mAP50-95=0.912`  
- **Test**: `mAP50=0.979`, `mAP50-95=0.910`

| Split | mAP@0.5 | mAP@0.5:0.95 | ImgSize | Model  |
|---:|---:|---:|---:|---|
| Val | **0.981** | **0.912** | 640 | YOLOv8n |
| Test | **0.979** | **0.910** | 640 | YOLOv8n |

> Speed(ref): ~0.3ms preprocess, 5.0ms inference, 3.2ms postprocess / image (T4)


**성능 평가 시각화**

<img width="567" height="455" alt="item detection" src="https://github.com/user-attachments/assets/e248f3d1-f9a8-4431-aa70-1dc5cbf5f412" />
<img width="567" height="455" alt="itemdetection2" src="https://github.com/user-attachments/assets/128888fa-e291-4220-86f1-c5009b8bdff4" />
<img width="1189" height="390" alt="itemdetection3" src="https://github.com/user-attachments/assets/080dad08-3f4d-4199-b986-8208990ef76c" />



---

## C) Change Detection (전후 변화 감지)

**모델**: TinyChangeUNet *(MobileNetV3 Small encoder + TinyDecoder)*  
**입력**: `before(3) + after(3) + diff(1)` = **7채널** (`diff = mean(|before - after|)`)

### 데이터셋 구축 (Before/After + GT 마스크)

**목표**
1. 실제 환경과 유사한 **다양한 변화(가림/블러/픽셀화/인페인트/이동)** 를 자동 적용해 `(before, after, mask)` 쌍 일괄 생성  
2. **마스크 규칙**: `0=배경`, `255=변경 영역`  
3. **활용**: 변화 감지, 전/후 비교, **분할(Segmentation)** 학습/벤치마킹

**생성 로직 요약**
- YOLO 라벨 박스를 기준으로 영역 선택 후, 아래 중 하나 적용  
  `black / rect(noise) / blur / pixelate / inpaint / move(영역 이동)`  
- 박스 **jitter/여유**와 **부분 가림**으로 난이도 다양화  
- 결과: `before_images/`(원본), `after_images/`(변형), `labels/`(0/255 PNG 마스크)
  
**생성 데이터 예시**
<img width="950" height="247" alt="change3" src="https://github.com/user-attachments/assets/fad3c96a-25fa-48f9-b470-d265750a26c5" />
<img width="950" height="196" alt="change4" src="https://github.com/user-attachments/assets/89c8297f-7ae5-46b0-867f-8c434cff6406" />
<img width="950" height="247" alt="change5" src="https://github.com/user-attachments/assets/cf163ec4-030f-4f0f-84b7-bb3f17741960" />
<img width="950" height="247" alt="change6" src="https://github.com/user-attachments/assets/5a86a376-045f-4b66-b920-d8ab19ac50c9" />


**모델 구조 요약**
- `concat([before, after, diff]) → 1×1 conv`로 **7ch → 3ch 축소**  
- **Encoder**: `MobileNetV3 Small (timm, features_only)` → 채널 정규화(24/40/64/96)  
- **Decoder(TinyDecoder)**: `ConvTranspose2d` 업샘플 + 스킵 + `DWConvBlock`  
- **Head**: `1×1 conv → logit(1ch)` → bilinear 업샘플(원해상도)

**학습/평가 (change_detection.ipynb)**
- 기본: `IMG_SIZE=256`, `BATCH=8`, `EPOCHS=40`, `LR=3e-4`  
- 루프: AMP(FP16), **Cosine+Warmup(2ep)**, **EMA(0.99)**, gradient clip  
- 손실: `BCEWithLogits(pos_weight)` + `Tversky(α=0.7, β=0.3)`  
- 검증 **threshold sweep**: `th ∈ [0.02, 0.40]`에서 **F1 최대**를 선택

**성능 요약 (Val 로그 기반)**  
- Early stop(F1), 총 Epoch **39**  
- **Best F1(EMA)**: **0.510 @ th=0.36**  
- 마지막 Epoch(38): `train loss=0.4530`, `val loss=0.6248`, `mIoU=0.413`, `F1=0.513`
<img width="700" height="400" alt="018df41c-b2a7-4fce-857b-5fb91b2bce7a" src="https://github.com/user-attachments/assets/423cec67-b84f-4cb0-8440-bc16c06121ee" />


**test 예시 사진**
<img width="1189" height="327" alt="chang" src="https://github.com/user-attachments/assets/11f73c91-58ab-4f07-bbb9-8710c0a4ef7b" />

| Split | Best-th | mIoU | F1 | AMP | EMA |
|---:|---:|---:|---:|:--:|:--:|
| Val | **0.36** | **0.413** | **0.513** | ✅ | ✅ |

<!-- 선택: 예측 마스크 시각화 유틸
```python
def overlay(rgb, mask, color=(0,255,255), alpha=0.35):
    import numpy as np
    m = (mask > 127).astype(np.uint8)
    tint = np.ones_like(rgb, dtype=np.uint8) * np.array(color, dtype=np.uint8)
    over = (rgb*(1-alpha) + tint*alpha).astype(np.uint8)
    out = rgb.copy(); out[m>0] = over[m>0]
    return out
```
-->

---

## 🔁 재현성
- 공통 시드: `42` (스플릿 누수 방지, 로그/체크포인트 고정)  
- 검증/저장: **EMA 가중치** 기준  
