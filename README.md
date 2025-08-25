🔍 Change Detection: Before/After + GT Mask (TinyChangeUNet)
<p align="center"> <img src="https://img.shields.io/badge/split-train%2Fval%2Ftest-1f6feb"> <img src="https://img.shields.io/badge/masks-binary%200%2F255-2ea043"> <img src="https://img.shields.io/badge/repro-seed%3D42-8957e5"> <img src="https://img.shields.io/badge/encoder-MobileNetV3%20Small-f2cc60"> <img src="https://img.shields.io/badge/AMP-FP16%20enabled-0ea5e9"> </p> <p align="center"> YOLO 라벨 박스로 합성한 <b>before/after 쌍과 GT 마스크</b>를 이용해, 경량 <b>TinyChangeUNet</b>(MobileNetV3 encoder)으로 <b>변화 감지</b>를 학습/평가하는 코드와 노트북입니다. </p>
✨ 한눈에 보는 핵심
합성 데이터 구축: 가림/블러/픽셀화/인페인트/이동 등 다양한 변형으로 after와 바이너리 GT(0/255) 를 자동 생성합니다.
경량 모델: before(3) · after(3) · diff(1) → 7채널 입력을 1×1 conv로 축소 후 MobileNetV3 Small 인코더 + 얕은 디코더로 추론합니다.
안정 학습 루프: AMP(FP16), Cosine+Warmup, EMA 가중치 검증/저장, class imbalance용 pos_weight 자동 추정을 지원합니다.
평가 루틴: 검증셋 threshold sweep으로 최적 th 선택 → 테스트 mIoU/F1 리포트 + PNG 마스크 저장까지 자동화합니다.
🧭 파이프라인 개요
flowchart LR
    A[Before (3ch)] ---|concat| R[Reduce 1x1 Conv → 3ch]
    B[After (3ch)]  ---|concat| R
    D[abs(Before-After) (1ch)] ---|concat| R
    R --> E[MobileNetV3 Encoder]
    E -->|multi-scale| Dec[Tiny Decoder]
    Dec --> H[Head 1x1 Conv]
    H --> M[Logit → Sigmoid → Binary Mask (0/255)]
diff = mean(|before - after|) 1채널을 함께 투입하여 변화 신호를 직접 활용합니다.
⚡ 10초 퀵스타트
pip install torch torchvision timm opencv-python numpy matplotlib tqdm
<details> <summary><b>📒 노트북 사용 순서(권장)</b></summary>
Dataset 로더 정의: PairDataset2In
Model 정의: TinyChangeUNet
Train/Eval 루프 실행: IMG_SIZE=256, BATCH=8, LR=3e-4, EPOCHS=40 기준으로 학습합니다.
Self-contained TEST Eval 실행: 재시작 후에도 체크포인트 로드 → <b>검증 sweep으로 th 선택</b> → 테스트 mIoU/F1 보고 및 ./test_preds/*.png 저장을 수행합니다.
</details>
🧩 데이터/폴더 규칙
pairs_out_cd/
  train/{before_images, after_images, labels}
  val/{before_images, after_images, labels}
  test/{before_images, after_images, labels}
meta/pairs_{train,val}.json
동일 파일명(클래스 접두사 포함)으로 before/after/mask 를 매칭합니다.
마스크는 0/255 바이너리 PNG 입니다. 리사이즈 시 nearest 보간을 사용합니다(경계 보존).
🔧 주요 하이퍼파라미터
이름	기본	설명
IMG_SIZE	256	입력 해상도(정사각)
BATCH	8	배치 크기
EPOCHS	40	학습 에폭
LR	3e-4	초기 러닝레이트(Cosine + 2-epoch warmup)
USE_POSTPROC	False	평가 시 소블랍 제거(간단 평균풀 기반)
SAVE_PATH	.../change_tiny_mnv3_best.pth	EMA 가중치 저장 경로
클래스 불균형은 pos_weight 자동 추정으로 보정합니다.
🧪 평가 & 시각화
검증 임계값 스윕: np.linspace(0.02, 0.40, 40) 범위에서 F1 최대화로 best_th를 선택합니다.
테스트 리포트: 선택된 th로 mIoU/F1을 보고하며, 예측 마스크를 ./test_preds/*.png로 저장합니다.
<details> <summary><b>🎨 오버레이 예시 코드(After 위에 반투명 마스크)</b></summary>
def overlay(rgb, mask, color=(0,255,255), alpha=0.35):
    import numpy as np
    m = (mask > 127).astype(np.uint8)
    tint = np.ones_like(rgb, dtype=np.uint8) * np.array(color, dtype=np.uint8)
    over = (rgb*(1-alpha) + tint*alpha).astype(np.uint8)
    out = rgb.copy(); out[m>0] = over[m>0]
    return out
</details>
🧷 결과 카드(템플릿)
Split	Best-th	mIoU	F1	AMP	EMA
Val	0.18	0.41	0.51	✅	✅
Test	0.18	0.40	0.50	✅	✅
정성 검증을 위해 Before/After/Overlay 샘플 2–3장을 함께 첨부하는 것을 권장합니다.
🔁 재현성
고정 시드 seed=42 를 사용합니다. 스플릿은 누수를 방지하도록 사례 단위로 구성하는 것을 권장합니다.
검증 및 저장은 EMA 가중치 기준으로 수행합니다(평가 분산 감소).
강한 결정론이 필요하면 torch.use_deterministic_algorithms(True) 및 cudnn.benchmark=False 설정을 추가하십시오.
📄 라이선스 / 데이터
코드/데이터 라이선스는 저장소 루트의 LICENSE, DATA_LICENSE 파일에 명시하시기 바랍니다.
민감 데이터의 비의도 사용을 금지하는 데이터카드를 함께 제공하는 것을 권장합니다.
🙏 감사의 말
timm 기반 MobileNetV3 백본을 사용합니다.
