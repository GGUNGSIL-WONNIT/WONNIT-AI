🔍 Change Detection: Before/After + GT Mask (TinyChangeUNet)
<p align="center"> <img src="https://img.shields.io/badge/split-train%2Fval%2Ftest-1f6feb"> <img src="https://img.shields.io/badge/masks-binary%200%2F255-2ea043"> <img src="https://img.shields.io/badge/repro-seed%3D42-8957e5"> <img src="https://img.shields.io/badge/encoder-MobileNetV3%20Small-f2cc60"> <img src="https://img.shields.io/badge/AMP-FP16%20enabled-0ea5e9"> </p> <p align="center"> <b>YOLO 라벨 박스로 합성한 before/after 쌍과 GT 마스크</b>를 이용해, 경량 <b>TinyChangeUNet</b>(MobileNetV3 encoder)으로 <b>변화 감지</b>를 학습/평가합니다. </p>
✨ 한눈에 보는 핵심
합성 데이터 구축: 가림/블러/픽셀화/인페인트/이동 등 다양한 변형으로 after와 바이너리 GT를 자동 생성합니다.
경량 모델: before(3) · after(3) · diff(1) → 7채널 입력을 1×1 conv로 축소 후 MobileNetV3 Small 인코더 + 얕은 디코더로 추론합니다.
안정 학습 루프: AMP(FP16), Cosine+Warmup, EMA 가중치로 검증/저장, class imbalance용 pos_weight 자동 추정을 지원합니다.
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
diff = mean(|before - after|) 한 채널을 추가하여 변화신호를 직접 투입합니다.
⚡ 10초 퀵스타트
# (로컬/콜랩 공통) 의존성 원라인 설치
pip install torch torchvision timm opencv-python numpy matplotlib tqdm
<details> <summary><b>📒 노트북 사용 순서(권장)</b></summary>
Dataset 로더 정의: PairDataset2In
Model 정의: TinyChangeUNet
Train/Eval 루프 실행: IMG_SIZE=256, BATCH=8, LR=3e-4, EPOCHS=40 기준
Self-contained TEST Eval: 재시작 후에도 체크포인트 로드 → <b>검증 sweep으로 th 선택</b> → 테스트 mIoU/F1 보고 & ./test_preds/*.png 저장
</details>
🧩 폴더 규칙 (요약)
pairs_out_cd/
  train/{before_images, after_images, labels}
  val/{before_images, after_images, labels}
  test/{before_images, after_images, labels}
동일 파일명(클래스 접두사 포함)으로 before/after/mask 매칭입니다.
마스크는 0/255 바이너리 PNG이며, 리사이즈 시 nearest 권장입니다.
🧪 재현성 & 설정
고정 시드: seed=42 입니다.
EMA 가중치로 검증/저장합니다(분산 감소).
클래스 불균형은 pos_weight 자동 추정으로 보정합니다.
강한 결정론이 필요하면 torch.use_deterministic_algorithms(True), cudnn.benchmark=False를 추가하십시오.
🧷 결과 카드(템플릿)
Split	Best-th	mIoU	F1	AMP	EMA
Val	0.18	0.41	0.51	✅	✅
Test	0.18	0.40	0.50	✅	✅
