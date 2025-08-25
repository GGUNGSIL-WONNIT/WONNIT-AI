🔍 Change Detection: Before/After + GT Mask (TinyChangeUNet)



YOLO 라벨의 박스를 이용해 before/after 쌍과 변경 영역 GT 마스크를 자동 생성하고, 경량 TinyChangeUNet(MobileNetV3 encoder)으로 변화 감지(Change Detection) 를 학습/평가하는 코드와 노트북입니다.
주요 특징
합성 데이터 구축: 가림/블러/픽셀화/인페인트/이동 등 다양한 변형으로 after와 바이너리 GT 생성
경량 모델: before(3)·after(3)·diff(1) → 7채널 입력을 1×1 conv로 축소 후 MobileNetV3 인코더 + 얕은 디코더
안정 학습 루프: AMP(FP16), Cosine+Warmup, EMA 가중치 검증/저장, class imbalance용 pos_weight 자동 추정
평가 루틴: 검증셋 threshold sweep으로 최적 th 선택 → 테스트 mIoU/F1 보고, 예측 마스크 PNG 저장
