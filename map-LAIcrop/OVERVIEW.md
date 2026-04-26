# map-LAIcrop

**작물 군락 LAI 추정 파이프라인** — lingbot-map 3D 재건 + SAM2 세그멘테이션

---

## 프로젝트 목적

농가·온실에서 작물을 **수직(nadir) 촬영**하여, lingbot-map의 metric-scale 3D 재건 결과와 SAM2 세그멘테이션을 결합해 **LAI(Leaf Area Index, 엽면적지수)** 를 간접 추정한다.

> LAI = 단위 지면 면적당 단면 잎 면적의 합  
> 높을수록 군락이 빽빽하고 광합성 효율이 높음

---

## 전체 파이프라인

```
수직 촬영 이미지 시퀀스
        │
        ▼
┌─────────────────────┐
│   lingbot-map       │  ← 코드 수정 없이 그대로 사용
│  (GCTStream)        │
│                     │
│  world_points       │  [S, H, W, 3]  metric-scale 3D 좌표
│  world_points_conf  │  [S, H, W]     신뢰도
│  images             │  [S, 3, H, W]  원본 RGB
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  LeafSegmentor      │  SAM2 Automatic Mask Generator
│  (segmentation.py)  │  → 녹색 비율 필터로 잎/배경 구분
│                     │  SAM2 미설치 시 녹색 채널 fallback
│  leaf_masks         │  [S, H, W] bool
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Ground Estimation  │  RANSAC으로 지면 평면 추정
│  (ground.py)        │  ax + by + cz + d = 0
│                     │  가장 낮은 z값 30% 포인트 후보 사용
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  LAI Calculation    │  직접법 또는 간접법 선택
│  (lai.py)           │
└─────────────────────┘
        │
        ▼
  lai_result.json / leaf_points.npy / leaf_masks.npy
```

---

## 파일 구조

```
map-LAIcrop/
├── OVERVIEW.md        ← 이 파일
├── __init__.py
├── pipeline.py        ← CLI 진입점, 전체 오케스트레이션
├── segmentation.py    ← SAM2 잎 마스킹
├── ground.py          ← RANSAC 지면 평면 추정
└── lai.py             ← LAI 계산 (직접법 + 간접법)
```

---

## LAI 추정 방법

### 직접법 (`--method direct`)

```
잎 포인트를 지면에 수직 투영
    ↓
복셀화 (voxel_size=0.005m) → 중복 제거
    ↓
점유 셀 수 × voxel_size² = 잎 투영 면적 S_leaf
    ↓
지면 포인트 convex hull = 지면 면적 S_ground
    ↓
LAI = S_leaf / S_ground
```

**장점**: 물리적으로 직관적, metric scale 활용  
**주의**: 가려진 잎(occlusion) → 과소추정 가능

---

### 간접법 (`--method gap_fraction`, 수직 촬영 권장)

```
전체 프레임 평균 Gap Fraction P(θ) = 잎 없는 픽셀 비율
    ↓
Beer-Lambert: LAI = -cos(θ) × ln(P(θ)) / k
    k = 소광계수 (구형 잎 분포 가정 ≈ 0.5)
    θ = 관측 천정각 (수직 촬영 = 0°)
```

**장점**: 전통적 LAI 추정과 일치, occlusion 문제 우회  
**전제**: 잎이 무작위로 분포한다는 가정 (랜덤 분포 모델)

---

## 설치 및 실행

### 1. 의존성 설치

```bash
# lingbot-map (상위 폴더에서)
pip install -e ..

# SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# 추가
pip install scipy
```

### 2. 모델 다운로드

```bash
# lingbot-map 체크포인트
hf download robbyant/lingbot-map lingbot-map-long.pt --local-dir ../models

# SAM2는 LeafSegmentor가 첫 실행 시 자동 다운로드
```

### 3. 실행

```bash
# 직접법
python -m map-LAIcrop.pipeline \
    --model_path ../models/lingbot-map-long.pt \
    --image_folder ./data/crop_images/ \
    --method direct \
    --output ./results/

# 간접법 (수직 촬영)
python -m map-LAIcrop.pipeline \
    --model_path ../models/lingbot-map-long.pt \
    --image_folder ./data/crop_images/ \
    --method gap_fraction \
    --keyframe_interval 2

# 메모리 부족 시
python -m map-LAIcrop.pipeline \
    --model_path ../models/lingbot-map-long.pt \
    --image_folder ./data/crop_images/ \
    --keyframe_interval 2 \
    --use_sdpa
```

### 4. 출력

```
results/
├── lai_result.json      ← LAI 수치 + 메타데이터
├── leaf_points.npy      ← 잎 3D 포인트 [N, 3]
└── leaf_masks.npy       ← 프레임별 잎 마스크 [S, H, W]
```

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--method` | `direct` | `direct` / `gap_fraction` |
| `--num_scale_frames` | `8` | lingbot-map Phase 1 프레임 수 |
| `--keyframe_interval` | `1` | KV cache 저장 간격 (클수록 메모리 절약) |
| `--conf_threshold` | `1.5` | 포인트 신뢰도 임계값 |
| `--extinction_coeff` | `0.5` | Beer-Lambert 소광계수 k |

---

## 촬영 가이드라인

| 항목 | 권장 |
|-----|------|
| 촬영 방향 | 수직(nadir) — 위에서 아래로 |
| 이동 경로 | 군락 위를 격자형으로 이동 |
| 기준물 | 알려진 크기 물체 배치 (스케일 검증용) |
| 조건 | 무풍, 균일 조명 (온실 유리) |
| 프레임 수 | 50~300장 권장 (>300은 `--keyframe_interval 2`) |

---

## lingbot-map과의 관계

```
lingbot-map/               ← 원본 프로젝트 (수정 없음)
├── lingbot_map/           ← 패키지 (import 하여 사용)
├── demo.py
└── map-LAIcrop/           ← 이 프로젝트
    └── ...
```

lingbot-map 코드는 **일절 수정하지 않는다**. `lingbot_map` 패키지를 import하여 출력 텐서만 활용한다.
