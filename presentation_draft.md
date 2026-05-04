# LingBot-Map 세미나 PPT 원고

> **컨셉**: "ML 기초가 부족해서 다시 공부해봤다" — 기존 방식 (SfM/SLAM) 과의 비교 분석 형식
> **청중**: 연구실 내부 세미나
> **논문 본문 인용/수치는 `[논문확인]`으로 표기** (Claude.ai에서 PDF 분석 후 채워넣기)

---

## Slide 1 — 표지

**제목**: LingBot-Map: 다시 공부해보는 3D 재건
부제: *Geometric Context Transformer for Streaming 3D Reconstruction*
- 발표자 / 날짜 / 연구실
- 한 줄 한 줄 따라가며 "왜 이 구조가 됐는지" 이해하는 발표

---

## Slide 2 — 발표의 목표

**우리가 답할 질문 3개**:
1. "사진 여러 장에서 3D를 만든다" — 원래 어떻게 했었나? (SfM, SLAM)
2. LingBot-Map은 그 방식들과 **무엇이 같고 무엇이 다른가?**
3. 우리 연구실에서 어디에 쓸 수 있나? (→ map-LAIcrop)

**컨셉 안내**: "ML 잘 모르는 척 다시 처음부터 따라가봅니다"

---

## Slide 3 — Mini ML 복습 (이 발표를 위한 최소한)

> *한 슬라이드로 끝, 뒤에서는 이 정도 어휘로 진행*

| 용어 | 한 줄 설명 |
|---|---|
| Feature | 이미지에서 뽑은 "특징 벡터" |
| Transformer / Attention | "어떤 입력이 어떤 입력에 주목할지" 학습 |
| Feed-forward | 한 번에 답을 뱉음 (반복 최적화 X) |
| KV Cache | 이전에 계산한 걸 저장해두고 재사용 → 빠름 |

**핵심 메타포**: SfM/SLAM = "수학으로 푸는 퍼즐", LingBot-Map = "사전학습된 모델이 그냥 답을 뱉음"

---

# 🟦 Part 1. 원래 어떻게 했나 — 비교군 정리

---

## Slide 4 — 비교군 ① Structure-from-Motion (SfM)

**가장 고전적인 방법** — COLMAP이 대표
```
사진 N장
  ↓ ① 특징점 검출 (SIFT)
  ↓ ② 사진 간 매칭
  ↓ ③ Bundle Adjustment (전체 최적화)
  ↓
3D 점 + 카메라 위치
```
- **장점**: 정확함, 수학적으로 명확
- **단점**:
  - **오프라인** — 모든 사진을 한 번에 봐야 함
  - 사진 늘어날수록 계산량 폭증 (O(N²) 매칭)
  - 텍스처 적은 면 (벽, 잎) 잘 못 잡음

**다이어그램**: 점들이 사방으로 매칭되는 그래프 + "전부 모이고 나서야 풀 수 있음"

---

## Slide 5 — 비교군 ② SLAM

**Simultaneous Localization And Mapping** — "지금 내 위치 + 지도를 동시에"
```
프레임 t 들어옴
  ↓ ① 특징점 추적
  ↓ ② 카메라 pose 추정 (이전 프레임 기반)
  ↓ ③ 새 점들을 지도에 추가
  ↓ ④ Loop closure (가끔 전체 보정)
```
- **장점**: **실시간(streaming)** — 한 프레임씩 처리
- **단점**:
  - **drift** — 시간 갈수록 오차 누적 → loop closure 필요
  - 텍스처 약한 영역 취약
  - 보통 sparse 점만 만듦 (dense는 별도)

**다이어그램**: 시간 축에 따라 점이 늘어나는 그림 + "오차도 같이 늘어남"

---

## Slide 6 — SfM vs SLAM 한눈에

| | SfM | SLAM |
|---|---|---|
| 처리 방식 | Batch (한 번에) | Streaming (프레임 단위) |
| 속도 | 느림 | 빠름 |
| 정확도 | 높음 | 중간 (drift) |
| 결과 | Sparse → Dense MVS | Sparse 위주 |
| 학습 필요? | ❌ (수학) | ❌ (수학) |

**핵심 한계 (둘 다 공통)**: **수학으로 푼다 → 사람이 만든 가정 (특징점, 매칭)에 의존**

---

# 🟩 Part 2. LingBot-Map은 무엇이 다른가

---

## Slide 7 — 한 문장 요약

> **"이미지 N장을 입력하면, 학습된 Transformer가 3D 점 + 카메라 위치를 한 번에 뱉는다.
> 그것도 streaming으로."**

세 가지 키워드:
- **Feed-forward** (Bundle Adjustment 없음)
- **Foundation Model** (사전학습 → 재학습 X)
- **Streaming + KV Cache** (긴 시퀀스에 강함)

---

## Slide 8 — 큰 그림 비교 (SfM/SLAM vs LingBot-Map)

| 단계 | SfM | SLAM | **LingBot-Map** |
|---|---|---|---|
| 특징 추출 | SIFT (수동) | ORB/SIFT | **DINOv2 ViT** (학습) |
| 매칭/추적 | 명시적 매칭 | 추적 | **Self-Attention** |
| 최적화 | Bundle Adjustment | EKF/PG | **없음 — 한 번에** |
| Pose | 수학적 풀이 | 추적 | **Camera Head**가 예측 |
| 처리 | Batch | Streaming | **Streaming + Causal Attention** |
| Drift 대응 | 전역 BA | Loop closure | **Anchor frames + KV cache** |

**다이어그램 후보**: 위 표를 가로 3컬럼 시각화 (각 단계 아이콘) + 화살표로 "LingBot-Map은 이 단계들을 하나의 모델 안에서"

---

## Slide 9 — LingBot-Map 파이프라인 5단계

> *코드 기반 — 각 단계 한 줄씩*

```
입력: 이미지 시퀀스 (518×378, N장)
  │
  ▼
① DINOv2 ViT-L/14: 이미지 → 패치 토큰
  │   (37×27 패치 + special tokens: camera, register, scale)
  ▼
② Frame Block: 프레임 내부 self-attention
  │
  ▼
③ Global Block + KV Cache (★ 핵심)
  │   Phase 1: 처음 8프레임 = bidirectional (서로 다 봄)
  │   Phase 2: 그 이후 = causal (과거만 봄)
  ▼
④ 3개의 예측 헤드:
  │   • Camera Head: pose [B, S, 9]
  │   • Depth Head: 깊이맵
  │   • Point Head: world_points [B, S, H, W, 3]
  ▼
출력: 3D 점 + 카메라 위치 (metric scale)
```

**다이어그램**: 위 흐름을 박스 + 화살표로

---

## Slide 10 — 왜 Phase 1 / Phase 2로 나눴나?

> *이 부분이 LingBot-Map의 가장 영리한 설계*

**문제**: 첫 프레임이 들어오자마자 streaming 시작하면 → **scale 모름** (단안 카메라는 절대 크기 추정 불가)

**해결**:
- **Phase 1 (처음 8프레임)** — 서로서로 보면서 **scale + 초기 기하** 잡음 (SfM처럼 batch)
- **Phase 2 (나머지)** — 그 다음부터는 streaming 하나씩, 과거만 참조 (SLAM처럼 causal)

**비교**:
- SfM: 전체 batch → 느리지만 정확
- SLAM: 전체 streaming → 빠르지만 drift
- **LingBot-Map: 짧게 batch + 길게 streaming → 두 마리 토끼**

**다이어그램**: 시간 축 위에 Phase 1 (양방향 화살표 8개) + Phase 2 (단방향 화살표 N개)

---

## Slide 11 — 핵심 무기: KV Cache

**KV Cache가 뭐냐 (한 줄)**: Transformer가 과거 프레임에 대해 계산한 Key/Value를 저장해서 새 프레임이 들어올 때 재계산 안 함.

**LingBot-Map의 KV Cache 전략 — 3중 구조**

| 종류 | 코드명 | 역할 | 비유 |
|---|---|---|---|
| Anchor (Scale frames) | `kv_cache_scale_frames=8` | 항상 보존, drift 방지 | "원점 기준" |
| Sliding window | `kv_cache_sliding_window=64` | 최근 64프레임 유지 | "단기 기억" |
| Keyframe interval | `keyframe_interval=N` | N번째만 저장 | "메모리 절약 (N배)" |

**README 마케팅 용어와 매핑**:
- "anchor context" → Scale frames
- "pose-reference window" → Sliding window
- "trajectory memory" → Keyframe-based long-range cache

**SLAM과의 비교**:
- SLAM: keyframe + bundle adjustment로 drift 보정
- LingBot-Map: **scale frames를 attention의 key로 항상 살려둠 → drift 자체가 잘 안 생김**

---

## Slide 12 — 성능 (논문 수치)

> `[논문확인]` — 이 슬라이드는 PDF 분석 후 채울 것
- FPS (~20 FPS @ 518×378)
- 긴 시퀀스 처리 가능 길이 (10,000+ frames)
- ATE / RPE / Chamfer Distance 등 SLAM·SfM 대비 벤치마크 수치
- 비교 baseline: VGGT, DUSt3R, MASt3R, COLMAP, ORB-SLAM3 등 (논문이 비교한 거)

**메시지 후보**: "더 빠른데 더 정확하다" — 표/그래프

---

## Slide 13 — 비교 정리: 한 슬라이드로 말하면

| | SfM (COLMAP) | SLAM (ORB-SLAM3) | **LingBot-Map** |
|---|---|---|---|
| 특징점 | Hand-crafted (SIFT) | Hand-crafted | **학습된 ViT feature** |
| 매칭 | 명시적 | 추적 | **Attention** |
| Pose 추정 | BA로 푼다 | EKF/그래프 최적화 | **MLP head가 예측** |
| 결과 밀도 | Sparse → Dense MVS 별도 | Sparse | **Dense (per-pixel depth)** |
| 처리 방식 | Batch | Streaming | **둘 다 (Phase 1 + 2)** |
| Drift 대응 | 전역 BA | Loop closure | **KV cache + scale anchor** |
| 학습 데이터 | 불필요 | 불필요 | **필수 (foundation model)** |
| **본질** | 수학 풀이 | 수학 풀이 | **사전학습된 표현으로 한 번에** |

---

# 🟨 Part 3. 우리 연구실 활용 — map-LAIcrop

---

## Slide 14 — map-LAIcrop이 뭐냐

> **🚧 개발 중 (in development)**

**목적**: 농가/온실에서 작물을 위에서 (수직) 촬영 → **LAI(엽면적지수)** 추정

**LAI**: 단위 지면 면적당 잎 단면 면적의 합 — 광합성 효율 지표
- 높을수록 군락이 빽빽함
- 전통적으로는 직접 잎을 따서 측정 or 비싼 LAI-2200 같은 장비

---

## Slide 15 — map-LAIcrop 파이프라인

```
수직 촬영 이미지
   ↓
① LingBot-Map (수정 없이 그대로) → metric-scale 3D 점
   ↓
② SAM2 잎 세그멘테이션 → 잎 마스크
   ↓
③ RANSAC으로 지면 평면 추정
   ↓
④ LAI 계산 (두 방법 중 선택)
     • Direct: 잎 투영 면적 / 지면 면적
     • Gap Fraction: Beer-Lambert (LAI = -cos(θ) × ln(P) / k)
   ↓
lai_result.json
```

**핵심 설계 결정**: LingBot-Map 코드 **일절 수정 안 함**, 출력 텐서만 활용 → 나중에 LingBot-Map이 업데이트되어도 그대로 적용 가능

---

## Slide 16 — 왜 LingBot-Map이 적합한가 (이 응용에)

| 작물 촬영의 특성 | 기존 방식의 어려움 | LingBot-Map의 강점 |
|---|---|---|
| 잎 — 비슷한 텍스처 반복 | SfM/SLAM 매칭 실패 | Attention은 globally 본다 |
| 좁은 범위 격자 이동 | SLAM drift 누적 | Anchor frames로 보정 |
| Metric scale 필요 (m²) | 단안 카메라는 scale 모름 | Phase 1에서 scale 추정 |
| 수백 장 이미지 | SfM 너무 느림 | Streaming + KV cache |

→ "**작물 군락 시나리오는 LingBot-Map이 빛나는 사례**"

---

## Slide 17 — 현재 상태 & 다음 단계

**개발 중 (WIP)**:
- 파이프라인 자체는 동작 (CLI 진입점 완성)
- SAM2 통합 완료, fallback (녹색 채널 마스킹) 까지
- **검증 필요**:
  - 실제 작물 데이터 수집
  - 직접 측정 LAI vs 추정 LAI 비교
  - 직접법 vs Gap Fraction 어느 게 우리 환경에 맞는지

**다음 단계**:
- 데이터 수집 프로토콜 정립
- 기준물 (스케일 검증용) 포함한 촬영
- LAI 외 다른 지표 (높이, 군락 부피) 확장 가능성

---

# 🟪 Part 4. 정리

---

## Slide 18 — 정리: 다시 공부해서 알게 된 것

1. **3D 재건의 패러다임 전환**
 - SfM/SLAM = "수학으로 푸는 퍼즐"
 - LingBot-Map = "Foundation Model이 한 번에 답을 뱉음"

2. **하지만 완전 새 발명은 아님**
 - Phase 1 = SfM의 batch 정신
 - Phase 2 = SLAM의 streaming 정신
 - **둘을 이어붙이고 transformer 위에 얹은 것**

3. **연구실 응용**: 작물 촬영 → LAI 추정 (개발 중)
 - LingBot-Map의 dense + metric + streaming 특성이 이 응용에 잘 맞음

---

## Slide 19 — Q&A / Discussion

**열려있는 질문**:
- LingBot-Map의 학습 데이터 분포 → 농작물 같은 도메인에서 어디까지 잘 동작할까?
- 우리 작물 데이터로 fine-tuning 한다면 어디부터 손대야 할까?
- LAI 외에 우리가 LingBot-Map으로 할 수 있는 다른 응용?

---

## (부록) Slide A1 — 레퍼런스

- **논문**: Chen et al., "Geometric Context Transformer for Streaming 3D Reconstruction", arXiv 2604.14141, 2026
- **코드**: github.com/... (LingBot-Map 공식 repo)
- **기반**: VGGT (Wang et al.), DINOv2 (Meta), FlashInfer
- **비교군 참고**:
 - COLMAP (Schönberger & Frahm, CVPR 2016)
 - ORB-SLAM3 (Campos et al., 2021)

---

# 다음 작업 (TODO)

1. **Slide 12 (성능 수치)** — Claude.ai에서 PDF 분석 후 `[논문확인]` 부분 채우기
2. **다이어그램 / 그림** — 위 코드 블록 형태 흐름도들을 Excalidraw 등으로 시각화
3. (선택) **Slide 13 비교표 분리** — 너무 빽빽하면 SfM 비교 / SLAM 비교 두 장으로

---

# 부록: Claude.ai 분석용 프롬프트

## Version A — 원고 채우기 + 다듬기 (추천)

```
# Role
당신은 3D Reconstruction / Computer Vision 분야의 전문 연구원입니다.
연구실 내부 세미나 발표용 PPT 원고 작성을 도와주세요.

# 입력
1. 논문 PDF: `lingbot-map_paper.pdf` (Geometric Context Transformer for
   Streaming 3D Reconstruction, Chen et al. 2026)
2. 아래 첨부된 PPT 원고 초안 — 코드베이스 분석 기반으로 이미 작성됨

# 발표 컨셉 (반드시 유지)
- 청중: 연구실 내부 세미나, ML 기초 지식이 부족한 멤버 포함
- 톤: "ML 기초가 부족해서 다시 공부해봤다" 라는 친근한 컨셉
- 핵심 형식: 기존 방식(SfM, SLAM) vs LingBot-Map 비교 분석
- ML 세부 설명은 한 슬라이드에 1~2줄로만 — 깊게 들어가지 않음

# 작업 (4가지)

## 1. `[논문확인]` 부분 채우기
원고에서 `[논문확인]`으로 표시된 부분 (특히 Slide 12 - 성능 수치)을
논문 본문의 정확한 수치로 채워주세요. 다음 항목들이 필요합니다:
- FPS, 처리 가능한 최대 시퀀스 길이
- 비교 baseline (실제로 논문이 비교한 방법들 - VGGT? DUSt3R? COLMAP? 등)
- 핵심 metric 수치 (ATE, RPE, Chamfer Distance, AbsRel 등 논문이 사용한 것)
- "X에서 Y 대비 Z% 우수" 형식으로 명료하게

## 2. 코드 분석 vs 논문 주장 검증
원고에 적힌 다음 주장들이 논문 본문과 일치하는지 확인하고,
틀리거나 부정확한 부분은 수정 제안:
- "Phase 1 = bidirectional, Phase 2 = causal streaming"
- "anchor context = scale frames, pose-reference window = sliding window,
  trajectory memory = keyframe-based long-range cache" (이 매핑)
- "feed-forward, Bundle Adjustment 없음"
- "DINOv2 ViT-L/14 backbone"

## 3. 추가 보강이 필요한 부분
- 논문에만 있고 원고에는 빠진 핵심 기여(contribution)가 있다면 추가 슬라이드 제안
- Ablation study 중 발표할 가치가 있는 것 1~2개 선별

## 4. 용어 통일
- 영어 기술 용어는 원문 유지 (Granularity, Outlier, Throughput, Latency,
  End-to-End, Bundle Adjustment, Feed-forward, KV Cache 등)
- 문장 구조는 자연스러운 한국어
- 명사/핵심 동사는 영어 섞어 사용 가능 (전문적 문체)

# 출력 형식
1. **수정된 PPT 원고** (변경된 슬라이드만, 변경 사유 한 줄씩)
2. **추가 제안 슬라이드** (있다면)
3. **검증 결과 요약**: 원고와 논문이 일치/불일치한 부분 표로 정리

---
[여기에 위 PPT 원고 전체 붙여넣기]
```

## Version B — 처음부터 다시

```
# Role
당신은 3D Reconstruction / Computer Vision 분야의 전문 연구원입니다.

# Constraints (Critical)
1. 핵심 기술 용어, Metric, 약어, 포맷 명칭은 영어 원문 그대로
   (예: Granularity, Outlier, Throughput, Bundle Adjustment, Feed-forward,
   KV Cache, Streaming, Causal Attention, Foundation Model)
2. 문장 구조는 자연스러운 한국어, 명사/핵심 동사는 영어 섞어 사용
3. 본문 시작 전 [Key Terms & Definitions] 섹션 필수

# 입력
- 논문 PDF: lingbot-map_paper.pdf (Chen et al. 2026,
  "Geometric Context Transformer for Streaming 3D Reconstruction")
- 첨부된 PPT 원고 초안 — 코드베이스 분석 기반 (참고용)

# 작업
연구실 내부 세미나용 PPT 원고를 작성해주세요.

## 발표 컨셉 (반드시 준수)
- 청중: ML 기초 지식이 부족한 연구실 멤버
- 톤: "ML 기초부터 다시 공부해봤다"
- 형식: **기존 방식(SfM, SLAM) vs LingBot-Map 비교 분석**
- ML 세부 설명은 한 슬라이드에 1~2줄로만
- 슬라이드 분량 제한 없음 (필요한 만큼)

## 출력 구조

### Part 0. 논문 요약 (PPT 작성 전 자체 분석)
다음 4단계로 먼저 논문을 정리:

**[1. Key Terms & Definitions]**
- 논문 핵심 약어/개념 3~5개

**[2. Motivation & Problem Statement]**
- 해결하려는 핵심 문제
- 기존 연구(SfM, SLAM, VGGT, DUSt3R 등)의 한계

**[3. Method & Key Results]**
- 제안 방법론 (architecture, training, inference)
- 핵심 정량 결과 (수치, baseline 대비 우위)

**[4. Conclusion & Impact]**
- 최종 결론
- 학계/산업계 영향

### Part 1. PPT 원고
위 분석을 토대로 다음 흐름의 PPT 원고 작성:
1. 표지 / 발표 목표
2. ML 최소 복습 (1슬라이드)
3. 비교군 ① SfM (COLMAP) — 어떻게 동작, 한계
4. 비교군 ② SLAM (ORB-SLAM3 등) — 어떻게 동작, 한계
5. SfM vs SLAM 한눈에 비교
6. LingBot-Map: 한 문장 요약 + 큰 그림 비교표
7. LingBot-Map 파이프라인 (DINOv2 → Frame → Global → 3 heads)
8. 핵심 설계 1: Phase 1 (bidirectional) + Phase 2 (causal) 의 의미
9. 핵심 설계 2: KV Cache 3중 구조 (anchor / sliding window / keyframe)
10. 성능 (논문 수치 + baseline 비교)
11. 최종 비교표 (SfM vs SLAM vs LingBot-Map)
12. **map-LAIcrop 소개** (작물 LAI 추정, 개발 중) — 첨부 원고 참고
13. 정리 / Q&A

각 슬라이드마다:
- 핵심 메시지 한 문장
- 본문 (bullet 또는 표)
- 다이어그램 후보 (있으면)

### Part 2. 디자인 노트
Claude Design에 넘길 때 도움이 될 시각화 제안 (다이어그램 종류,
컬러 코딩 제안, 강조하면 좋을 부분 등)

---
[참고용 PPT 원고 초안 — 첨부]
[그 다음 PDF 업로드]
```
