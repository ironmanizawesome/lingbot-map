"""
LAI 계산 모듈

두 가지 방법을 제공한다:
  1. direct        — 3D 포인트 클라우드에서 잎 표면적 / 지면 면적
  2. gap_fraction  — Beer-Lambert 법칙 기반 간접 추정
"""

import numpy as np
from .ground import ground_area, project_to_ground


# ──────────────────────────────────────────────────────────────
# 직접법
# ──────────────────────────────────────────────────────────────

def compute_lai_direct(
    leaf_pts: np.ndarray,
    ground_normal: np.ndarray,
    ground_d: float,
    voxel_size: float = 0.005,
) -> tuple[float, dict]:
    """
    3D 잎 포인트에서 LAI를 직접 추정한다.

    LAI = (잎 투영 면적) / (지면 면적)

    잎 포인트를 지면에 수직 투영한 뒤 복셀화하여 면적을 추정한다.
    복셀 하나 = voxel_size² m² 로 가정한다.

    Args:
        leaf_pts: [N, 3] 잎 3D 포인트 (m 단위)
        ground_normal: [3,] 지면 법선 벡터
        ground_d: 지면 평면 offset
        voxel_size: 복셀 한 변의 크기 (m), 작을수록 정밀하지만 느림

    Returns:
        lai: float
        meta: 중간 계산값 딕셔너리
    """
    if len(leaf_pts) == 0:
        return 0.0, {"leaf_area_m2": 0.0, "ground_area_m2": 0.0, "note": "잎 포인트 없음"}

    # 잎 포인트를 지면에 투영
    leaf_proj = project_to_ground(leaf_pts, ground_normal, ground_d)

    # 지면 투영 면적 (convex hull)
    ground_area_m2 = ground_area(leaf_proj, ground_normal)

    # 잎 투영 면적: 복셀화로 중복 제거 후 격자 개수 × voxel_size²
    leaf_area_m2 = _voxel_projected_area(leaf_proj, ground_normal, voxel_size)

    lai = leaf_area_m2 / ground_area_m2 if ground_area_m2 > 0 else 0.0

    meta = {
        "leaf_area_m2": round(leaf_area_m2, 4),
        "ground_area_m2": round(ground_area_m2, 4),
        "voxel_size_m": voxel_size,
    }
    return lai, meta


def _voxel_projected_area(
    projected_pts: np.ndarray,
    normal: np.ndarray,
    voxel_size: float,
) -> float:
    """투영된 포인트를 2D 격자에 뿌린 뒤 점유 셀 수 × voxel_size² 를 반환한다."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    coords_u = projected_pts @ u
    coords_v = projected_pts @ v

    grid_u = np.floor(coords_u / voxel_size).astype(np.int64)
    grid_v = np.floor(coords_v / voxel_size).astype(np.int64)

    occupied = set(zip(grid_u.tolist(), grid_v.tolist()))
    return len(occupied) * (voxel_size ** 2)


# ──────────────────────────────────────────────────────────────
# 간접법 (Gap Fraction / Beer-Lambert)
# ──────────────────────────────────────────────────────────────

def compute_lai_gap_fraction(
    leaf_masks: np.ndarray,
    extinction_coeff: float = 0.5,
    zenith_angle_deg: float = 0.0,
) -> tuple[float, dict]:
    """
    Gap Fraction 방법으로 LAI를 추정한다.

    Beer-Lambert:  P(θ) = exp(-k × LAI / cos(θ))
    → LAI = -cos(θ) × ln(P(θ)) / k

    P(θ): 빛이 캐노피를 통과하는 비율 = 잎이 없는 픽셀 비율 (Gap Fraction)

    Args:
        leaf_masks: [S, H, W] bool, True = 잎 픽셀
        extinction_coeff: 소광계수 k (구형 잎 분포 가정 시 ≈ 0.5)
        zenith_angle_deg: 관측 천정각 (수직 촬영 = 0도)

    Returns:
        lai: float
        meta: 중간 계산값 딕셔너리
    """
    S, H, W = leaf_masks.shape
    total_pixels = H * W

    # 프레임별 gap fraction (잎이 없는 픽셀 비율)
    gap_per_frame = 1.0 - leaf_masks.reshape(S, -1).mean(axis=1)  # [S,]

    # 전체 시퀀스 평균
    mean_gap = float(gap_per_frame.mean())
    mean_gap = max(mean_gap, 1e-6)  # ln(0) 방지

    cos_theta = np.cos(np.radians(zenith_angle_deg))
    lai = -cos_theta * np.log(mean_gap) / extinction_coeff

    meta = {
        "mean_gap_fraction": round(mean_gap, 4),
        "extinction_coeff_k": extinction_coeff,
        "zenith_angle_deg": zenith_angle_deg,
        "num_frames": S,
    }
    return float(lai), meta
