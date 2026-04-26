"""
지면 평면 추정: RANSAC 기반

lingbot-map의 world_points에서 지면 평면 ax + by + cz + d = 0 을 추정한다.
수직(nadir) 촬영에서 가장 낮은 z값 영역이 지면이라고 가정한다.
"""

import numpy as np


def estimate_ground_plane(
    points: np.ndarray,
    ransac_iterations: int = 1000,
    distance_threshold: float = 0.02,
    bottom_ratio: float = 0.3,
) -> tuple[np.ndarray, float]:
    """
    RANSAC으로 지면 평면 법선 벡터와 offset d를 추정한다.

    Args:
        points: [N, 3] world coordinates (numpy)
        ransac_iterations: RANSAC 반복 횟수
        distance_threshold: 인라이어 판정 거리 (m 단위, metric scale 기준)
        bottom_ratio: 가장 낮은 z값 비율의 포인트만 RANSAC 후보로 사용

    Returns:
        normal: [3,] 법선 벡터 (단위 벡터, 위쪽 방향 보장)
        d: 평면 offset  →  normal @ point + d ≈ 0
    """
    if len(points) < 3:
        raise ValueError("포인트가 3개 미만입니다.")

    # 수직 촬영 → z값이 작은 쪽이 지면
    z_thresh = np.percentile(points[:, 2], bottom_ratio * 100)
    candidates = points[points[:, 2] <= z_thresh]

    if len(candidates) < 3:
        candidates = points

    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = -float(np.median(candidates[:, 2]))
    best_inlier_count = 0

    rng = np.random.default_rng(seed=42)

    for _ in range(ransac_iterations):
        idx = rng.choice(len(candidates), 3, replace=False)
        p0, p1, p2 = candidates[idx]

        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm

        d = -float(normal @ p0)

        distances = np.abs(candidates @ normal + d)
        inlier_count = (distances < distance_threshold).sum()

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

    # 법선이 위(+z)를 향하도록 부호 보정
    if best_normal[2] < 0:
        best_normal = -best_normal
        best_d = -best_d

    inlier_ratio = best_inlier_count / len(candidates)
    print(f"  지면 평면 추정 완료: normal={best_normal.round(3)}, d={best_d:.3f}, "
          f"inlier={best_inlier_count}/{len(candidates)} ({inlier_ratio:.1%})")

    return best_normal, best_d


def point_to_plane_distance(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """각 포인트에서 평면까지의 부호 있는 거리를 반환한다."""
    return points @ normal + d


def project_to_ground(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """포인트를 지면 평면에 수직 투영한다."""
    dists = point_to_plane_distance(points, normal, d)
    return points - np.outer(dists, normal)


def ground_area(points: np.ndarray, normal: np.ndarray) -> float:
    """
    지면에 투영된 포인트들의 볼록 껍질(Convex Hull) 면적을 반환한다 (m²).

    지면 법선에 수직인 두 기저 벡터로 2D 투영 후 면적 계산.
    """
    from scipy.spatial import ConvexHull

    # 법선에 수직인 두 단위 벡터 생성
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    pts_2d = np.stack([points @ u, points @ v], axis=1)

    try:
        hull = ConvexHull(pts_2d)
        return float(hull.volume)  # 2D에서 volume = 면적
    except Exception:
        # 포인트가 거의 일직선인 경우 falloff
        x_range = pts_2d[:, 0].max() - pts_2d[:, 0].min()
        y_range = pts_2d[:, 1].max() - pts_2d[:, 1].min()
        return float(x_range * y_range)
