"""
LeafSegmentor: SAM2 기반 잎 세그멘테이션

각 프레임에서 작물 잎 픽셀을 True로 표시하는 bool 마스크 [S, H, W]를 반환한다.
SAM2가 설치되어 있지 않으면 ImportError를 출력하고 더미 마스크를 반환한다.
"""

import numpy as np
import torch
from tqdm.auto import tqdm


class LeafSegmentor:
    """
    SAM2 Automatic Mask Generator를 이용해 프레임별 잎 마스크를 생성한다.

    작물 잎의 특성 (녹색 계열, 중간 크기) 에 맞게 SAM2 파라미터를 조정해
    지면/배경 영역을 최대한 제거한다.
    """

    def __init__(
        self,
        sam2_checkpoint: str = None,
        model_cfg: str = "sam2_hiera_large.yaml",
        device: torch.device = None,
        # 잎 후보 필터링 파라미터
        green_threshold: float = 0.05,   # G - max(R,B) > threshold → 녹색 픽셀
        min_leaf_ratio: float = 0.3,     # 마스크 내 녹색 픽셀 비율이 이 이상이어야 잎으로 판정
        min_area_ratio: float = 0.001,   # 이미지 대비 최소 마스크 면적 비율
        max_area_ratio: float = 0.5,     # 이미지 대비 최대 마스크 면적 비율 (배경 제거)
    ):
        self.device = device or torch.device("cpu")
        self.green_threshold = green_threshold
        self.min_leaf_ratio = min_leaf_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.generator = self._load_sam2(sam2_checkpoint, model_cfg)

    def _load_sam2(self, checkpoint, model_cfg):
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            if checkpoint is None:
                checkpoint = self._auto_download_sam2()

            sam2 = build_sam2(model_cfg, checkpoint, device=self.device)
            generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            print("SAM2 로드 완료.")
            return generator

        except ImportError:
            print(
                "[경고] SAM2가 설치되지 않았습니다.\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                "녹색 픽셀 기반 단순 세그멘테이션으로 대체합니다."
            )
            return None

    @staticmethod
    def _auto_download_sam2() -> str:
        """SAM2-large 체크포인트를 huggingface에서 자동 다운로드한다."""
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="facebook/sam2-hiera-large",
            filename="sam2_hiera_large.pt",
        )
        print(f"SAM2 체크포인트 다운로드 완료: {path}")
        return path

    # ──────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────

    def segment_sequence(self, images: torch.Tensor) -> np.ndarray:
        """
        Args:
            images: [S, 3, H, W] float32 텐서, 픽셀 범위 [0, 1]

        Returns:
            leaf_masks: [S, H, W] bool numpy 배열
        """
        S, _, H, W = images.shape
        leaf_masks = np.zeros((S, H, W), dtype=bool)

        for i in tqdm(range(S), desc="세그멘테이션"):
            frame = images[i]  # [3, H, W]
            rgb_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            if self.generator is not None:
                leaf_masks[i] = self._segment_sam2(rgb_np)
            else:
                leaf_masks[i] = self._segment_green(rgb_np)

        return leaf_masks

    # ──────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────

    def _segment_sam2(self, rgb_np: np.ndarray) -> np.ndarray:
        """SAM2 Automatic Mask Generator로 잎 마스크 생성."""
        H, W = rgb_np.shape[:2]
        total_pixels = H * W

        masks_data = self.generator.generate(rgb_np)
        leaf_mask = np.zeros((H, W), dtype=bool)

        for m in masks_data:
            seg = m["segmentation"]  # bool [H, W]
            area = seg.sum()

            # 크기 필터
            area_ratio = area / total_pixels
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            # 녹색 비율 필터
            if not self._is_leaf_mask(rgb_np, seg):
                continue

            leaf_mask |= seg

        return leaf_mask

    def _segment_green(self, rgb_np: np.ndarray) -> np.ndarray:
        """SAM2 없을 때: 단순 녹색 채널 기반 마스킹."""
        r = rgb_np[:, :, 0].astype(float)
        g = rgb_np[:, :, 1].astype(float)
        b = rgb_np[:, :, 2].astype(float)
        green_excess = g - np.maximum(r, b)
        return green_excess > (self.green_threshold * 255)

    def _is_leaf_mask(self, rgb_np: np.ndarray, seg: np.ndarray) -> bool:
        """마스크 영역의 녹색 비율로 잎 여부 판정."""
        region = rgb_np[seg]  # [N, 3]
        if len(region) == 0:
            return False
        r, g, b = region[:, 0].astype(float), region[:, 1].astype(float), region[:, 2].astype(float)
        green_excess = g - np.maximum(r, b)
        green_ratio = (green_excess > self.green_threshold * 255).mean()
        return green_ratio >= self.min_leaf_ratio
