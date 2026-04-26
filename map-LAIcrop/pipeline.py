"""
map-LAIcrop: 작물 군락 LAI 추정 파이프라인

lingbot-map 3D 재건 + SAM2 세그멘테이션 → LAI (직접법 / 간접법)

Usage:
    python -m map-LAIcrop.pipeline \
        --model_path ../models/lingbot-map-long.pt \
        --image_folder ./data/crop_images/ \
        --method direct \
        --output ./results/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images

from .segmentation import LeafSegmentor
from .ground import estimate_ground_plane
from .lai import compute_lai_direct, compute_lai_gap_fraction


def build_model(model_path: str, device: torch.device, use_sdpa: bool = False) -> GCTStream:
    model = GCTStream(
        img_size=518,
        patch_size=14,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        use_sdpa=use_sdpa,
        camera_num_iterations=4,
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────
    # 1. 이미지 로드
    # ──────────────────────────────────────────
    image_folder = Path(args.image_folder)
    exts = [".jpg", ".jpeg", ".png"]
    paths = sorted(p for p in image_folder.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"이미지가 없습니다: {image_folder}")

    print(f"[1/4] {len(paths)}장 이미지 로드 중...")
    images = load_and_preprocess_images(
        [str(p) for p in paths],
        mode="crop",
        image_size=518,
        patch_size=14,
    )  # [S, 3, H, W]

    # ──────────────────────────────────────────
    # 2. lingbot-map 스트리밍 3D 재건
    # ──────────────────────────────────────────
    print("[2/4] lingbot-map 3D 재건 중...")
    model = build_model(args.model_path, device, use_sdpa=args.use_sdpa)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    images_gpu = images.to(device=device, dtype=dtype)

    with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        predictions = model.inference_streaming(
            images_gpu,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=torch.device("cpu"),
        )

    # [S, H, W, 3]  world_points, [S, H, W] conf
    world_points = predictions["world_points"].squeeze(0).float()     # CPU
    world_points_conf = predictions["world_points_conf"].squeeze(0).float()
    recon_images = predictions["images"].squeeze(0)                   # [S, 3, H, W]

    del model, images_gpu, predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    S, H, W, _ = world_points.shape

    # ──────────────────────────────────────────
    # 3. SAM2 잎 세그멘테이션
    # ──────────────────────────────────────────
    print("[3/4] 잎 세그멘테이션 중...")
    segmentor = LeafSegmentor(
        sam2_checkpoint=args.sam2_checkpoint,
        device=device,
    )
    # leaf_masks: [S, H, W] bool
    leaf_masks = segmentor.segment_sequence(recon_images)

    # confidence 기반 노이즈 포인트 제거
    conf_mask = world_points_conf > args.conf_threshold  # [S, H, W]
    valid_mask = leaf_masks & conf_mask                   # [S, H, W]

    # ──────────────────────────────────────────
    # 4. LAI 계산
    # ──────────────────────────────────────────
    print("[4/4] LAI 계산 중...")

    # 잎 포인트: [N_leaf, 3]
    leaf_pts = world_points[valid_mask]

    # 전체 유효 포인트에서 지면 평면 추정
    all_valid_pts = world_points[conf_mask].numpy()
    ground_normal, ground_d = estimate_ground_plane(all_valid_pts)

    if args.method == "direct":
        lai, meta = compute_lai_direct(
            leaf_pts=leaf_pts.numpy(),
            ground_normal=ground_normal,
            ground_d=ground_d,
        )
    else:  # gap_fraction
        lai, meta = compute_lai_gap_fraction(
            leaf_masks=leaf_masks.numpy(),
            extinction_coeff=args.extinction_coeff,
        )

    # ──────────────────────────────────────────
    # 결과 저장
    # ──────────────────────────────────────────
    result = {
        "lai": float(lai),
        "method": args.method,
        "num_frames": S,
        "num_leaf_points": int(leaf_pts.shape[0]),
        **meta,
    }

    result_path = output_dir / "lai_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    np.save(output_dir / "leaf_points.npy", leaf_pts.numpy())
    np.save(output_dir / "leaf_masks.npy", leaf_masks.numpy())

    print(f"\n{'─'*40}")
    print(f"  LAI ({args.method}): {lai:.4f}")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"  결과 저장: {output_dir}")
    print(f"{'─'*40}")

    return result


def parse_args():
    p = argparse.ArgumentParser(description="map-LAIcrop: 작물 군락 LAI 추정")
    p.add_argument("--model_path", required=True, help="lingbot-map 체크포인트 경로")
    p.add_argument("--image_folder", required=True, help="입력 이미지 폴더")
    p.add_argument("--sam2_checkpoint", default=None,
                   help="SAM2 체크포인트 경로 (없으면 자동 다운로드)")
    p.add_argument("--output", default="./results", help="결과 저장 폴더")
    p.add_argument("--method", choices=["direct", "gap_fraction"], default="direct",
                   help="LAI 추정 방법: direct (직접법) / gap_fraction (간접법)")
    p.add_argument("--num_scale_frames", type=int, default=8,
                   help="Phase 1 scale frames 수")
    p.add_argument("--keyframe_interval", type=int, default=1,
                   help="KV cache keyframe 간격 (메모리 절약 시 2 이상)")
    p.add_argument("--conf_threshold", type=float, default=1.5,
                   help="포인트 신뢰도 필터링 임계값")
    p.add_argument("--extinction_coeff", type=float, default=0.5,
                   help="Beer-Lambert 소광계수 k (간접법 전용, 기본 0.5)")
    p.add_argument("--use_sdpa", action="store_true",
                   help="FlashInfer 없이 SDPA 백엔드 사용")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
