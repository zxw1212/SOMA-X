from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from soma.bones_smplx import (  # noqa: E402
    convert_bvh_to_smplx_direct,
    iter_bvh_files,
    relative_output_path,
    save_conversion_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BONES-SEED SOMA BVH motions to SMPL-X body parameters."
    )
    parser.add_argument("input", help="Input BVH file or directory.")
    parser.add_argument("output", help="Output .npz path or output directory.")
    parser.add_argument(
        "--model-path",
        default=str(repo_root / "assets" / "SMPLX" / "SMPLX_NEUTRAL.npz"),
        help="Path to SMPL-X model file.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device. Default: cpu.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Subsample frames by this stride before conversion. Default: 1.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of processed frames after subsampling.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Only process the first N BVH files when input is a directory.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="Gender metadata written into the exported SMPL-X NPZ. Default: neutral.",
    )
    parser.add_argument(
        "--direct-tpose-frame",
        type=int,
        default=None,
        help="Optional absolute BVH frame index used as the T-pose calibration frame. "
        "If omitted, the script estimates body axes from a reference frame and constructs a synthetic standard T-pose.",
    )
    parser.add_argument(
        "--calibration-bvh",
        type=str,
        default=None,
        help="Optional external BVH used only for calibration. "
        "When provided, the input BVH supplies the motion, while this BVH supplies the calibration pose.",
    )
    parser.add_argument(
        "--calibration-bvh-frame",
        type=int,
        default=None,
        help="Optional frame index inside --calibration-bvh used as the calibration pose. "
        "If omitted, a synthetic standard T-pose is built from the calibration BVH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    files = list(iter_bvh_files(input_path))
    if args.limit_files is not None:
        files = files[: args.limit_files]

    if not files:
        raise FileNotFoundError(f"No BVH files found under {input_path}")

    if input_path.is_file() and output_path.suffix.lower() != ".npz":
        raise ValueError("When input is a single BVH file, output must be a .npz file path.")

    for index, bvh_path in enumerate(files, start=1):
        out_file = relative_output_path(bvh_path, input_path, output_path)
        print(f"[{index}/{len(files)}] Converting {bvh_path} -> {out_file}")
        result = convert_bvh_to_smplx_direct(
            bvh_path,
            model_path=args.model_path,
            device=args.device,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            calibration_frame=args.direct_tpose_frame,
            calibration_bvh_path=args.calibration_bvh,
            calibration_bvh_frame=args.calibration_bvh_frame,
        )
        save_conversion_result(
            result,
            out_file,
            gender=args.gender,
            export_z_up=True,
            model_path=args.model_path,
            device=args.device,
            export_refine_iters=0,
        )


if __name__ == "__main__":
    main()
