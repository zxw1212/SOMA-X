from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp
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
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for directory batch conversion on CPU. Default: 1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing non-empty target .npz files so batch conversion can resume from a previous run.",
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


def _convert_one_file(
    *,
    bvh_path: str,
    out_file: str,
    model_path: str,
    device: str,
    frame_stride: int,
    max_frames: int | None,
    calibration_frame: int | None,
    calibration_bvh_path: str | None,
    calibration_bvh_frame: int | None,
    gender: str,
) -> tuple[str, str]:
    result = convert_bvh_to_smplx_direct(
        bvh_path,
        model_path=model_path,
        device=device,
        frame_stride=frame_stride,
        max_frames=max_frames,
        calibration_frame=calibration_frame,
        calibration_bvh_path=calibration_bvh_path,
        calibration_bvh_frame=calibration_bvh_frame,
    )
    save_conversion_result(
        result,
        out_file,
        gender=gender,
        export_z_up=True,
        model_path=model_path,
        device=device,
        export_refine_iters=0,
    )
    return bvh_path, out_file


def _init_worker() -> None:
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only allows setting this once per process.
            pass
    except Exception:
        pass


def _submit_parallel_job(
    executor: ProcessPoolExecutor,
    item: tuple[int, Path, Path],
    args: argparse.Namespace,
):
    index, bvh_path, out_file = item
    future = executor.submit(
        _convert_one_file,
        bvh_path=str(bvh_path),
        out_file=str(out_file),
        model_path=str(args.model_path),
        device=str(args.device),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        calibration_frame=args.direct_tpose_frame,
        calibration_bvh_path=args.calibration_bvh,
        calibration_bvh_frame=args.calibration_bvh_frame,
        gender=args.gender,
    )
    return future, (index, bvh_path, out_file)


def _should_skip_output(out_file: Path) -> bool:
    return out_file.is_file() and out_file.stat().st_size > 0


def _format_failure_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _write_failure_log(output_path: Path, failures: list[tuple[Path, str]]) -> Path:
    log_dir = output_path if output_path.suffix.lower() != ".npz" else output_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "_conversion_failures.txt"
    lines = [f"{path}\t{message}" for path, message in failures]
    log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return log_path


def main() -> None:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError(f"--num-workers must be > 0, got {args.num_workers}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    files = list(iter_bvh_files(input_path))
    if args.limit_files is not None:
        files = files[: args.limit_files]

    if not files:
        raise FileNotFoundError(f"No BVH files found under {input_path}")

    if input_path.is_file() and output_path.suffix.lower() != ".npz":
        raise ValueError("When input is a single BVH file, output must be a .npz file path.")

    all_work_items = [
        (bvh_path, relative_output_path(bvh_path, input_path, output_path))
        for bvh_path in files
    ]
    if args.resume:
        skipped_items = [item for item in all_work_items if _should_skip_output(item[1])]
        pending_items = [item for item in all_work_items if not _should_skip_output(item[1])]
        if skipped_items:
            print(
                f"Skipping {len(skipped_items)} existing non-empty output files; "
                f"{len(pending_items)} files remain to convert.",
                flush=True,
            )
        if not pending_items:
            print("Nothing to do: all target npz files already exist.", flush=True)
            return
    else:
        pending_items = all_work_items

    work_items = [
        (index, bvh_path, out_file) for index, (bvh_path, out_file) in enumerate(pending_items, start=1)
    ]

    use_parallel = len(work_items) > 1 and args.num_workers > 1
    if use_parallel and str(args.device).lower() != "cpu":
        print(f"Using 1 worker because --num-workers>1 is only enabled for --device cpu, got {args.device}.")
        use_parallel = False

    if use_parallel:
        work_iter = iter(work_items)
        in_flight: dict[object, tuple[int, Path, Path]] = {}
        failures: list[tuple[Path, str]] = []
        spawn_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=spawn_ctx,
            initializer=_init_worker,
        ) as executor:
            while len(in_flight) < args.num_workers:
                try:
                    item = next(work_iter)
                except StopIteration:
                    break
                future, future_item = _submit_parallel_job(executor, item, args)
                in_flight[future] = future_item
                index, bvh_path, out_file = future_item
                print(f"[{index}/{len(work_items)}] Converting {bvh_path} -> {out_file}", flush=True)

            completed = 0
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    index, bvh_path, out_file = in_flight.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        message = _format_failure_message(exc)
                        failures.append((bvh_path, message))
                        print(
                            f"[fail {len(failures)}/{len(work_items)}] {bvh_path} -> {out_file}: {message}",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        completed += 1
                        print(f"[{completed}/{len(work_items)}] Converted {bvh_path} -> {out_file}", flush=True)

                    try:
                        item = next(work_iter)
                    except StopIteration:
                        continue
                    next_future, next_item = _submit_parallel_job(executor, item, args)
                    in_flight[next_future] = next_item
                    next_index, next_bvh_path, next_out_file = next_item
                    print(
                        f"[{next_index}/{len(work_items)}] Converting {next_bvh_path} -> {next_out_file}",
                        flush=True,
                    )
        if failures:
            log_path = _write_failure_log(output_path, failures)
            print(
                f"Finished with {completed} converted and {len(failures)} failed files. "
                f"Failure log: {log_path}",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)
        return

    failures: list[tuple[Path, str]] = []
    for index, bvh_path, out_file in work_items:
        print(f"[{index}/{len(files)}] Converting {bvh_path} -> {out_file}", flush=True)
        try:
            _convert_one_file(
                bvh_path=str(bvh_path),
                out_file=str(out_file),
                model_path=str(args.model_path),
                device=str(args.device),
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                calibration_frame=args.direct_tpose_frame,
                calibration_bvh_path=args.calibration_bvh,
                calibration_bvh_frame=args.calibration_bvh_frame,
                gender=args.gender,
            )
        except Exception as exc:
            message = _format_failure_message(exc)
            failures.append((bvh_path, message))
            print(
                f"[fail {len(failures)}/{len(work_items)}] {bvh_path} -> {out_file}: {message}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"[ok] Converted {bvh_path} -> {out_file}", flush=True)

    if failures:
        log_path = _write_failure_log(output_path, failures)
        print(
            f"Finished with {len(work_items) - len(failures)} converted and {len(failures)} failed files. "
            f"Failure log: {log_path}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
