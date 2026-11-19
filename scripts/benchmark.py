"""Computational cost of the full pipeline, per study (= 4 standard views).

Reports, for each stage and for the end-to-end system:
  * parameter count and size on disk
  * peak GPU memory during inference
  * latency (mean +- sd over repeated runs, after warm-up)

Stages:
  preprocess   DICOM-like 16-bit array -> preprocessed 8-bit image (CPU), x4 views
  convnext     4 x (3, 1280, 1280) -> 4 x 768 embeddings
  transformer  (4, 768) -> greedy generation of the full conclusion (up to 39 tokens)

Each stage runs in its own process so GPU memory peaks are measured in isolation
(PyTorch and TensorFlow do not share an allocator).

Example:
    python scripts/benchmark.py --checkpoint trained_model.pth --transformer runs/all_seed42 --out benchmark.json
    python scripts/benchmark.py ... --device cpu          # CPU-only deployment
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def timeit(fn, n: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": float(np.mean(times)), "sd_ms": float(np.std(times)), "n": n}


def dir_size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(f.stat().st_size for f in files) / 2**20


# ---------------------------------------------------------------------- stages
def stage_preprocess(a) -> dict:
    from mammo_report.preprocessing import load_average_cdf, preprocess_array, to_uint8

    cdf = load_average_cdf()
    rng = np.random.default_rng(0)
    h, w = a.raw_size
    # synthetic breast-shaped 16-bit image: bright half-ellipse on dark background + noise
    yy, xx = np.mgrid[:h, :w]
    breast = ((yy - h / 2) / (h / 2)) ** 2 + (xx / (0.6 * w)) ** 2 < 1
    raw = (breast * rng.normal(2000, 300, (h, w)) + rng.normal(100, 20, (h, w))).clip(0, 4095).astype(np.uint16)
    one = lambda: to_uint8(preprocess_array(raw, None, cdf).image)  # noqa: E731
    t = timeit(one, a.repeats_cpu, warmup=1)
    return {"raw_image_size": [h, w], "per_image": t,
            "per_study_ms": 4 * t["mean_ms"], "note": "CPU, single thread of numpy/OpenCV; views can run in parallel"}


def stage_convnext(a) -> dict:
    import torch

    from mammo_report.convnext import INPUT_SIZE, ConvNeXtEncoder

    dev = a.device
    model = ConvNeXtEncoder.from_mmcls_checkpoint(a.checkpoint).to(dev)
    x = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE, device=dev)
    res = {"params": sum(p.numel() for p in model.parameters()),
           "checkpoint_mb": dir_size_mb(Path(a.checkpoint)),
           "weights_fp32_mb": sum(p.numel() for p in model.parameters()) * 4 / 2**20}
    for name, amp in [("fp32", False), ("fp16", True)]:
        if amp and dev == "cpu":
            continue
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        def run():
            with torch.no_grad(), torch.autocast(dev, enabled=amp):
                model(x)
            if dev == "cuda":
                torch.cuda.synchronize()
        res[name] = timeit(run, a.repeats if dev == "cuda" else a.repeats_cpu)
        if dev == "cuda":
            res[name]["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return res


def stage_transformer(a) -> dict:
    from mammo_report.generate import get_tokenizer

    get_tokenizer()  # load transformers/torch (and their CUDA DLLs) before TensorFlow initialises the GPU
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    from mammo_report.generate import greedy_generate
    from mammo_report.model import TransformerConfig, build_transformer

    run_dir = Path(a.transformer)
    cfg = TransformerConfig.load(run_dir / "model_config.json")
    with tf.device("/GPU:0" if a.device == "cuda" else "/CPU:0"):
        model = build_transformer(cfg)
        model.load_weights(str(run_dir / "weights" / "transformer")).expect_partial()
        x = np.random.default_rng(0).normal(size=(1, cfg.n_views, cfg.embed_dim)).astype("float32")
        if a.device == "cuda":
            tf.config.experimental.reset_memory_stats("GPU:0")
        t = timeit(lambda: greedy_generate(model, x, cfg.max_len, batch_size=1), a.repeats if a.device == "cuda" else a.repeats_cpu)
        ids, _ = greedy_generate(model, x, cfg.max_len, batch_size=1)
    res = {"params": int(model.count_params()), "weights_dir_mb": dir_size_mb(run_dir / "weights"),
           "weights_fp32_mb": model.count_params() * 4 / 2**20,
           "generation": t, "tokens_generated": int((ids[0] > 0).sum() - 1),
           "note": "greedy decoding, one forward pass per generated token, batch size 1"}
    if a.device == "cuda":
        res["peak_gpu_mb"] = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
    return res


def hardware() -> dict:
    info = {"os": f"{platform.system()} {platform.release()}", "cpu": platform.processor(), "python": platform.python_version()}
    try:
        import psutil

        info.update(cpu_cores=psutil.cpu_count(logical=False), ram_gb=round(psutil.virtual_memory().total / 2**30, 1))
    except ImportError:
        pass
    try:
        info["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                               "--format=csv,noheader"], text=True).strip()
    except Exception:
        info["gpu"] = None
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="ConvNeXt mmcls checkpoint")
    p.add_argument("--transformer", required=True, help="run directory written by mammo_report.train")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--repeats-cpu", type=int, default=10)
    p.add_argument("--raw-size", type=int, nargs=2, default=[4096, 3328], help="raw DICOM size (H W)")
    p.add_argument("--stage", choices=["all", "preprocess", "convnext", "transformer"], default="all")
    p.add_argument("--out", default="benchmark.json")
    a = p.parse_args(argv)

    if a.stage != "all":
        fn = {"preprocess": stage_preprocess, "convnext": stage_convnext, "transformer": stage_transformer}[a.stage]
        print(json.dumps(fn(a)))
        return

    res = {"hardware": hardware(), "device": a.device}
    for stage in ["preprocess", "convnext", "transformer"]:
        cmd = [sys.executable, __file__, "--stage", stage, *[s for s in sys.argv[1:] if s != "--stage"]]
        env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
        if a.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        outp = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if outp.returncode:
            raise RuntimeError(f"{stage} failed:\n{outp.stderr[-3000:]}")
        res[stage] = json.loads(outp.stdout.strip().splitlines()[-1])
    cn = res["convnext"].get("fp32")
    total = res["preprocess"]["per_study_ms"] + cn["mean_ms"] + res["transformer"]["generation"]["mean_ms"]
    res["end_to_end_per_study"] = {
        "total_ms_fp32": total,
        "breakdown_ms": {"preprocess_4_views": res["preprocess"]["per_study_ms"], "convnext_4_views": cn["mean_ms"],
                         "transformer_generation": res["transformer"]["generation"]["mean_ms"]},
        "total_params": res["convnext"]["params"] + res["transformer"]["params"],
        "peak_gpu_mb_max_stage": max(cn.get("peak_gpu_mb", 0), res["transformer"].get("peak_gpu_mb", 0)),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["end_to_end_per_study"], indent=2))


if __name__ == "__main__":
    main()
