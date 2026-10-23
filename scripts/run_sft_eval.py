"""SFT and evaluation in one resumable command: data -> base answers -> SFT -> SFT answers -> verify -> report.

Written for Apertus 8B with QLoRA on one 16 GB GPU (e.g. RTX 4080 Super), but it takes any config:

    python scripts/run_sft_eval.py --check     # a few minutes after the download: 3 SFT steps, 4 answers
    python scripts/run_sft_eval.py             # the full run on configs/apertus8b_16gb.yaml

Finished steps are skipped and SFT resumes from its last checkpoint, so after an interruption run the same
command again. At the end the results table, training log, verified answers, GPU and timings are copied to
--out (default results/8b), which is small enough to commit.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")])),
    "PYTHONUTF8": "1",                                   # the data is UTF-8; Windows defaults to cp1252
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # less fragmentation on 16 GB GPUs
}


def environment(cfg: dict, allow_cpu: bool) -> dict:
    """Check the GPU and libraries before spending an hour on a download, and record them with the results."""
    import torch

    info = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("torch", "transformers", "trl", "peft", "bitsandbytes"):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info[pkg] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_memory_gb=round(props.total_memory / 2**30, 1),
                    bf16=torch.cuda.get_device_capability()[0] >= 8)
        if info["gpu_memory_gb"] < 15:
            print(f"warning: {info['gpu']} has {info['gpu_memory_gb']} GB; this config is sized for 16 GB")
    elif not allow_cpu:
        sys.exit("No CUDA GPU visible to PyTorch. Check `nvidia-smi` (inside WSL2 if you run there) and that "
                 "torch is a CUDA build: python -c \"import torch; print(torch.cuda.is_available())\"")
    if cfg["model"].get("load_in_4bit") and not info["bitsandbytes"]:
        sys.exit('This config loads the model in 4 bits, which needs bitsandbytes: pip install -e ".[qlora]"')
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/apertus8b_16gb.yaml")
    parser.add_argument("--out", default="results/8b", help="Where the results to commit are copied")
    parser.add_argument("--check", action="store_true",
                        help="Quick end-to-end check (3 SFT steps, 4 test answers) in a throwaway run directory")
    parser.add_argument("--allow-cpu", action="store_true", help="Skip the GPU check (for testing this script)")
    args = parser.parse_args()

    config = Path(args.config).resolve()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    info = environment(cfg, args.allow_cpu)

    sft = Path(cfg["sft"]["output_dir"])
    runs = sft.parent
    overrides = []
    if args.check:
        runs = runs.with_name(runs.name + "_check")
        sft = runs / "sft"
        shutil.rmtree(REPO / runs, ignore_errors=True)
        overrides = ["--set", f"sft.output_dir={sft.as_posix()}", "--set", "sft.max_steps=3",
                     "--set", "sft.eval_steps=3", "--set", "sft.logging_steps=1"]
    preds, verified = runs / "preds", runs / "verified"
    data = Path(cfg["data"]["dir"])
    limit = ["--limit", "4"] if args.check else []
    report = runs / "results.md" if args.check else Path(args.out) / "results.md"

    timings_file = REPO / runs / "timings.json"
    timings = json.loads(timings_file.read_text()) if timings_file.exists() else {}

    def step(name: str, module_args: list[str], done: list[Path]) -> None:
        if done and all((REPO / p).exists() for p in done):
            print(f"skip, already done: {name}")
            return
        print(f"\n=== {name} ===\n$ python -m {' '.join(module_args)}", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, "-m", *module_args], cwd=REPO, env=ENV)
        if result.returncode != 0:
            sys.exit(f"\nstep '{name}' failed (exit {result.returncode}). Fix the cause and run the same command "
                     "again; finished steps are skipped and SFT resumes from its last checkpoint.")
        timings[name] = round((time.time() - start) / 60, 1)
        timings_file.parent.mkdir(parents=True, exist_ok=True)
        timings_file.write_text(json.dumps(timings, indent=2))

    c = ["--config", str(config), *overrides]
    step("prepare data", ["groundqa.prepare", *c], [data / "test.jsonl"])
    step("base answers", ["groundqa.generate", *c, *limit, "--out", str(preds / "base.jsonl")],
         [preds / "base.jsonl"])
    step("SFT", ["groundqa.train_sft", *c], [sft / "adapter_config.json", sft / "adapter_model.safetensors"])
    step("SFT answers", ["groundqa.generate", *c, *limit, "--adapter", str(sft), "--out", str(preds / "sft.jsonl")],
         [preds / "sft.jsonl"])
    for name in ("base", "sft"):
        step(f"verify {name}", ["groundqa.verify", *c, "--preds", str(preds / f"{name}.jsonl"),
                                "--out", str(verified / f"{name}.jsonl")], [verified / f"{name}.jsonl"])
    step("report", ["groundqa.report", "--run", f"base={verified / 'base.jsonl'}",
                    "--run", f"sft={verified / 'sft.jsonl'}", "--out", str(report)], [])

    if args.check:
        print(f"\ncheck passed ({sum(timings.values()):.0f} min). Look at the peak GPU memory printed after SFT, "
              "then start the full run without --check.")
        return

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / sft / "log_history.json", out / "sft_log_history.json")
    for name in ("base", "sft"):
        shutil.copy(REPO / verified / f"{name}.jsonl", out / f"verified_{name}.jsonl")
    info.update(config=Path(os.path.relpath(config, REPO)).as_posix(), minutes=timings,
                finished=time.strftime("%Y-%m-%d %H:%M"))
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print("\n" + (out / "results.md").read_text(encoding="utf-8"))
    print(f"Results are in {args.out}. To share them:\n"
          f"  git add {args.out} && git commit -m \"Apertus 8B SFT results\" && git push")


if __name__ == "__main__":
    main()
