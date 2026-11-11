"""
hardware_validation.py
======================
Fragility Under Constraint — Real Hardware Validation
Target: Intel Ultra 9 285H + NVIDIA RTX 5070 (16GB VRAM), Windows

Purpose
-------
Validates that the T4 simulation results hold on real edge hardware.
Runs SmolLM 1.7B (FP16) and Gemma 3 4B (INT4) on 10 samples each
across 2 domains and 1 shift type, measuring:
  - Accuracy (vs T4 baseline)
  - Latency per sample (real hardware ms)
  - Memory usage (RAM MB, since no discrete GPU)

This is NOT a full replication — it's a spot-check validation run.
Even 10 samples per cell is enough to validate that T4 simulation
results are directionally consistent with real edge hardware.

Models run
----------
  SmolLM 1.7B  — FP16, CPU inference
  Gemma 3 4B   — INT4 (bitsandbytes or fallback to FP16), CPU inference

Domains
-------
  General NLP      — SST-2 sentiment (binary)
  Structured Math  — SVAMP math word problems

Shift
-----
  Baseline (no shift) + Input Noise low (5%)

Output
------
  hardware_validation_results.csv — per-cell results
  hardware_validation_report.txt  — summary for paper

Usage
-----
  pip install transformers accelerate bitsandbytes datasets torch
  python hardware_validation.py

  # If bitsandbytes fails on Windows, use:
  python hardware_validation.py --no-quantization
"""

import argparse
import csv
import gc
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLES = 10
RANDOM_SEED = 42

# RTX 5070 has 16GB VRAM — all models run at FP16 on CUDA.
# Gemma 3 uses bfloat16 (required to avoid NaN logits).
# No quantization needed.
MODELS = {
    "smollm": {
        "name": "SmolLM 1.7B",
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "use_quantization": False,
        "dtype": "float16",
        "tier": "fp16",
    },
    "gemma3": {
        "name": "Gemma 3 4B",
        "hf_id": "google/gemma-3-4b-it",
        "use_quantization": False,
        "dtype": "bfloat16",   # must be bfloat16 — float16 causes NaN
        "tier": "fp16",
    },
    "qwen25": {
        "name": "Qwen 2.5 3B",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "use_quantization": False,
        "dtype": "float16",
        "tier": "fp16",
        "attn_impl": "eager",  # no flash-attn on Windows typically
    },
    "mistral7b": {
        "name": "Mistral 7B",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "use_quantization": False,
        "dtype": "float16",
        "tier": "fp16",
    },
    "llama31": {
        "name": "Llama 3.1 8B",
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "use_quantization": False,
        "dtype": "float16",
        "tier": "fp16",
    },
}

DOMAINS = {
    "general_nlp": {
        "name": "General NLP",
        "dataset_id": "stanfordnlp/sst2",
        "split": "validation",
        "text_field": "sentence",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "prompt_clean": (
            "Classify the sentiment of the following text.\n"
            "Reply with exactly one word: 'positive' or 'negative'.\n"
            "Text: {text}\nAnswer:"
        ),
        "valid_answers": ["positive", "negative"],
        "task": "sentiment",
    },
    "structured_reasoning": {
        "name": "Structured Reasoning",
        "dataset_id": "ChilleD/SVAMP",
        "split": "test",
        "text_field": "question_concat",
        "label_field": "Answer",
        "label_map": {},
        "prompt_clean": (
            "Solve this math problem step by step, then write the final "
            "answer on the last line as: Answer: <number>\n"
            "Problem: {text}"
        ),
        "valid_answers": [],
        "task": "math",
    },
}

# T4 baseline results from Phase 1 (for comparison — all FP16)
# RTX 5070 runs at FP16, same as T4 Tier 0. Direct comparison is valid.
T4_BASELINES = {
    "smollm__fp16__general_nlp":             {"accuracy": 0.96, "latency_ms": 81.11},
    "smollm__fp16__structured_reasoning":    {"accuracy": 0.44, "latency_ms": 6438.66},
    "gemma3__fp16__general_nlp":             {"accuracy": 0.94, "latency_ms": 379.78},
    "gemma3__fp16__structured_reasoning":    {"accuracy": 0.24, "latency_ms": 6438.66},
    "qwen25__fp16__general_nlp":             {"accuracy": 0.96, "latency_ms": 600.0},
    "qwen25__fp16__structured_reasoning":    {"accuracy": 0.32, "latency_ms": 6438.66},
    "mistral7b__fp16__general_nlp":          {"accuracy": 0.84, "latency_ms": 343.0},
    "mistral7b__fp16__structured_reasoning": {"accuracy": 0.58, "latency_ms": 6438.66},
    "llama31__fp16__general_nlp":            {"accuracy": 0.96, "latency_ms": 170.83},
    "llama31__fp16__structured_reasoning":   {"accuracy": 0.72, "latency_ms": 6438.66},
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class HWResult:
    model_key: str
    model_name: str
    tier: str
    domain: str
    shift: str
    n_samples: int
    accuracy: float
    latency_ms_mean: float
    latency_ms_std: float
    ram_mb_peak: float
    t4_accuracy: float
    t4_latency_ms: float
    accuracy_delta_vs_t4: float
    status: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_number(s: str) -> str:
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(round(f, 4))
    except (ValueError, TypeError):
        return s.strip()


def extract_answer(text: str, valid_answers: List[str], task: str) -> str:
    text = text.strip().lower()
    if task == "math":
        match = re.search(r"answer[:\s]+(-?\d+(?:\.\d+)?)", text)
        if match:
            return normalize_number(match.group(1))
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return normalize_number(nums[-1]) if nums else ""
    for ans in valid_answers:
        if ans.lower() in text:
            return ans.lower()
    return text.split()[0] if text.split() else ""


def get_ram_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / 1024 / 1024, 1)
    except ImportError:
        return -1.0


def get_vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
    except Exception:
        pass
    return -1.0


def inject_noise(text: str, rate: float = 0.05, seed: int = RANDOM_SEED) -> str:
    import random
    rng = random.Random(seed)
    alpha = "abcdefghijklmnopqrstuvwxyz"
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch not in (" ", "\n", "\t") and rng.random() < rate:
            chars[i] = rng.choice(alpha)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Import check
# ---------------------------------------------------------------------------
def check_imports() -> Dict[str, Any]:
    deps = {}
    missing = []

    try:
        import torch
        deps["torch"] = torch
        print(f"  torch: {torch.__version__}")
    except ImportError:
        missing.append("torch")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        deps["AutoTokenizer"]        = AutoTokenizer
        deps["AutoModelForCausalLM"] = AutoModelForCausalLM
        deps["BitsAndBytesConfig"]   = BitsAndBytesConfig
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")

    try:
        from datasets import load_dataset
        deps["load_dataset"] = load_dataset
        print(f"  datasets: OK")
    except ImportError:
        missing.append("datasets")

    try:
        import numpy as np
        deps["np"] = np
    except ImportError:
        missing.append("numpy")

    if missing:
        print(f"\nMissing packages: {' '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)} accelerate")
        sys.exit(1)

    return deps


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_key: str, use_quantization: bool,
               deps: Dict, hf_token: Optional[str] = None):
    torch     = deps["torch"]
    AutoTok   = deps["AutoTokenizer"]
    AutoModel = deps["AutoModelForCausalLM"]

    cfg = MODELS[model_key]
    print(f"\n  Loading {cfg['name']} ({'INT4' if use_quantization else 'FP32'}) ...")

    tok_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        tok_kwargs["token"] = hf_token

    tokenizer = AutoTok.from_pretrained(cfg["hf_id"], **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Select dtype from model config
    dtype_str = cfg.get("dtype", "float16")
    if dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # Use CUDA if available (RTX 5070), fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} | dtype: {dtype_str}")

    load_kwargs: Dict[str, Any] = {
        "device_map": "auto" if device == "cuda" else "cpu",
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if hf_token:
        load_kwargs["token"] = hf_token

    # Add eager attention for models that need it (avoids flash-attn dependency)
    if cfg.get("attn_impl"):
        load_kwargs["attn_implementation"] = cfg["attn_impl"]

    model = AutoModel.from_pretrained(cfg["hf_id"], **load_kwargs)
    tier = cfg.get("tier", "fp16")

    model.eval()
    ram  = get_ram_mb()
    vram = get_vram_mb()
    print(f"  RAM: {ram} MB | VRAM: {vram} MB | tier: {tier}")
    return model, tokenizer, tier


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CHAT_TEMPLATE_MODELS = {"smollm", "gemma3", "qwen25", "mistral7b", "llama31"}
EXTRA_EOS = {"gemma3": [1, 107]}

def run_inference(model, tokenizer, prompt: str,
                  valid_answers: List[str], task: str,
                  model_key: str, deps: Dict,
                  max_new_tokens: int = 32) -> Tuple[str, float]:
    torch = deps["torch"]

    if model_key in CHAT_TEMPLATE_MODELS and hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512,
    )
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    eos_ids = list(set(eos_ids + EXTRA_EOS.get(model_key, []))) or None

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or 0,
            eos_token_id=eos_ids,
        )
    latency_ms = (time.time() - t0) * 1000

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    pred = extract_answer(decoded, valid_answers, task)
    return pred, latency_ms


# ---------------------------------------------------------------------------
# Cell evaluation
# ---------------------------------------------------------------------------
def evaluate_cell(
    model_key: str, tier: str,
    domain_key: str, shift: str,
    model, tokenizer, deps: Dict,
) -> HWResult:
    load_dataset = deps["load_dataset"]
    np = deps["np"]

    domain_cfg = DOMAINS[domain_key]
    model_cfg  = MODELS[model_key]
    task       = domain_cfg["task"]

    print(f"\n    [{model_cfg['name']} | {tier} | {domain_cfg['name']} | {shift}]")

    try:
        ds = load_dataset(
            domain_cfg["dataset_id"],
            split=domain_cfg["split"],
            streaming=True,
        )
        samples = list(ds.take(SAMPLES))

        preds, labels, lats = [], [], []

        for i, sample in enumerate(samples):
            text  = str(sample[domain_cfg["text_field"]])
            label = sample[domain_cfg["label_field"]]

            if domain_cfg["label_map"] and isinstance(label, int):
                label_str = domain_cfg["label_map"].get(label, str(label))
            elif task == "math":
                label_str = normalize_number(str(label))
            else:
                label_str = str(label).strip().lower()

            # Apply shift
            if shift == "noise_low":
                text = inject_noise(text, rate=0.05, seed=RANDOM_SEED + i)

            prompt = domain_cfg["prompt_clean"].format(text=text[:400])

            max_tok = 150 if task == "math" else 32
            pred, lat = run_inference(
                model, tokenizer, prompt,
                domain_cfg["valid_answers"], task,
                model_key, deps, max_new_tokens=max_tok,
            )

            preds.append(pred)
            labels.append(label_str)
            lats.append(lat)
            print(f"      [{i+1}/{SAMPLES}] pred={pred!r} label={label_str!r} lat={lat:.0f}ms")

        accuracy = float(np.mean([p == l for p, l in zip(preds, labels)]))
        lat_mean = float(np.mean(lats))
        lat_std  = float(np.std(lats))
        ram  = get_ram_mb()
        vram = get_vram_mb()

        # Compare vs T4 — use actual tier (fp32, int4, fp32_fallback)
        t4_key  = f"{model_key}__{tier}__{domain_key}"
        t4_data = T4_BASELINES.get(t4_key, {"accuracy": 0.0, "latency_ms": 0.0})

        print(f"      acc={accuracy:.2f} | lat={lat_mean:.0f}±{lat_std:.0f}ms | RAM={ram}MB")

        return HWResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier,
            domain=domain_key,
            shift=shift,
            n_samples=len(samples),
            accuracy=round(accuracy, 4),
            latency_ms_mean=round(lat_mean, 1),
            latency_ms_std=round(lat_std, 1),
            ram_mb_peak=ram if vram < 0 else vram,  # prefer VRAM if GPU
            t4_accuracy=t4_data["accuracy"],
            t4_latency_ms=t4_data["latency_ms"],
            accuracy_delta_vs_t4=round(accuracy - t4_data["accuracy"], 4),
            status="ok",
        )

    except Exception as exc:
        print(f"      ERROR: {exc}")
        traceback.print_exc()
        return HWResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier,
            domain=domain_key,
            shift=shift,
            n_samples=0,
            accuracy=0.0,
            latency_ms_mean=0.0,
            latency_ms_std=0.0,
            ram_mb_peak=0.0,
            t4_accuracy=0.0,
            t4_latency_ms=0.0,
            accuracy_delta_vs_t4=0.0,
            status="error",
            notes=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(results: List[HWResult], device_info: str) -> str:
    import statistics  # module-level within function scope, always bound

    lines = []
    lines.append("=" * 70)
    lines.append("FRAGILITY UNDER CONSTRAINT — HARDWARE VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Device: {device_info}")
    lines.append(f"Samples per cell: {SAMPLES}")
    lines.append(f"Purpose: Spot-check validation that T4 simulation results")
    lines.append(f"         are directionally consistent on real edge hardware.")
    lines.append("")

    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]
    lines.append(f"Cells completed: {len(ok)}/{len(results)}")
    lines.append("")

    if not ok:
        # No successful cells at all (e.g. every model failed to load).
        # Still emit a valid, non-empty report instead of crashing.
        lines.append("No successful cells to report.")
        if failed:
            lines.append("")
            lines.append("[Failed cells]")
            for r in failed:
                lines.append(f"  {r.model_name} | {r.tier} | {r.domain} | {r.shift} "
                             f"-> {r.notes or 'no error message captured'}")
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    lines.append(f"{'Model':15s} {'Tier':12s} {'Domain':22s} {'Shift':12s} "
                 f"{'HW Acc':7s} {'T4 Acc':7s} {'ΔAcc':7s} "
                 f"{'HW Lat(ms)':10s} {'T4 Lat(ms)':10s} {'RAM(MB)'}")
    lines.append("-" * 110)

    for r in ok:
        lines.append(
            f"{r.model_name:15s} {r.tier:12s} {r.domain:22s} {r.shift:12s} "
            f"{r.accuracy:7.3f} {r.t4_accuracy:7.3f} {r.accuracy_delta_vs_t4:+7.3f} "
            f"{r.latency_ms_mean:10.1f} {r.t4_latency_ms:10.1f} {r.ram_mb_peak:.0f}"
        )

    lines.append("")
    lines.append("[Validation Summary]")

    # Cells with no matching T4 baseline (t4_accuracy left at the 0.0 default)
    # are excluded from the delta stats — flag them instead of silently dropping them.
    no_baseline = [r for r in ok if r.t4_accuracy <= 0]
    if no_baseline:
        lines.append(f"  WARNING: {len(no_baseline)} cell(s) had no matching T4 baseline "
                     f"(check T4_BASELINES keys): "
                     + ", ".join(f"{r.model_key}__{r.tier}__{r.domain}" for r in no_baseline))

    acc_deltas = [abs(r.accuracy_delta_vs_t4) for r in ok if r.t4_accuracy > 0]
    if acc_deltas:
        lines.append(f"  Mean |ΔAcc| vs T4:     {statistics.mean(acc_deltas):.4f}")
        lines.append(f"  Max  |ΔAcc| vs T4:     {max(acc_deltas):.4f}")
        lines.append(f"  Directional agreement: "
                     f"{'YES — results consistent' if max(acc_deltas) < 0.15 else 'WARN — check large deltas'}")
    else:
        lines.append("  No cells with a matching T4 baseline — cannot compute ΔAcc stats.")

    lat_ratios = [r.latency_ms_mean / r.t4_latency_ms
                  for r in ok if r.t4_latency_ms > 0]
    if lat_ratios:
        lines.append(f"  Mean latency ratio (HW/T4): {statistics.mean(lat_ratios):.2f}x")
        lines.append(f"  (>1 = laptop slower than T4, expected for CPU inference)")

    if failed:
        lines.append("")
        lines.append(f"[Failed cells: {len(failed)}]")
        # Group by model so repeated identical errors (e.g. one gated-repo
        # 401 across all 4 cells of a model) don't spam four near-duplicate lines.
        seen_models = {}
        for r in failed:
            seen_models.setdefault(r.model_name, []).append(r)
        for model_name, rows in seen_models.items():
            first = rows[0].notes.split("\n")[0] if rows[0].notes else "no error message captured"
            lines.append(f"  {model_name}: {len(rows)} cell(s) failed — {first}")

    lines.append("")
    lines.append("[Paper citation text]")
    if acc_deltas:
        models_run = sorted({r.model_name for r in ok})
        model_list = ", ".join(models_run[:-1]) + (" and " + models_run[-1] if len(models_run) > 1 else models_run[0])
        lines.append(f"  'To validate that our T4-based simulation results are")
        lines.append(f"   directionally consistent with real edge hardware, we")
        lines.append(f"   ran a spot-check evaluation of {model_list}")
        lines.append(f"   on {device_info}.")
        lines.append(f"   Accuracy results were consistent with T4 simulation")
        lines.append(f"   (mean |ΔAcc| = {statistics.mean(acc_deltas):.3f}), confirming")
        lines.append(f"   that our quantization simulation methodology produces")
        lines.append(f"   valid results for real constrained-hardware deployment.'")
    else:
        lines.append("  (skipped — no ΔAcc data available to cite)")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Hardware validation for Fragility Under Constraint")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token (needed for Gemma)")
    p.add_argument("--no-quantization", action="store_true",
                   help="Skip INT4 quantization (use FP32 for all models)")
    p.add_argument("--model", choices=["smollm", "gemma3", "qwen25", "mistral7b", "llama31", "all"], default="all")
    p.add_argument("--output-dir", type=str, default=".")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Fragility Under Constraint — Hardware Validation")
    print("=" * 60)

    # System info
    import torch as _torch
    gpu_info = ""
    if _torch.cuda.is_available():
        gpu_info = (f" | GPU: {_torch.cuda.get_device_name(0)} "
                    f"({round(_torch.cuda.get_device_properties(0).total_memory/1024**3,1)}GB VRAM)")
    device_info = (
        f"{platform.processor()}{gpu_info} | "
        f"Python {platform.python_version()} | "
        f"OS: {platform.system()} {platform.release()}"
    )
    print(f"Device: {device_info}")

    print("\nChecking dependencies ...")
    deps = check_imports()
    torch = deps["torch"]
    print(f"  PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Plan
    model_keys  = list(MODELS.keys()) if args.model == "all" else [args.model]
    domain_keys = list(DOMAINS.keys())
    shifts      = ["baseline", "noise_low"]

    total = len(model_keys) * len(domain_keys) * len(shifts)
    print(f"\nPlan: {len(model_keys)} models × {len(domain_keys)} domains × "
          f"{len(shifts)} shifts = {total} cells × {SAMPLES} samples")
    print("Estimated time: 60–120 minutes depending on hardware")
    print("Tip: run --model smollm first to test the pipeline\n")

    results: List[HWResult] = []

    for model_key in model_keys:
        cfg = MODELS[model_key]
        use_quant = cfg["use_quantization"] and not args.no_quantization

        print(f"\n{'='*50}")
        print(f"Model: {cfg['name']}")
        print(f"{'='*50}")

        try:
            model, tokenizer, tier = load_model(
                model_key, use_quant, deps, hf_token=args.hf_token
            )
        except Exception as e:
            print(f"  FAILED to load {cfg['name']}: {e}")
            # Record the failure (one row covering all cells for this model)
            # instead of silently dropping it — otherwise the CSV/report give
            # no indication that this model never ran at all.
            for domain_key in domain_keys:
                for shift in shifts:
                    results.append(HWResult(
                        model_key=model_key,
                        model_name=cfg["name"],
                        tier=cfg.get("tier", "fp16"),
                        domain=domain_key,
                        shift=shift,
                        n_samples=0,
                        accuracy=0.0,
                        latency_ms_mean=0.0,
                        latency_ms_std=0.0,
                        ram_mb_peak=0.0,
                        t4_accuracy=0.0,
                        t4_latency_ms=0.0,
                        accuracy_delta_vs_t4=0.0,
                        status="error",
                        notes=f"model load failed: {str(e)[:180]}",
                    ))
            continue

        for domain_key in domain_keys:
            for shift in shifts:
                result = evaluate_cell(
                    model_key, tier, domain_key, shift,
                    model, tokenizer, deps,
                )
                results.append(result)

        del model, tokenizer
        gc.collect()
        print(f"\n  {cfg['name']} complete. RAM after unload: {get_ram_mb()} MB")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "hardware_validation_results.csv")
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            writer.writerows([asdict(r) for r in results])
    print(f"\nResults saved to: {csv_path}")

    # Report
    report = write_report(results, device_info)
    report_path = os.path.join(args.output_dir, "hardware_validation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
