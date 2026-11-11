<<<<<<< HEAD
"""
feasibility_check.py  v3  â€” FINAL
===================================
Fragility Under Constraint â€” Pre-experiment Feasibility Checker
Target hardware: Kaggle T4 (15 GB VRAM)

Changes in v3
-------------
- Dataset fixes (all confirmed working):
    financial_phrasebank -> zeroshot/twitter-financial-news-sentiment
                           (Twitter financial news sentiment; fields: text, label)
    tabmwp               -> Arietem/tabmwp, split="train"
                           (test split embedded in data; filter by sample['split']=='test')
- Phi-3 fix:
    Added explicit fallback: if 'type' KeyError on first load attempt,
    retry with low_cpu_mem_usage=True and no custom config keys
- All other v2 fixes retained:
    dtype= instead of torch_dtype=
    attn_implementation='eager' for Phi-3 and Qwen
    RuntimeError caught alongside OutOfMemoryError
    gc.collect() on every flush

Usage
-----
python feasibility_check.py                          # full run
python feasibility_check.py --skip-smoke             # audit only (faster)
python feasibility_check.py --model phi3             # single model
python feasibility_check.py --output my_report.json  # custom path
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("feasibility")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VRAM_LIMIT_GB = 14.0
SAFE_FP16_GB  = 12.0
SMOKE_SAMPLES = 20

# Models needing eager attention (no flash-attn on Kaggle)
EAGER_ATTN_MODELS = {"phi3", "qwen25"}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "phi3": {
        "name": "Phi-3 Mini",
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "params_b": 3.8,
    },
    "gemma3": {
        "name": "Gemma 3 4B",
        "hf_id": "google/gemma-3-4b-it",
        "params_b": 4.0,
    },
    "mistral7b": {
        "name": "Mistral 7B",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "params_b": 7.0,
    },
    "llama31": {
        "name": "Llama 3.1 8B",
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "params_b": 8.0,
    },
    "qwen25": {
        "name": "Qwen 2.5 3B",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3.0,
    },
    "smollm": {
        "name": "SmolLM 1.7B",
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params_b": 1.7,
    },
}

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- General NLP ---
    "sst2": {
        "hf_id": "stanfordnlp/sst2",
        "split": "validation",
        "domain": "General NLP",
    },
    "hellaswag": {
        "hf_id": "Rowan/hellaswag",
        "split": "validation",
        "domain": "General NLP",
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "split": "validation",
        "config": "all",
        "domain": "General NLP",
    },
    # --- Financial NLP ---
    # Confirmed working: fields are 'text' and 'label' (0=bearish,1=bullish,2=neutral)
    "financial_sentiment": {
        "hf_id": "zeroshot/twitter-financial-news-sentiment",
        "split": "train",
        "domain": "Financial NLP",
        "note": "Twitter financial news sentiment (substitute for FinancialPhraseBank). "
                "Fields: text (str), label (int: 0=bearish, 1=bullish, 2=neutral).",
    },
    "fiqa": {
        "hf_id": "BeIR/fiqa",
        "split": "corpus",
        "config": "corpus",
        "domain": "Financial NLP",
    },
    # --- Structured Reasoning ---
    "svamp": {
        "hf_id": "ChilleD/SVAMP",
        "split": "test",
        "domain": "Structured Reasoning",
    },
    # Confirmed working: load with split="train", filter by sample['split']=='test'
    "tabmwp": {
        "hf_id": "Arietem/tabmwp",
        "split": "train",        # test rows are embedded; filter sample['split']=='test'
        "domain": "Structured Reasoning",
        "note": "TabMWP test rows embedded in train split. "
                "Filter by sample['split'] == 'test' in experiment pipeline.",
    },
    "humaneval": {
        "hf_id": "openai/openai_humaneval",
        "split": "test",
        "domain": "Structured Reasoning",
    },
}

TIERS: Dict[str, Dict[str, Any]] = {
    "fp16": {"label": "Tier 0 â€” FP16", "load_in_8bit": False, "load_in_4bit": False},
    "int8": {"label": "Tier 1 â€” INT8", "load_in_8bit": True,  "load_in_4bit": False},
    "int4": {"label": "Tier 2 â€” INT4", "load_in_8bit": False, "load_in_4bit": True},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelTierResult:
    model_key: str
    model_name: str
    tier: str
    peak_vram_gb: float
    load_time_s: float
    status: str          # "ok" | "oom" | "error"
    within_limit: bool
    notes: str = ""


@dataclass
class DatasetResult:
    dataset_key: str
    hf_id: str
    domain: str
    accessible: bool
    num_rows: int
    status: str          # "ok" | "error"
    notes: str = ""


@dataclass
class SmokeTestResult:
    ran: bool
    model: str
    shift_type: str
    tier: str
    domain: str
    baseline_acc: float
    shifted_acc: float
    delta_acc: float
    ece_proxy: float
    latency_ms_per_sample: float
    status: str          # "ok" | "error"
    notes: str = ""


@dataclass
class FeasibilityReport:
    timestamp: str
    python_version: str
    cuda_available: bool
    cuda_device: str
    vram_total_gb: float
    model_results: List[ModelTierResult] = field(default_factory=list)
    dataset_results: List[DatasetResult] = field(default_factory=list)
    smoke_test: Optional[SmokeTestResult] = None
    recommendations: List[str] = field(default_factory=list)
    overall_status: str = "pending"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gb(b: int) -> float:
    return round(b / (1024 ** 3), 2)


def flush_gpu(torch_mod) -> None:
    gc.collect()
    torch_mod.cuda.empty_cache()
    torch_mod.cuda.reset_peak_memory_stats()


def get_vram_info(torch_mod) -> Tuple[float, float]:
    allocated = gb(torch_mod.cuda.memory_allocated())
    total     = gb(torch_mod.cuda.get_device_properties(0).total_memory)
    return allocated, total


# ---------------------------------------------------------------------------
# Dependency import guard
# ---------------------------------------------------------------------------
def import_dependencies() -> Dict[str, Any]:
    missing = []
    deps: Dict[str, Any] = {}

    for pkg, attr in [("torch", "torch"), ("numpy", "np")]:
        try:
            import importlib
            mod = importlib.import_module(pkg)
            deps[attr] = mod
        except ImportError:
            missing.append(pkg)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
        deps["AutoTokenizer"]        = AutoTokenizer
        deps["AutoModelForCausalLM"] = AutoModelForCausalLM
        deps["BitsAndBytesConfig"]   = BitsAndBytesConfig
        deps["AutoConfig"]           = AutoConfig
    except ImportError:
        missing.append("transformers")

    try:
        from datasets import load_dataset
        deps["load_dataset"] = load_dataset
    except ImportError:
        missing.append("datasets")

    if missing:
        raise ImportError(
            f"\n\nMissing packages: {' '.join(missing)}\n"
            f"Run: pip install {' '.join(missing)} accelerate bitsandbytes\n"
        )
    return deps


# ---------------------------------------------------------------------------
# Step 1 â€” VRAM audit
# ---------------------------------------------------------------------------
def _build_load_kwargs(
    model_key: str,
    tier_key: str,
    torch_mod,
    BnBConfig,
    hf_token: Optional[str],
    low_cpu_mem: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Build (bnb_cfg, load_kwargs) for a given model + tier."""
    tier_cfg = TIERS[tier_key]

    bnb_cfg = None
    if tier_cfg["load_in_8bit"]:
        bnb_cfg = BnBConfig(load_in_8bit=True)
    elif tier_cfg["load_in_4bit"]:
        bnb_cfg = BnBConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_mod.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb_cfg is not None:
        kwargs["quantization_config"] = bnb_cfg
    else:
        kwargs["dtype"] = torch_mod.float16

    if model_key in EAGER_ATTN_MODELS:
        kwargs["attn_implementation"] = "eager"

    if low_cpu_mem:
        kwargs["low_cpu_mem_usage"] = True

    if hf_token:
        kwargs["token"] = hf_token

    return bnb_cfg, kwargs


def audit_model_vram(
    model_key: str,
    tier_key: str,
    deps: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> ModelTierResult:

    torch      = deps["torch"]
    AutoTok    = deps["AutoTokenizer"]
    AutoModel  = deps["AutoModelForCausalLM"]
    BnBConfig  = deps["BitsAndBytesConfig"]
    AutoConfig = deps["AutoConfig"]

    model_cfg = MODEL_REGISTRY[model_key]
    tier_cfg  = TIERS[tier_key]

    log.info(f"  Loading {model_cfg['name']} @ {tier_cfg['label']} ...")
    flush_gpu(torch)
    t0 = time.time()

    try:
        tok_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            tok_kwargs["token"] = hf_token

        tokenizer = AutoTok.from_pretrained(model_cfg["hf_id"], **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _, load_kwargs = _build_load_kwargs(
            model_key, tier_key, torch, BnBConfig, hf_token
        )

        # Phi-3 fix: newer config.json uses rope_type key but modeling_phi3.py
        # expects type key. Setting rope_scaling=None uses standard RoPE
        # which works correctly for inference on T4.
        if model_key == "phi3":
            cfg = AutoConfig.from_pretrained(
                model_cfg["hf_id"], trust_remote_code=True, **tok_kwargs
            )
            if isinstance(cfg.rope_scaling, dict) and "type" not in cfg.rope_scaling:
                log.info("  Phi-3: patching rope_scaling=None (rope_type/type mismatch)")
                cfg.rope_scaling = None
            load_kwargs["config"] = cfg

        model = AutoModel.from_pretrained(model_cfg["hf_id"], **load_kwargs)

        model.eval()
        load_time = round(time.time() - t0, 1)

        # Warm-up pass for true peak VRAM
        dummy = tokenizer("Hello world", return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = model(**dummy)

        peak_vram, _ = get_vram_info(torch)
        within = peak_vram <= VRAM_LIMIT_GB

        notes = []
        if not within:
            notes.append(f"Exceeds {VRAM_LIMIT_GB} GB limit.")
        if tier_key == "fp16" and peak_vram > SAFE_FP16_GB:
            notes.append("FP16 > 12 GB â€” use INT8 as baseline tier.")

        result = ModelTierResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier_key,
            peak_vram_gb=peak_vram,
            load_time_s=load_time,
            status="ok",
            within_limit=within,
            notes=" | ".join(notes),
        )

        del model, tokenizer, dummy
        flush_gpu(torch)
        return result

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        flush_gpu(torch)
        err = str(exc)
        is_oom = "out of memory" in err.lower() or "outofmemory" in err.lower()
        status = "oom" if is_oom else "error"
        log.warning(f"  {status.upper()}: {model_cfg['name']} @ {tier_cfg['label']}: {err[:120]}")
        return ModelTierResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier_key,
            peak_vram_gb=-1.0,
            load_time_s=-1.0,
            status=status,
            within_limit=False,
            notes=err[:200],
        )

    except Exception as exc:
        flush_gpu(torch)
        log.error(f"  ERROR {model_cfg['name']} @ {tier_cfg['label']}: {exc}")
        return ModelTierResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier_key,
            peak_vram_gb=-1.0,
            load_time_s=-1.0,
            status="error",
            within_limit=False,
            notes=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# Step 2 â€” Dataset audit
# ---------------------------------------------------------------------------
def audit_dataset(
    dataset_key: str,
    deps: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> DatasetResult:

    load_dataset = deps["load_dataset"]
    cfg = DATASET_REGISTRY[dataset_key]
    log.info(f"  Checking: {cfg['hf_id']} ({cfg['split']}) ...")

    try:
        kwargs: Dict[str, Any] = {
            "split": cfg["split"],
            "streaming": True,
            "trust_remote_code": True,
        }
        if "config" in cfg:
            kwargs["name"] = cfg["config"]
        if hf_token:
            kwargs["token"] = hf_token

        ds     = load_dataset(cfg["hf_id"], **kwargs)
        sample = list(ds.take(3))

        return DatasetResult(
            dataset_key=dataset_key,
            hf_id=cfg["hf_id"],
            domain=cfg["domain"],
            accessible=True,
            num_rows=len(sample),
            status="ok",
            notes=cfg.get("note", ""),
        )

    except Exception as exc:
        log.warning(f"  FAIL {dataset_key}: {exc}")
        return DatasetResult(
            dataset_key=dataset_key,
            hf_id=cfg["hf_id"],
            domain=cfg["domain"],
            accessible=False,
            num_rows=0,
            status="error",
            notes=str(exc)[:300],
        )


# ---------------------------------------------------------------------------
# Step 3 â€” Smoke test
# ---------------------------------------------------------------------------
def run_smoke_test(
    deps: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> SmokeTestResult:
    """
    Full pipeline cell:
      Model  : Gemma 3 4B @ FP16  (most stable on T4 from audit)
      Shift  : 5% character noise
      Domain : SST-2 sentiment (binary)
    Measures: baseline acc, shifted acc, Î”Acc, ECE proxy, latency.
    """
    torch        = deps["torch"]
    AutoTok      = deps["AutoTokenizer"]
    AutoModel    = deps["AutoModelForCausalLM"]
    np           = deps["np"]
    load_dataset = deps["load_dataset"]

    log.info("Smoke test: Gemma 3 4B | FP16 | SST-2 | 5% noise ...")

    try:
        # Load SST-2
        ds = load_dataset(
            "stanfordnlp/sst2",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
        samples = list(ds.take(SMOKE_SAMPLES))
        texts  = [s["sentence"] for s in samples]
        labels = [s["label"]    for s in samples]

        # Load Gemma 3 4B FP16
        flush_gpu(torch)
        model_id = MODEL_REGISTRY["gemma3"]["hf_id"]
        tok_kw: Dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            tok_kw["token"] = hf_token

        tokenizer = AutoTok.from_pretrained(model_id, **tok_kw)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        mdl_kw: Dict[str, Any] = {
            "dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if hf_token:
            mdl_kw["token"] = hf_token

        model = AutoModel.from_pretrained(model_id, **mdl_kw)
        model.eval()

        PROMPT = (
            "Classify the sentiment as exactly 'positive' or 'negative'.\n"
            "Text: {text}\nAnswer:"
        )

        def predict(text: str) -> Tuple[int, float]:
            inp = tokenizer(
                PROMPT.format(text=text),
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to("cuda")
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[:, -1, :]
            probs  = torch.softmax(logits, dim=-1)
            pos_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
            neg_id = tokenizer.encode(" negative", add_special_tokens=False)[0]
            pos_p  = probs[0, pos_id].item()
            neg_p  = probs[0, neg_id].item()
            denom  = pos_p + neg_p + 1e-9
            pred   = 1 if pos_p >= neg_p else 0
            conf   = max(pos_p, neg_p) / denom
            return pred, conf

        def inject_noise(text: str, rate: float = 0.05) -> str:
            import random
            alpha = "abcdefghijklmnopqrstuvwxyz"
            chars = list(text)
            for i, ch in enumerate(chars):
                if ch != " " and random.random() < rate:
                    chars[i] = random.choice(alpha)
            return "".join(chars)

        # Baseline
        b_preds, b_confs, lats = [], [], []
        for t in texts:
            t0 = time.time()
            p, c = predict(t)
            lats.append((time.time() - t0) * 1000)
            b_preds.append(p)
            b_confs.append(c)
        baseline_acc = float(np.mean([p == l for p, l in zip(b_preds, labels)]))

        # Shifted
        noisy = [inject_noise(t) for t in texts]
        s_preds, s_confs = [], []
        for t in noisy:
            p, c = predict(t)
            s_preds.append(p)
            s_confs.append(c)
        shifted_acc = float(np.mean([p == l for p, l in zip(s_preds, labels)]))
        delta_acc   = round(baseline_acc - shifted_acc, 4)
        ece_proxy   = float(np.mean([
            abs(c - int(p == l))
            for c, p, l in zip(s_confs, s_preds, labels)
        ]))
        avg_lat = round(float(np.mean(lats)), 2)

        log.info(
            f"  âœ“ baseline={baseline_acc:.3f}  shifted={shifted_acc:.3f}  "
            f"Î”={delta_acc:.3f}  ECE={ece_proxy:.3f}  {avg_lat}ms/sample"
        )

        del model, tokenizer
        flush_gpu(torch)

        return SmokeTestResult(
            ran=True,
            model="Gemma 3 4B (FP16)",
            shift_type="Input Noise 5% char substitution",
            tier="fp16",
            domain="General NLP â€” SST-2",
            baseline_acc=round(baseline_acc, 4),
            shifted_acc=round(shifted_acc, 4),
            delta_acc=delta_acc,
            ece_proxy=round(ece_proxy, 4),
            latency_ms_per_sample=avg_lat,
            status="ok",
        )

    except Exception as exc:
        log.error(f"  Smoke test FAILED: {exc}")
        log.debug(traceback.format_exc())
        return SmokeTestResult(
            ran=False,
            model="Gemma 3 4B (FP16)",
            shift_type="Input Noise 5%",
            tier="fp16",
            domain="General NLP â€” SST-2",
            baseline_acc=0.0, shifted_acc=0.0, delta_acc=0.0,
            ece_proxy=0.0, latency_ms_per_sample=0.0,
            status="error",
            notes=str(exc)[:300],
        )


# ---------------------------------------------------------------------------
# Step 4 â€” Recommendations
# ---------------------------------------------------------------------------
def generate_recommendations(report: FeasibilityReport) -> List[str]:
    recs: List[str] = []

    failed_fp16 = [
        r.model_name for r in report.model_results
        if r.tier == "fp16" and r.status in ("oom", "error")
    ]
    if failed_fp16:
        recs.append(
            f"FP16 failures: {', '.join(failed_fp16)}. "
            "Check error notes â€” likely a library compatibility issue, not true OOM."
        )

    tight_fp16 = [
        r.model_name for r in report.model_results
        if r.tier == "fp16" and r.within_limit and r.peak_vram_gb > SAFE_FP16_GB
    ]
    if tight_fp16:
        recs.append(
            f"{', '.join(tight_fp16)} uses >12 GB at FP16. "
            "Treat INT8 as the Tier 0 baseline for these models."
        )

    failed_int8 = [
        r.model_name for r in report.model_results
        if r.tier == "int8" and r.status in ("oom", "error")
    ]
    if failed_int8:
        recs.append(
            f"INT8 failures: {', '.join(failed_int8)}. "
            "Try: pip install bitsandbytes --upgrade"
        )

    failed_ds = [
        r.dataset_key for r in report.dataset_results if not r.accessible
    ]
    if failed_ds:
        recs.append(
            f"Dataset failures: {', '.join(failed_ds)}. "
            "Check HuggingFace Hub for updated dataset IDs."
        )

    if report.smoke_test:
        if report.smoke_test.status == "ok":
            if report.smoke_test.delta_acc > 0.05:
                recs.append(
                    f"Smoke test Î”Acc={report.smoke_test.delta_acc:.3f} at 5% noise. "
                    "Pipeline is correctly capturing shift sensitivity."
                )
            if report.smoke_test.latency_ms_per_sample > 2000:
                est_h = round(
                    SMOKE_SAMPLES * 500 * report.smoke_test.latency_ms_per_sample
                    / 1000 / 3600, 1
                )
                recs.append(
                    f"Latency {report.smoke_test.latency_ms_per_sample}ms/sample. "
                    f"Estimated full-run time: ~{est_h}h. Use batched inference."
                )
        else:
            recs.append(f"Smoke test FAILED: {report.smoke_test.notes[:150]}")

    if not recs:
        recs.append("All checks passed. Ready for full experiment run.")

    return recs


# ---------------------------------------------------------------------------
# CLI + orchestration
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feasibility checker v3 â€” Fragility Under Constraint"
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
    )
    p.add_argument(
        "--tiers",
        nargs="+",
        choices=list(TIERS.keys()),
        default=["fp16", "int8", "int4"],
    )
    p.add_argument("--skip-datasets", action="store_true")
    p.add_argument("--skip-smoke",    action="store_true")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    p.add_argument("--output",    type=str, default="feasibility_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("Fragility Under Constraint â€” Feasibility Checker v3")
    log.info("=" * 60)

    try:
        deps = import_dependencies()
    except ImportError as e:
        log.error(str(e))
        sys.exit(1)

    torch = deps["torch"]

    if not torch.cuda.is_available():
        log.error("CUDA not available. Run on a GPU instance.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    _, vram_total = get_vram_info(torch)
    log.info(f"GPU: {device_name} | VRAM: {vram_total} GB")

    report = FeasibilityReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        python_version=sys.version.split()[0],
        cuda_available=True,
        cuda_device=device_name,
        vram_total_gb=vram_total,
    )

    # --- VRAM audit ---
    model_keys = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    log.info("\n[1/3] VRAM Audit")
    log.info("-" * 40)
    for mk in model_keys:
        log.info(f"Model: {MODEL_REGISTRY[mk]['name']}")
        for tk in args.tiers:
            r = audit_model_vram(mk, tk, deps, hf_token=args.hf_token)
            report.model_results.append(r)
            icon = "âœ“" if r.status == "ok" and r.within_limit else "âœ—"
            log.info(
                f"    {icon} {TIERS[tk]['label']:20s} | "
                f"VRAM: {r.peak_vram_gb:5.2f} GB | "
                f"Load: {r.load_time_s:5.1f}s | "
                f"Status: {r.status}"
                + (f" | {r.notes[:80]}" if r.notes else "")
            )

    # --- Dataset audit ---
    if not args.skip_datasets:
        log.info("\n[2/3] Dataset Audit")
        log.info("-" * 40)
        for dk in DATASET_REGISTRY:
            r = audit_dataset(dk, deps, hf_token=args.hf_token)
            report.dataset_results.append(r)
            icon = "âœ“" if r.accessible else "âœ—"
            log.info(
                f"  {icon} {dk:30s} | {r.domain:25s} | {r.status}"
                + (f" | {r.notes[:60]}" if r.notes else "")
            )
    else:
        log.info("\n[2/3] Dataset Audit â€” SKIPPED")

    # --- Smoke test ---
    if not args.skip_smoke:
        log.info("\n[3/3] Smoke Test")
        log.info("-" * 40)
        smoke = run_smoke_test(deps, hf_token=args.hf_token)
        report.smoke_test = smoke
        if smoke.status != "ok":
            log.info(f"  âœ— Failed: {smoke.notes}")
    else:
        log.info("\n[3/3] Smoke Test â€” SKIPPED")

    # --- Recommendations ---
    log.info("\n[Recommendations]")
    log.info("-" * 40)
    report.recommendations = generate_recommendations(report)
    for i, rec in enumerate(report.recommendations, 1):
        log.info(f"  {i}. {rec}")

    # --- Overall status ---
    has_model_fail = any(
        r.status in ("oom", "error") for r in report.model_results
    )
    has_ds_fail    = any(not r.accessible for r in report.dataset_results)
    smoke_failed   = report.smoke_test and report.smoke_test.status == "error"

    if smoke_failed or (has_model_fail and has_ds_fail):
        report.overall_status = "fail"
    elif has_model_fail or has_ds_fail:
        report.overall_status = "warn"
    else:
        report.overall_status = "pass"

    log.info(f"\nOverall status: {report.overall_status.upper()}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    log.info(f"Report saved to: {args.output}")
    log.info("=" * 60)


if __name__ == "__main__":
=======
"""
feasibility_check.py  v3  â€” FINAL
===================================
Fragility Under Constraint â€” Pre-experiment Feasibility Checker
Target hardware: Kaggle T4 (15 GB VRAM)

Changes in v3
-------------
- Dataset fixes (all confirmed working):
    financial_phrasebank -> zeroshot/twitter-financial-news-sentiment
                           (Twitter financial news sentiment; fields: text, label)
    tabmwp               -> Arietem/tabmwp, split="train"
                           (test split embedded in data; filter by sample['split']=='test')
- Phi-3 fix:
    Added explicit fallback: if 'type' KeyError on first load attempt,
    retry with low_cpu_mem_usage=True and no custom config keys
- All other v2 fixes retained:
    dtype= instead of torch_dtype=
    attn_implementation='eager' for Phi-3 and Qwen
    RuntimeError caught alongside OutOfMemoryError
    gc.collect() on every flush

Usage
-----
python feasibility_check.py                          # full run
python feasibility_check.py --skip-smoke             # audit only (faster)
python feasibility_check.py --model phi3             # single model
python feasibility_check.py --output my_report.json  # custom path
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("feasibility")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VRAM_LIMIT_GB = 14.0
SAFE_FP16_GB  = 12.0
SMOKE_SAMPLES = 20

# Models needing eager attention (no flash-attn on Kaggle)
EAGER_ATTN_MODELS = {"phi3", "qwen25"}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "phi3": {
        "name": "Phi-3 Mini",
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "params_b": 3.8,
    },
    "gemma3": {
        "name": "Gemma 3 4B",
        "hf_id": "google/gemma-3-4b-it",
        "params_b": 4.0,
    },
    "mistral7b": {
        "name": "Mistral 7B",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "params_b": 7.0,
    },
    "llama31": {
        "name": "Llama 3.1 8B",
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "params_b": 8.0,
    },
    "qwen25": {
        "name": "Qwen 2.5 3B",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3.0,
    },
    "smollm": {
        "name": "SmolLM 1.7B",
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params_b": 1.7,
    },
}

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- General NLP ---
    "sst2": {
        "hf_id": "stanfordnlp/sst2",
        "split": "validation",
        "domain": "General NLP",
    },
    "hellaswag": {
        "hf_id": "Rowan/hellaswag",
        "split": "validation",
        "domain": "General NLP",
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "split": "validation",
        "config": "all",
        "domain": "General NLP",
    },
    # --- Financial NLP ---
    # Confirmed working: fields are 'text' and 'label' (0=bearish,1=bullish,2=neutral)
    "financial_sentiment": {
        "hf_id": "zeroshot/twitter-financial-news-sentiment",
        "split": "train",
        "domain": "Financial NLP",
        "note": "Twitter financial news sentiment (substitute for FinancialPhraseBank). "
                "Fields: text (str), label (int: 0=bearish, 1=bullish, 2=neutral).",
    },
    "fiqa": {
        "hf_id": "BeIR/fiqa",
        "split": "corpus",
        "config": "corpus",
        "domain": "Financial NLP",
    },
    # --- Structured Reasoning ---
    "svamp": {
        "hf_id": "ChilleD/SVAMP",
        "split": "test",
        "domain": "Structured Reasoning",
    },
    # Confirmed working: load with split="train", filter by sample['split']=='test'
    "tabmwp": {
        "hf_id": "Arietem/tabmwp",
        "split": "train",        # test rows are embedded; filter sample['split']=='test'
        "domain": "Structured Reasoning",
        "note": "TabMWP test rows embedded in train split. "
                "Filter by sample['split'] == 'test' in experiment pipeline.",
    },
    "humaneval": {
        "hf_id": "openai/openai_humaneval",
        "split": "test",
        "domain": "Structured Reasoning",
    },
}

TIERS: Dict[str, Dict[str, Any]] = {
    "fp16": {"label": "Tier 0 â€” FP16", "load_in_8bit": False, "load_in_4bit": False},
    "int8": {"label": "Tier 1 â€” INT8", "load_in_8bit": True,  "load_in_4bit": False},
    "int4": {"label": "Tier 2 â€” INT4", "load_in_8bit": False, "load_in_4bit": True},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelTierResult:
    model_key: str
    model_name: str
    tier: str
    peak_vram_gb: float
    load_time_s: float
    status: str          # "ok" | "oom" | "error"
    within_limit: bool
    notes: str = ""


@dataclass
class DatasetResult:
    dataset_key: str
    hf_id: str
    domain: str
    accessible: bool
    num_rows: int
    status: str          # "ok" | "error"
    notes: str = ""


@dataclass
class SmokeTestResult:
    ran: bool
    model: str
    shift_type: str
    tier: str
    domain: str
    baseline_acc: float
    shifted_acc: float
    delta_acc: float
    ece_proxy: float
    latency_ms_per_sample: float
    status: str          # "ok" | "error"
    notes: str = ""


@dataclass
class FeasibilityReport:
    timestamp: str
    python_version: str
    cuda_available: bool
    cuda_device: str
    vram_total_gb: float
    model_results: List[ModelTierResult] = field(default_factory=list)
    dataset_results: List[DatasetResult] = field(default_factory=list)
    smoke_test: Optional[SmokeTestResult] = None
    recommendations: List[str] = field(default_factory=list)
    overall_status: str = "pending"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gb(b: int) -> float:
    return round(b / (1024 ** 3), 2)


def flush_gpu(torch_mod) -> None:
    gc.collect()
    torch_mod.cuda.empty_cache()
    torch_mod.cuda.reset_peak_memory_stats()


def get_vram_info(torch_mod) -> Tuple[float, float]:
    allocated = gb(torch_mod.cuda.memory_allocated())
    total     = gb(torch_mod.cuda.get_device_properties(0).total_memory)
    return allocated, total


# ---------------------------------------------------------------------------
# Dependency import guard
# ---------------------------------------------------------------------------
def import_dependencies() -> Dict[str, Any]:
    missing = []
    deps: Dict[str, Any] = {}

    for pkg, attr in [("torch", "torch"), ("numpy", "np")]:
        try:
            import importlib
            mod = importlib.import_module(pkg)
            deps[attr] = mod
        except ImportError:
            missing.append(pkg)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
        deps["AutoTokenizer"]        = AutoTokenizer
        deps["AutoModelForCausalLM"] = AutoModelForCausalLM
        deps["BitsAndBytesConfig"]   = BitsAndBytesConfig
        deps["AutoConfig"]           = AutoConfig
    except ImportError:
        missing.append("transformers")

    try:
        from datasets import load_dataset
        deps["load_dataset"] = load_dataset
    except ImportError:
        missing.append("datasets")

    if missing:
        raise ImportError(
            f"\n\nMissing packages: {' '.join(missing)}\n"
            f"Run: pip install {' '.join(missing)} accelerate bitsandbytes\n"
        )
    return deps


# ---------------------------------------------------------------------------
# Step 1 â€” VRAM audit
# ---------------------------------------------------------------------------
def _build_load_kwargs(
    model_key: str,
    tier_key: str,
    torch_mod,
    BnBConfig,
    hf_token: Optional[str],
    low_cpu_mem: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Build (bnb_cfg, load_kwargs) for a given model + tier."""
    tier_cfg = TIERS[tier_key]

    bnb_cfg = None
    if tier_cfg["load_in_8bit"]:
        bnb_cfg = BnBConfig(load_in_8bit=True)
    elif tier_cfg["load_in_4bit"]:
        bnb_cfg = BnBConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_mod.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb_cfg is not None:
        kwargs["quantization_config"] = bnb_cfg
    else:
        kwargs["dtype"] = torch_mod.float16

    if model_key in EAGER_ATTN_MODELS:
        kwargs["attn_implementation"] = "eager"

    if low_cpu_mem:
        kwargs["low_cpu_mem_usage"] = True

    if hf_token:
        kwargs["token"] = hf_token

    return bnb_cfg, kwargs


def audit_model_vram(
    model_key: str,
    tier_key: str,
    deps: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> ModelTierResult:

    torch      = deps["torch"]
    AutoTok    = deps["AutoTokenizer"]
    AutoModel  = deps["AutoModelForCausalLM"]
    BnBConfig  = deps["BitsAndBytesConfig"]
    AutoConfig = deps["AutoConfig"]

    model_cfg = MODEL_REGISTRY[model_key]
    tier_cfg  = TIERS[tier_key]

    log.info(f"  Loading {model_cfg['name']} @ {tier_cfg['label']} ...")
    flush_gpu(torch)
    t0 = time.time()

    try:
        tok_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            tok_kwargs["token"] = hf_token

        tokenizer = AutoTok.from_pretrained(model_cfg["hf_id"], **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _, load_kwargs = _build_load_kwargs(
            model_key, tier_key, torch, BnBConfig, hf_token
        )

        # Phi-3 fix: newer config.json uses rope_type key but modeling_phi3.py
        # expects type key. Setting rope_scaling=None uses standard RoPE
        # which works correctly for inference on T4.
        if model_key == "phi3":
            cfg = AutoConfig.from_pretrained(
                model_cfg["hf_id"], trust_remote_code=True, **tok_kwargs
            )
            if isinstance(cfg.rope_scaling, dict) and "type" not in cfg.rope_scaling:
                log.info("  Phi-3: patching rope_scaling=None (rope_type/type mismatch)")
                cfg.rope_scaling = None
            load_kwargs["config"] = cfg

        model = AutoModel.from_pretrained(model_cfg["hf_id"], **load_kwargs)

        model.eval()
        load_time = round(time.time() - t0, 1)

        # Warm-up pass for true peak VRAM
        dummy = tokenizer("Hello world", return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = model(**dummy)

        peak_vram, _ = get_vram_info(torch)
        within = peak_vram <= VRAM_LIMIT_GB

        notes = []
        if not within:
            notes.append(f"Exceeds {VRAM_LIMIT_GB} GB limit.")
        if tier_key == "fp16" and peak_vram > SAFE_FP16_GB:
            notes.append("FP16 > 12 GB â€” use INT8 as baseline tier.")

        result = ModelTierResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier_key,
            peak_vram_gb=peak_vram,
            load_time_s=load_time,
            status="ok",
            within_limit=within,
            notes=" | ".join(notes),
        )

        del model, tokenizer, dummy
        flush_gpu(torch)
        return result

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        flush_gpu(torch)
        err = str(exc)
        is_oom = "out of memory" in err.lower() or "outofmemory" in err.lower()
        status = "oom" if is_oom else "error"
        log.warning(f"  {status.upper()}: {model_cfg['name']} @ {tier_cfg['label']}: {err[:120]}")
        return ModelTierResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier_key,
            peak_vram_gb=-1.0,
            load_time_s=-1.0,
            status=status,
            within_limit=False,
            notes=err[:200],
        )

    except Exception as exc:
        flush_gpu(torch)
        log.error(f"  ERROR {model_cfg['name']} @ {tier_cfg['label']}: {exc}")
        return ModelTierResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier_key,
            peak_vram_gb=-1.0,
            load_time_s=-1.0,
            status="error",
            within_limit=False,
            notes=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# Step 2 â€” Dataset audit
# ---------------------------------------------------------------------------
def audit_dataset(
    dataset_key: str,
    deps: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> DatasetResult:

    load_dataset = deps["load_dataset"]
    cfg = DATASET_REGISTRY[dataset_key]
    log.info(f"  Checking: {cfg['hf_id']} ({cfg['split']}) ...")

    try:
        kwargs: Dict[str, Any] = {
            "split": cfg["split"],
            "streaming": True,
            "trust_remote_code": True,
        }
        if "config" in cfg:
            kwargs["name"] = cfg["config"]
        if hf_token:
            kwargs["token"] = hf_token

        ds     = load_dataset(cfg["hf_id"], **kwargs)
        sample = list(ds.take(3))

        return DatasetResult(
            dataset_key=dataset_key,
            hf_id=cfg["hf_id"],
            domain=cfg["domain"],
            accessible=True,
            num_rows=len(sample),
            status="ok",
            notes=cfg.get("note", ""),
        )

    except Exception as exc:
        log.warning(f"  FAIL {dataset_key}: {exc}")
        return DatasetResult(
            dataset_key=dataset_key,
            hf_id=cfg["hf_id"],
            domain=cfg["domain"],
            accessible=False,
            num_rows=0,
            status="error",
            notes=str(exc)[:300],
        )


# ---------------------------------------------------------------------------
# Step 3 â€” Smoke test
# ---------------------------------------------------------------------------
def run_smoke_test(
    deps: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> SmokeTestResult:
    """
    Full pipeline cell:
      Model  : Gemma 3 4B @ FP16  (most stable on T4 from audit)
      Shift  : 5% character noise
      Domain : SST-2 sentiment (binary)
    Measures: baseline acc, shifted acc, Î”Acc, ECE proxy, latency.
    """
    torch        = deps["torch"]
    AutoTok      = deps["AutoTokenizer"]
    AutoModel    = deps["AutoModelForCausalLM"]
    np           = deps["np"]
    load_dataset = deps["load_dataset"]

    log.info("Smoke test: Gemma 3 4B | FP16 | SST-2 | 5% noise ...")

    try:
        # Load SST-2
        ds = load_dataset(
            "stanfordnlp/sst2",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
        samples = list(ds.take(SMOKE_SAMPLES))
        texts  = [s["sentence"] for s in samples]
        labels = [s["label"]    for s in samples]

        # Load Gemma 3 4B FP16
        flush_gpu(torch)
        model_id = MODEL_REGISTRY["gemma3"]["hf_id"]
        tok_kw: Dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            tok_kw["token"] = hf_token

        tokenizer = AutoTok.from_pretrained(model_id, **tok_kw)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        mdl_kw: Dict[str, Any] = {
            "dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if hf_token:
            mdl_kw["token"] = hf_token

        model = AutoModel.from_pretrained(model_id, **mdl_kw)
        model.eval()

        PROMPT = (
            "Classify the sentiment as exactly 'positive' or 'negative'.\n"
            "Text: {text}\nAnswer:"
        )

        def predict(text: str) -> Tuple[int, float]:
            inp = tokenizer(
                PROMPT.format(text=text),
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to("cuda")
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[:, -1, :]
            probs  = torch.softmax(logits, dim=-1)
            pos_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
            neg_id = tokenizer.encode(" negative", add_special_tokens=False)[0]
            pos_p  = probs[0, pos_id].item()
            neg_p  = probs[0, neg_id].item()
            denom  = pos_p + neg_p + 1e-9
            pred   = 1 if pos_p >= neg_p else 0
            conf   = max(pos_p, neg_p) / denom
            return pred, conf

        def inject_noise(text: str, rate: float = 0.05) -> str:
            import random
            alpha = "abcdefghijklmnopqrstuvwxyz"
            chars = list(text)
            for i, ch in enumerate(chars):
                if ch != " " and random.random() < rate:
                    chars[i] = random.choice(alpha)
            return "".join(chars)

        # Baseline
        b_preds, b_confs, lats = [], [], []
        for t in texts:
            t0 = time.time()
            p, c = predict(t)
            lats.append((time.time() - t0) * 1000)
            b_preds.append(p)
            b_confs.append(c)
        baseline_acc = float(np.mean([p == l for p, l in zip(b_preds, labels)]))

        # Shifted
        noisy = [inject_noise(t) for t in texts]
        s_preds, s_confs = [], []
        for t in noisy:
            p, c = predict(t)
            s_preds.append(p)
            s_confs.append(c)
        shifted_acc = float(np.mean([p == l for p, l in zip(s_preds, labels)]))
        delta_acc   = round(baseline_acc - shifted_acc, 4)
        ece_proxy   = float(np.mean([
            abs(c - int(p == l))
            for c, p, l in zip(s_confs, s_preds, labels)
        ]))
        avg_lat = round(float(np.mean(lats)), 2)

        log.info(
            f"  âœ“ baseline={baseline_acc:.3f}  shifted={shifted_acc:.3f}  "
            f"Î”={delta_acc:.3f}  ECE={ece_proxy:.3f}  {avg_lat}ms/sample"
        )

        del model, tokenizer
        flush_gpu(torch)

        return SmokeTestResult(
            ran=True,
            model="Gemma 3 4B (FP16)",
            shift_type="Input Noise 5% char substitution",
            tier="fp16",
            domain="General NLP â€” SST-2",
            baseline_acc=round(baseline_acc, 4),
            shifted_acc=round(shifted_acc, 4),
            delta_acc=delta_acc,
            ece_proxy=round(ece_proxy, 4),
            latency_ms_per_sample=avg_lat,
            status="ok",
        )

    except Exception as exc:
        log.error(f"  Smoke test FAILED: {exc}")
        log.debug(traceback.format_exc())
        return SmokeTestResult(
            ran=False,
            model="Gemma 3 4B (FP16)",
            shift_type="Input Noise 5%",
            tier="fp16",
            domain="General NLP â€” SST-2",
            baseline_acc=0.0, shifted_acc=0.0, delta_acc=0.0,
            ece_proxy=0.0, latency_ms_per_sample=0.0,
            status="error",
            notes=str(exc)[:300],
        )


# ---------------------------------------------------------------------------
# Step 4 â€” Recommendations
# ---------------------------------------------------------------------------
def generate_recommendations(report: FeasibilityReport) -> List[str]:
    recs: List[str] = []

    failed_fp16 = [
        r.model_name for r in report.model_results
        if r.tier == "fp16" and r.status in ("oom", "error")
    ]
    if failed_fp16:
        recs.append(
            f"FP16 failures: {', '.join(failed_fp16)}. "
            "Check error notes â€” likely a library compatibility issue, not true OOM."
        )

    tight_fp16 = [
        r.model_name for r in report.model_results
        if r.tier == "fp16" and r.within_limit and r.peak_vram_gb > SAFE_FP16_GB
    ]
    if tight_fp16:
        recs.append(
            f"{', '.join(tight_fp16)} uses >12 GB at FP16. "
            "Treat INT8 as the Tier 0 baseline for these models."
        )

    failed_int8 = [
        r.model_name for r in report.model_results
        if r.tier == "int8" and r.status in ("oom", "error")
    ]
    if failed_int8:
        recs.append(
            f"INT8 failures: {', '.join(failed_int8)}. "
            "Try: pip install bitsandbytes --upgrade"
        )

    failed_ds = [
        r.dataset_key for r in report.dataset_results if not r.accessible
    ]
    if failed_ds:
        recs.append(
            f"Dataset failures: {', '.join(failed_ds)}. "
            "Check HuggingFace Hub for updated dataset IDs."
        )

    if report.smoke_test:
        if report.smoke_test.status == "ok":
            if report.smoke_test.delta_acc > 0.05:
                recs.append(
                    f"Smoke test Î”Acc={report.smoke_test.delta_acc:.3f} at 5% noise. "
                    "Pipeline is correctly capturing shift sensitivity."
                )
            if report.smoke_test.latency_ms_per_sample > 2000:
                est_h = round(
                    SMOKE_SAMPLES * 500 * report.smoke_test.latency_ms_per_sample
                    / 1000 / 3600, 1
                )
                recs.append(
                    f"Latency {report.smoke_test.latency_ms_per_sample}ms/sample. "
                    f"Estimated full-run time: ~{est_h}h. Use batched inference."
                )
        else:
            recs.append(f"Smoke test FAILED: {report.smoke_test.notes[:150]}")

    if not recs:
        recs.append("All checks passed. Ready for full experiment run.")

    return recs


# ---------------------------------------------------------------------------
# CLI + orchestration
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feasibility checker v3 â€” Fragility Under Constraint"
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
    )
    p.add_argument(
        "--tiers",
        nargs="+",
        choices=list(TIERS.keys()),
        default=["fp16", "int8", "int4"],
    )
    p.add_argument("--skip-datasets", action="store_true")
    p.add_argument("--skip-smoke",    action="store_true")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    p.add_argument("--output",    type=str, default="feasibility_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("Fragility Under Constraint â€” Feasibility Checker v3")
    log.info("=" * 60)

    try:
        deps = import_dependencies()
    except ImportError as e:
        log.error(str(e))
        sys.exit(1)

    torch = deps["torch"]

    if not torch.cuda.is_available():
        log.error("CUDA not available. Run on a GPU instance.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    _, vram_total = get_vram_info(torch)
    log.info(f"GPU: {device_name} | VRAM: {vram_total} GB")

    report = FeasibilityReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        python_version=sys.version.split()[0],
        cuda_available=True,
        cuda_device=device_name,
        vram_total_gb=vram_total,
    )

    # --- VRAM audit ---
    model_keys = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    log.info("\n[1/3] VRAM Audit")
    log.info("-" * 40)
    for mk in model_keys:
        log.info(f"Model: {MODEL_REGISTRY[mk]['name']}")
        for tk in args.tiers:
            r = audit_model_vram(mk, tk, deps, hf_token=args.hf_token)
            report.model_results.append(r)
            icon = "âœ“" if r.status == "ok" and r.within_limit else "âœ—"
            log.info(
                f"    {icon} {TIERS[tk]['label']:20s} | "
                f"VRAM: {r.peak_vram_gb:5.2f} GB | "
                f"Load: {r.load_time_s:5.1f}s | "
                f"Status: {r.status}"
                + (f" | {r.notes[:80]}" if r.notes else "")
            )

    # --- Dataset audit ---
    if not args.skip_datasets:
        log.info("\n[2/3] Dataset Audit")
        log.info("-" * 40)
        for dk in DATASET_REGISTRY:
            r = audit_dataset(dk, deps, hf_token=args.hf_token)
            report.dataset_results.append(r)
            icon = "âœ“" if r.accessible else "âœ—"
            log.info(
                f"  {icon} {dk:30s} | {r.domain:25s} | {r.status}"
                + (f" | {r.notes[:60]}" if r.notes else "")
            )
    else:
        log.info("\n[2/3] Dataset Audit â€” SKIPPED")

    # --- Smoke test ---
    if not args.skip_smoke:
        log.info("\n[3/3] Smoke Test")
        log.info("-" * 40)
        smoke = run_smoke_test(deps, hf_token=args.hf_token)
        report.smoke_test = smoke
        if smoke.status != "ok":
            log.info(f"  âœ— Failed: {smoke.notes}")
    else:
        log.info("\n[3/3] Smoke Test â€” SKIPPED")

    # --- Recommendations ---
    log.info("\n[Recommendations]")
    log.info("-" * 40)
    report.recommendations = generate_recommendations(report)
    for i, rec in enumerate(report.recommendations, 1):
        log.info(f"  {i}. {rec}")

    # --- Overall status ---
    has_model_fail = any(
        r.status in ("oom", "error") for r in report.model_results
    )
    has_ds_fail    = any(not r.accessible for r in report.dataset_results)
    smoke_failed   = report.smoke_test and report.smoke_test.status == "error"

    if smoke_failed or (has_model_fail and has_ds_fail):
        report.overall_status = "fail"
    elif has_model_fail or has_ds_fail:
        report.overall_status = "warn"
    else:
        report.overall_status = "pass"

    log.info(f"\nOverall status: {report.overall_status.upper()}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    log.info(f"Report saved to: {args.output}")
    log.info("=" * 60)


if __name__ == "__main__":
>>>>>>> e852c4aa5c571777bba8eaa27f5d459e50ce4428
    main()