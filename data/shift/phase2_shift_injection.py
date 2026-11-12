"""
phase2_shift_injection.py  v3
==============================
Fragility Under Constraint — Phase 2: Distribution Shift Injection
Target hardware: Kaggle T4 (15 GB VRAM)

CHANGES IN V3 (Financial NLP sampling fix)
--------------------------------------------
1. financial_nlp dataset was shipping label-sorted, not shuffled — first-N
   sampling produced monolithic-label draws. Fixed with a gated shuffle
   (buffer_size=10000 covers the full 9,543-row train split = true global
   shuffle). Only financial_nlp is shuffled; general_nlp/structured_reasoning
   were confirmed already balanced under first-N sampling and are left
   untouched so their existing results stay reproducible.
2. Removed forced torch_dtype=bfloat16 on top of INT8/INT4 quantization for
   Gemma — bitsandbytes handles compute dtype internally; forcing bfloat16
   caused a per-matmul bfloat16->float16 cast, which was both extremely
   noisy (thousands of repeated warnings) and added real overhead.
3. Added MAX_CELL_SECONDS time budget per cell. A repeat of the Mistral
   INT8 + structured_reasoning situation (3h46m for one 50-sample cell in
   Phase 1) is now capped and marked "timeout_partial" instead of silently
   running unbounded.
4. Warning/log noise suppression added at module level (must live inside
   this file, not a notebook cell above it — this runs as a subprocess).

CHANGES IN V2 (paper strengthening pass)
------------------------------------------
1. Samples per cell increased 30 -> 100 (statistical significance fix)
2. Input Noise shift now has 3 severity levels (5%, 15%, 30%) instead of 2,
   restoring the original 3-tier plan from the research brief.
   Prompt degradation and vocab mismatch remain at 2 severities (low/high)
   since they don't have a natural third level the way noise rate does.

What this script does
---------------------
Applies shift types to each domain, runs all model x tier x shift x
severity combinations, and computes delta-acc vs Phase 1 baseline.
Writes results to shift_results.csv after every cell (crash-safe).

Shift types implemented
-----------------------
Shift 1 - Input Noise
  Character-level corruption (typos, OCR errors, keyboard substitution)
  Severity: low=5%, mid=15%, high=30% token corruption rate (3 levels)

Shift 2 - Prompt Structure Degradation
  Systematic removal of prompt components (instructions, context)
  Severity: low=partial degradation, high=minimal prompt (text only)

Shift 3 - Domain Vocabulary Mismatch
  Inject out-of-domain jargon tokens into the input text
  Severity: low=10% token replacement, high=25% token replacement
  (Shift 4 / temporal drift excluded: hard to operationalize rigorously
   on these datasets without a temporal split -- noted in paper limitations)

Experiment matrix (v2)
-----------------------
Noise:        14 model-tier combos x 3 domains x 3 severities = 126 cells
Prompt degr:  14 model-tier combos x 3 domains x 2 severities = 84 cells
Vocab mism:   14 model-tier combos x 3 domains x 2 severities = 84 cells
Total: 294 cells (14 model-tier combos = 5 models x 3 tiers, minus Mistral INT8/INT4)
Samples per cell: 100
Estimated time: 20-26 hours on T4 (run via Kaggle commit, resumable via --resume)

Usage
-----
python phase2_shift_injection.py --hf-token YOUR_TOKEN
python phase2_shift_injection.py --model gemma3 --shift noise --hf-token YOUR_TOKEN
python phase2_shift_injection.py --resume --hf-token YOUR_TOKEN

# Scoped rerun for financial_nlp only (use a fresh --output-dir, do NOT
# restore an old checkpoint into it, so broken cells actually get redone
# rather than skipped by --resume):
python phase2_shift_injection.py --domains financial_nlp \
    --output-dir /kaggle/working/financial_rerun \
    --baseline-csv /kaggle/working/baseline_results.csv \
    --hf-token YOUR_TOKEN
"""

import argparse
import csv
import gc
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Warning / log noise suppression — must be here, in the subprocess itself.
# Setting this in a notebook cell above has no effect since this script
# runs as a separate `python phase2_shift_injection.py` process.
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore", message=".*inputs will be cast from torch.bfloat16 to float16.*"
)
warnings.filterwarnings("once")

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phase2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLES_PER_CELL  = 100
RANDOM_SEED       = 42
CHECKPOINT_FILE   = "phase2_checkpoint.json"
RESULTS_CSV       = "shift_results.csv"
RESULTS_JSON      = "shift_results.json"

# Hard time budget per cell (seconds). Prevents one pathological combo
# from eating hours unsupervised — mark it timeout_partial and move on.
MAX_CELL_SECONDS  = 1800  # 30 min

EAGER_ATTN_MODELS = {"qwen25"}
BFLOAT16_MODELS   = {"gemma3"}
CHAT_TEMPLATE_MODELS = {"gemma3", "mistral7b", "llama31", "qwen25", "smollm"}
EXTRA_EOS_TOKENS  = {"gemma3": [1, 107]}

# Skip Mistral INT8 (28s/sample confirmed in Phase 1) and Mistral INT4
# (OOM on T4 after heavy prior model usage confirmed in Phase 2 v2 run)
SKIP_COMBINATIONS = {("mistral7b", "int8"), ("mistral7b", "int4")}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gemma3":   {"name": "Gemma 3 4B",   "hf_id": "google/gemma-3-4b-it",                        "params_b": 4.0},
    "mistral7b":{"name": "Mistral 7B",   "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",           "params_b": 7.0},
    "llama31":  {"name": "Llama 3.1 8B", "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",        "params_b": 8.0},
    "qwen25":   {"name": "Qwen 2.5 3B",  "hf_id": "Qwen/Qwen2.5-3B-Instruct",                    "params_b": 3.0},
    "smollm":   {"name": "SmolLM 1.7B",  "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",         "params_b": 1.7},
}

TIERS: Dict[str, Dict[str, Any]] = {
    "fp16": {"label": "Tier 0 — FP16", "load_in_8bit": False, "load_in_4bit": False},
    "int8": {"label": "Tier 1 — INT8", "load_in_8bit": True,  "load_in_4bit": False},
    "int4": {"label": "Tier 2 — INT4", "load_in_8bit": False, "load_in_4bit": True},
}

DOMAIN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "general_nlp": {
        "name": "General NLP",
        "task": "sentiment",
        "dataset_id": "stanfordnlp/sst2",
        "dataset_split": "validation",
        "text_field": "sentence",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "prompt_template": (
            "Classify the sentiment of the following text.\n"
            "Reply with exactly one word: 'positive' or 'negative'.\n"
            "Text: {text}\nAnswer:"
        ),
        "prompt_template_degraded_low": (
            "Sentiment of this text — positive or negative?\n"
            "Text: {text}\nAnswer:"
        ),
        "prompt_template_degraded_high": "{text}",
        "valid_answers": ["positive", "negative"],
        "metric": "accuracy",
    },
    "financial_nlp": {
        "name": "Financial NLP",
        "task": "sentiment",
        "dataset_id": "zeroshot/twitter-financial-news-sentiment",
        "dataset_split": "train",
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive", 2: "neutral"},
        "prompt_template": (
            "Classify the sentiment of this financial news as exactly "
            "'positive', 'negative', or 'neutral'.\n"
            "Text: {text}\nAnswer:"
        ),
        "prompt_template_degraded_low": (
            "Financial sentiment — positive, negative, or neutral?\n"
            "{text}\nAnswer:"
        ),
        "prompt_template_degraded_high": "{text}",
        "valid_answers": ["positive", "negative", "neutral"],
        "metric": "accuracy",
    },
    "structured_reasoning": {
        "name": "Structured Reasoning",
        "task": "math",
        "dataset_id": "ChilleD/SVAMP",
        "dataset_split": "test",
        "text_field": "question_concat",
        "label_field": "Answer",
        "label_map": {},
        "prompt_template": (
            "Solve this math problem step by step, then write the final answer "
            "on the last line as: Answer: <number>\n"
            "Problem: {text}"
        ),
        "prompt_template_degraded_low": (
            "Math problem:\n{text}\nAnswer:"
        ),
        "prompt_template_degraded_high": "{text}",
        "valid_answers": [],
        "metric": "exact_match",
    },
}

# Shift configs
SHIFT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "noise": {
        "name": "Input Noise",
        "description": "Character-level corruption simulating typos and OCR errors",
        "severities": {"low": 0.05, "mid": 0.15, "high": 0.30},
    },
    "prompt_degradation": {
        "name": "Prompt Structure Degradation",
        "description": "Systematic removal of prompt instructions and context",
        "severities": {"low": "degraded_low", "high": "degraded_high"},
    },
    "vocab_mismatch": {
        "name": "Domain Vocabulary Mismatch",
        "description": "Injection of out-of-domain jargon tokens",
        "severities": {"low": 0.10, "high": 0.25},
    },
}

# Out-of-domain jargon pools for vocab mismatch shift
JARGON_POOLS: Dict[str, List[str]] = {
    "financial": [
        "amortization", "arbitrage", "collateral", "derivatives", "equities",
        "fiduciary", "hedging", "illiquid", "leverage", "liquidity",
        "monetization", "portfolio", "rebalancing", "securitization", "volatility",
        "tranche", "covenant", "divestiture", "prospectus", "syndication",
    ],
    "legal": [
        "adjudication", "affidavit", "arbitration", "bailment", "certiorari",
        "deposition", "estoppel", "indemnification", "injunction", "jurisdiction",
        "liability", "litigation", "mandamus", "negligence", "subrogation",
        "tortious", "venue", "waiver", "writ", "culpability",
    ],
    "technical": [
        "asynchronous", "bandwidth", "caching", "daemon", "encryption",
        "heuristic", "idempotent", "kernel", "latency", "middleware",
        "namespace", "orchestration", "polymorphism", "recursive", "serialization",
        "sharding", "throughput", "vectorization", "webhook", "containerization",
    ],
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class ShiftCellResult:
    model_key: str
    model_name: str
    tier: str
    domain: str
    task: str
    shift_type: str
    shift_name: str
    severity: str
    severity_value: Any
    n_samples: int
    baseline_acc: float
    shifted_acc: float
    delta_acc: float
    baseline_f1: float
    shifted_f1: float
    delta_f1: float
    ece_proxy: float
    latency_ms_mean: float
    latency_ms_std: float
    status: str            # "ok" | "error" | "timeout_partial"
    error_msg: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gb(b: int) -> float:
    return round(b / (1024 ** 3), 2)


def flush_gpu(torch_mod) -> None:
    """
    Aggressive VRAM release between model loads. Plain empty_cache() alone
    has shown intermittent CUDA OOM (llama31/int8, mistral7b/int4) across
    multiple runs — likely allocator fragmentation that needs a moment to
    actually release.
    """
    gc.collect()
    torch_mod.cuda.empty_cache()
    torch_mod.cuda.synchronize()
    torch_mod.cuda.reset_peak_memory_stats()
    time.sleep(2)


def cell_id(model_key: str, tier: str, domain: str, shift: str, severity: str) -> str:
    return f"{model_key}__{tier}__{domain}__{shift}__{severity}"


def load_checkpoint(path: str) -> set:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        completed = set(data.get("completed", []))
        log.info(f"Checkpoint: {len(completed)} cells already done.")
        return completed
    return set()


def save_checkpoint(path: str, completed: set) -> None:
    with open(path, "w") as f:
        json.dump({"completed": list(completed)}, f)


def append_csv(result: ShiftCellResult, path: str) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(result))


# ---------------------------------------------------------------------------
# Shift injection functions
# ---------------------------------------------------------------------------
def inject_noise(text: str, rate: float, seed: int = RANDOM_SEED) -> str:
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch not in (" ", "\n", "\t") and rng.random() < rate:
            chars[i] = rng.choice(alphabet)
    return "".join(chars)


def inject_vocab_mismatch(text: str, rate: float, seed: int = RANDOM_SEED) -> str:
    rng = random.Random(seed)
    all_jargon = (
        JARGON_POOLS["financial"]
        + JARGON_POOLS["legal"]
        + JARGON_POOLS["technical"]
    )
    words = text.split()
    for i, word in enumerate(words):
        if rng.random() < rate:
            words[i] = rng.choice(all_jargon)
    return " ".join(words)


def apply_shift(
    text: str,
    prompt_template: str,
    shift_type: str,
    severity_value: Any,
    domain_cfg: Dict[str, Any],
    sample_idx: int = 0,
) -> str:
    if shift_type == "noise":
        shifted_text = inject_noise(text, severity_value, seed=RANDOM_SEED + sample_idx)
        return prompt_template.format(text=shifted_text[:400])

    elif shift_type == "vocab_mismatch":
        shifted_text = inject_vocab_mismatch(text, severity_value, seed=RANDOM_SEED + sample_idx)
        return prompt_template.format(text=shifted_text[:400])

    elif shift_type == "prompt_degradation":
        template_key = f"prompt_template_{severity_value}"
        degraded_template = domain_cfg.get(template_key, prompt_template)
        return degraded_template.format(text=text[:400])

    return prompt_template.format(text=text[:400])


# ---------------------------------------------------------------------------
# Dependency imports
# ---------------------------------------------------------------------------
def import_dependencies() -> Dict[str, Any]:
    missing = []
    deps: Dict[str, Any] = {}

    try:
        import torch
        deps["torch"] = torch
    except ImportError:
        missing.append("torch")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        deps["AutoTokenizer"]        = AutoTokenizer
        deps["AutoModelForCausalLM"] = AutoModelForCausalLM
        deps["BitsAndBytesConfig"]   = BitsAndBytesConfig
    except ImportError:
        missing.append("transformers")

    try:
        from datasets import load_dataset
        deps["load_dataset"] = load_dataset
    except ImportError:
        missing.append("datasets")

    try:
        import numpy as np
        deps["np"] = np
    except ImportError:
        missing.append("numpy")

    try:
        from sklearn.metrics import f1_score
        deps["f1_score"] = f1_score
    except ImportError:
        missing.append("scikit-learn")

    if missing:
        raise ImportError(
            f"\n\nMissing: {' '.join(missing)}\n"
            f"Run: pip install {' '.join(missing)} accelerate bitsandbytes\n"
        )
    return deps


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(
    model_key: str,
    tier_key: str,
    deps: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any]:
    torch     = deps["torch"]
    AutoTok   = deps["AutoTokenizer"]
    AutoModel = deps["AutoModelForCausalLM"]
    BnBConfig = deps["BitsAndBytesConfig"]

    model_cfg = MODEL_REGISTRY[model_key]
    tier_cfg  = TIERS[tier_key]
    dtype     = torch.bfloat16 if model_key in BFLOAT16_MODELS else torch.float16

    log.info(f"Loading {model_cfg['name']} @ {tier_cfg['label']} ...")
    flush_gpu(torch)

    tok_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        tok_kwargs["token"] = hf_token

    tokenizer = AutoTok.from_pretrained(model_cfg["hf_id"], **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = None
    if tier_cfg["load_in_8bit"]:
        bnb_cfg = BnBConfig(load_in_8bit=True)
    elif tier_cfg["load_in_4bit"]:
        bnb_cfg = BnBConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    load_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if bnb_cfg is not None:
        load_kwargs["quantization_config"] = bnb_cfg
        # Deliberately NOT forcing torch_dtype here for quantized tiers —
        # bitsandbytes handles compute dtype internally. Forcing bfloat16
        # on top of INT8/INT4 caused a per-matmul cast (noisy + slower).
    else:
        load_kwargs["dtype"] = dtype

    if model_key in EAGER_ATTN_MODELS:
        load_kwargs["attn_implementation"] = "eager"
    if hf_token:
        load_kwargs["token"] = hf_token

    model = AutoModel.from_pretrained(model_cfg["hf_id"], **load_kwargs)
    model.eval()
    log.info(f"  Loaded. VRAM: {gb(torch.cuda.memory_allocated())} GB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def normalize_number(s: str) -> str:
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(round(f, 4))
    except (ValueError, TypeError):
        return s.strip()


def extract_answer(output_text: str, valid_answers: List[str], task: str) -> str:
    text = output_text.strip().lower()
    if task == "math":
        cot_match = re.search(r"answer[:\s]+(-?\d+(?:\.\d+)?)", text)
        if cot_match:
            return normalize_number(cot_match.group(1))
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return normalize_number(nums[-1]) if nums else ""
    for ans in valid_answers:
        if ans.lower() in text:
            return ans.lower()
    return text.split()[0] if text.split() else ""


def run_inference(
    model,
    tokenizer,
    prompt: str,
    valid_answers: List[str],
    task: str,
    deps: Dict[str, Any],
    model_key: str = "",
    max_new_tokens: int = 16,
) -> Tuple[str, float, float]:
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
        text, return_tensors="pt", truncation=True,
        max_length=512, padding=False,
    ).to("cuda")

    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    eos_ids = list(set(eos_ids + EXTRA_EOS_TOKENS.get(model_key, []))) or None

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
    decoded   = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    pred = extract_answer(decoded, valid_answers, task)

    if task != "math" and valid_answers:
        try:
            with torch.no_grad():
                logits = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ).logits[:, -1, :]
            if logits.isnan().any():
                confidence = 1.0 / max(len(valid_answers), 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                ans_probs = []
                for ans in valid_answers:
                    tok_id = tokenizer.encode(f" {ans}", add_special_tokens=False)
                    ans_probs.append(probs[0, tok_id[0]].item() if tok_id else 0.0)
                confidence = max(ans_probs) / (sum(ans_probs) + 1e-9)
        except Exception:
            confidence = 1.0 / max(len(valid_answers), 1)
    else:
        confidence = 1.0

    return pred, confidence, latency_ms


# ---------------------------------------------------------------------------
# Cell evaluation
# ---------------------------------------------------------------------------
def evaluate_shifted_cell(
    model_key: str,
    tier_key: str,
    domain_key: str,
    shift_type: str,
    severity: str,
    severity_value: Any,
    model,
    tokenizer,
    deps: Dict[str, Any],
    baseline_results: Dict[str, float],
    n_samples: int = SAMPLES_PER_CELL,
) -> ShiftCellResult:

    np           = deps["np"]
    f1_score     = deps["f1_score"]
    load_dataset = deps["load_dataset"]

    domain_cfg  = DOMAIN_REGISTRY[domain_key]
    model_cfg   = MODEL_REGISTRY[model_key]
    shift_cfg   = SHIFT_REGISTRY[shift_type]
    task        = domain_cfg["task"]
    template    = domain_cfg["prompt_template"]

    baseline_key = f"{model_key}__{tier_key}__{domain_key}"
    baseline_acc = baseline_results.get(baseline_key + "__acc", 0.0)
    baseline_f1  = baseline_results.get(baseline_key + "__f1", 0.0)

    log.info(
        f"  Shift: {shift_cfg['name']} | {severity} | "
        f"baseline_acc={baseline_acc:.3f}"
    )

    try:
        ds = load_dataset(
            domain_cfg["dataset_id"],
            split=domain_cfg["dataset_split"],
            streaming=True,
        )
        if domain_key == "financial_nlp":
            # Fix: dataset ships label-sorted, not shuffled. buffer_size=10000
            # covers the entire train split (9,543 rows) = true global shuffle.
            ds = ds.shuffle(buffer_size=10000, seed=42)
        samples = list(ds.take(n_samples))

        preds, labels, confs, lats = [], [], [], []
        cell_start = time.time()
        timed_out = False

        for i, sample in enumerate(samples):
            elapsed = time.time() - cell_start
            if elapsed > MAX_CELL_SECONDS:
                log.warning(
                    f"  ⚠ Cell exceeded {MAX_CELL_SECONDS}s budget after "
                    f"{i}/{n_samples} samples ({elapsed:.0f}s elapsed) — "
                    f"truncating."
                )
                timed_out = True
                break

            text  = str(sample[domain_cfg["text_field"]])
            label = sample[domain_cfg["label_field"]]

            if domain_cfg["label_map"] and isinstance(label, int):
                label_str = domain_cfg["label_map"].get(label, str(label))
            elif task == "math":
                label_str = normalize_number(str(label))
            else:
                label_str = str(label).strip().lower()

            shifted_prompt = apply_shift(
                text, template, shift_type, severity_value,
                domain_cfg, sample_idx=i,
            )

            pred, conf, lat = run_inference(
                model, tokenizer, shifted_prompt,
                domain_cfg["valid_answers"], task, deps,
                model_key=model_key,
                max_new_tokens=150 if task == "math" else 16,
            )

            preds.append(pred)
            labels.append(label_str)
            confs.append(conf)
            lats.append(lat)

        correct     = [p == l for p, l in zip(preds, labels)]
        shifted_acc = float(np.mean(correct)) if correct else 0.0
        delta_acc   = round(baseline_acc - shifted_acc, 4)

        unique_labels = list(set(labels))
        if len(unique_labels) > 1:
            try:
                shifted_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
            except Exception:
                shifted_f1 = shifted_acc
        else:
            shifted_f1 = shifted_acc
        delta_f1 = round(baseline_f1 - shifted_f1, 4)

        ece = float(np.mean([abs(c - int(ok)) for c, ok in zip(confs, correct)])) if correct else 0.0
        lat_arr = np.array(lats) if lats else np.array([0.0])

        status = "timeout_partial" if timed_out else "ok"

        log.info(
            f"  Done ({status}) — shifted_acc={shifted_acc:.3f} Δacc={delta_acc:+.3f} "
            f"Δf1={delta_f1:+.3f} ece={ece:.3f} lat={np.mean(lat_arr):.1f}ms "
            f"n={len(preds)}/{n_samples}"
        )

        return ShiftCellResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier_key,
            domain=domain_key,
            task=task,
            shift_type=shift_type,
            shift_name=shift_cfg["name"],
            severity=severity,
            severity_value=str(severity_value),
            n_samples=len(preds),
            baseline_acc=baseline_acc,
            shifted_acc=round(shifted_acc, 4),
            delta_acc=delta_acc,
            baseline_f1=baseline_f1,
            shifted_f1=round(shifted_f1, 4),
            delta_f1=delta_f1,
            ece_proxy=round(ece, 4),
            latency_ms_mean=round(float(np.mean(lat_arr)), 2),
            latency_ms_std=round(float(np.std(lat_arr)), 2),
            status=status,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    except Exception as exc:
        log.error(f"  Cell failed: {exc}")
        log.debug(traceback.format_exc())
        return ShiftCellResult(
            model_key=model_key,
            model_name=model_cfg["name"],
            tier=tier_key,
            domain=domain_key,
            task=task,
            shift_type=shift_type,
            shift_name=shift_cfg["name"],
            severity=severity,
            severity_value=str(severity_value),
            n_samples=0,
            baseline_acc=baseline_acc,
            shifted_acc=0.0,
            delta_acc=0.0,
            baseline_f1=baseline_f1,
            shifted_f1=0.0,
            delta_f1=0.0,
            ece_proxy=0.0,
            latency_ms_mean=0.0,
            latency_ms_std=0.0,
            status="error",
            error_msg=str(exc)[:200],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )


# ---------------------------------------------------------------------------
# Load Phase 1 baselines
# ---------------------------------------------------------------------------
def load_baselines(csv_path: str) -> Dict[str, float]:
    baselines: Dict[str, float] = {}
    if not os.path.exists(csv_path):
        log.warning(f"Baseline CSV not found at {csv_path}. ΔAcc will be 0.")
        return baselines

    import csv as csv_mod
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            key = f"{row['model_key']}__{row['tier']}__{row['domain']}"
            try:
                baselines[key + "__acc"] = float(row["accuracy"])
                baselines[key + "__f1"]  = float(row["f1_macro"])
            except (ValueError, KeyError):
                pass

    log.info(f"Loaded {len(baselines)//2} baseline cells from {csv_path}")
    return baselines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 shift injection — Fragility Under Constraint"
    )
    p.add_argument("--model",   choices=list(MODEL_REGISTRY.keys()) + ["all"], default="all")
    p.add_argument("--tiers",   nargs="+", choices=list(TIERS.keys()),          default=["fp16","int8","int4"])
    p.add_argument("--domains", nargs="+", choices=list(DOMAIN_REGISTRY.keys()),default=list(DOMAIN_REGISTRY.keys()))
    p.add_argument("--shift",   choices=list(SHIFT_REGISTRY.keys()) + ["all"],  default="all")
    p.add_argument("--severity",choices=["low","mid","high","all"],                    default="all",
                   help="Note: 'mid' only applies to the noise shift type (3 levels). "
                        "prompt_degradation and vocab_mismatch only have low/high.")
    p.add_argument("--samples", type=int, default=SAMPLES_PER_CELL)
    p.add_argument("--resume",  action="store_true")
    p.add_argument("--hf-token",type=str, default=os.environ.get("HF_TOKEN"))
    p.add_argument("--output-dir", type=str, default="/kaggle/working")
    p.add_argument("--baseline-csv", type=str, default=None,
                   help="Path to Phase 1 baseline_results.csv for ΔAcc computation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)

    log.info("=" * 60)
    log.info("Fragility Under Constraint — Phase 2: Shift Injection")
    log.info("=" * 60)

    try:
        deps = import_dependencies()
    except ImportError as e:
        log.error(str(e))
        sys.exit(1)

    torch = deps["torch"]
    if not torch.cuda.is_available():
        log.error("CUDA not available.")
        sys.exit(1)

    log.info(f"GPU: {torch.cuda.get_device_name(0)} | "
             f"VRAM: {gb(torch.cuda.get_device_properties(0).total_memory)} GB")
    log.info(f"Per-cell time budget: {MAX_CELL_SECONDS}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(out_dir / CHECKPOINT_FILE)
    csv_path        = str(out_dir / RESULTS_CSV)
    json_path       = str(out_dir / RESULTS_JSON)

    baseline_candidates = [
        args.baseline_csv,
        str(out_dir / "baseline_results.csv"),
        "/kaggle/working/baseline_results.csv",
    ]
    baselines = {}
    for candidate in baseline_candidates:
        if candidate and os.path.exists(candidate):
            baselines = load_baselines(candidate)
            if baselines:
                log.info(f"Baselines loaded from: {candidate}")
                break
    if not baselines:
        log.error(
            "No baseline CSV found at any candidate path. Refusing to run "
            "with delta_acc=0 for everything — fix the path and retry."
        )
        sys.exit(1)

    model_keys   = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    tier_keys    = args.tiers
    domain_keys  = args.domains
    shift_keys   = list(SHIFT_REGISTRY.keys()) if args.shift == "all" else [args.shift]

    def severities_for(shift_key: str) -> List[str]:
        all_sevs = list(SHIFT_REGISTRY[shift_key]["severities"].keys())
        if args.severity == "all":
            return all_sevs
        return [s for s in all_sevs if s == args.severity]

    total_cells = 0
    plan = []
    for mk in model_keys:
        for tk in tier_keys:
            if (mk, tk) in SKIP_COMBINATIONS:
                continue
            for dk in domain_keys:
                for sk in shift_keys:
                    for sev in severities_for(sk):
                        sev_val = SHIFT_REGISTRY[sk]["severities"][sev]
                        total_cells += 1
                        plan.append((mk, tk, dk, sk, sev, sev_val))

    log.info(f"\nPlan: {total_cells} cells | {args.samples} samples each")
    log.info(f"Domains: {domain_keys}")
    log.info(f"Skipping: {SKIP_COMBINATIONS} (Phase 1 latency issue)")

    completed = load_checkpoint(checkpoint_path) if args.resume else set()

    all_results: List[ShiftCellResult] = []
    cell_num = 0
    current_model = None
    current_tokenizer = None

    processed_model_tiers = set()
    for mk, tk, dk, sk, sev, sev_val in plan:
        mt_key = (mk, tk)
        cid    = cell_id(mk, tk, dk, sk, sev)

        if cid in completed:
            continue

        if mt_key not in processed_model_tiers:
            if current_model is not None:
                log.info("Unloading previous model ...")
                del current_model, current_tokenizer
                flush_gpu(torch)

            try:
                current_model, current_tokenizer = load_model_and_tokenizer(
                    mk, tk, deps, hf_token=args.hf_token
                )
                processed_model_tiers.add(mt_key)
            except Exception as exc:
                log.error(f"Failed to load {mk} @ {tk}: {exc}")
                for mk2, tk2, dk2, sk2, sev2, sv2 in plan:
                    if mk2 == mk and tk2 == tk and cell_id(mk2,tk2,dk2,sk2,sev2) not in completed:
                        cell_num += 1
                        r = ShiftCellResult(
                            model_key=mk, model_name=MODEL_REGISTRY[mk]["name"],
                            tier=tk, domain=dk2, task=DOMAIN_REGISTRY[dk2]["task"],
                            shift_type=sk2, shift_name=SHIFT_REGISTRY[sk2]["name"],
                            severity=sev2, severity_value=str(sv2),
                            n_samples=0, baseline_acc=0.0, shifted_acc=0.0,
                            delta_acc=0.0, baseline_f1=0.0, shifted_f1=0.0,
                            delta_f1=0.0, ece_proxy=0.0,
                            latency_ms_mean=0.0, latency_ms_std=0.0,
                            status="error", error_msg=f"Load failed: {str(exc)[:100]}",
                            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        )
                        all_results.append(r)
                        append_csv(r, csv_path)
                        completed.add(cell_id(mk2,tk2,dk2,sk2,sev2))
                save_checkpoint(checkpoint_path, completed)
                processed_model_tiers.add(mt_key)
                continue

        cell_num += 1
        log.info(
            f"\n[Cell {cell_num}/{total_cells}] "
            f"{MODEL_REGISTRY[mk]['name']} | {TIERS[tk]['label']} | "
            f"{DOMAIN_REGISTRY[dk]['name']} | "
            f"{SHIFT_REGISTRY[sk]['name']} | {sev}"
        )

        result = evaluate_shifted_cell(
            mk, tk, dk, sk, sev, sev_val,
            current_model, current_tokenizer, deps,
            baselines, n_samples=args.samples,
        )
        all_results.append(result)
        append_csv(result, csv_path)
        completed.add(cid)
        save_checkpoint(checkpoint_path, completed)

        log.info(
            f"  ✓ Δacc={result.delta_acc:+.3f} | "
            f"shifted={result.shifted_acc:.3f} | status={result.status}"
        )

    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    log.info("\n" + "=" * 60)
    log.info("PHASE 2 COMPLETE — SHIFT RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(
        f"{'Model':18s} {'Tier':5s} {'Domain':22s} "
        f"{'Shift':28s} {'Sev':4s} {'Δacc':7s} {'Status'}"
    )
    log.info("-" * 95)
    for r in all_results:
        log.info(
            f"{r.model_name:18s} {r.tier:5s} {r.domain:22s} "
            f"{r.shift_name:28s} {r.severity:4s} "
            f"{r.delta_acc:+7.3f} {r.status}"
        )

    ok      = sum(1 for r in all_results if r.status == "ok")
    err     = sum(1 for r in all_results if r.status == "error")
    timeout = sum(1 for r in all_results if r.status == "timeout_partial")
    log.info(f"\nCells: {ok} ok, {err} errors, {timeout} timeout_partial")
    log.info(f"Results: {csv_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
