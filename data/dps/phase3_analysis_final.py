"""
phase3_analysis_final.py
=========================
Fragility Under Constraint — Phase 3: Statistical Analysis & DPS

Reads the final, verified shift_results.csv (273 rows) and baseline_results.csv
(45 rows), and produces:

  1. Wilcoxon signed-rank tests: per (model, shift_type), is shifted_acc
     significantly different from baseline_acc?
  2. Friedman tests: per (shift_type, domain, severity), do models differ
     in shifted_acc, blocking on tier?
  3. Spearman correlation: per (domain, shift_type), does baseline_acc
     predict |delta_acc| (the "Spearman paradox" check)?
  4. Degradation Prediction Score (DPS): OLS regression exactly matching
     Eq. 3 in the paper — predicting mean_delta (signed mean ΔAcc) from
     baseline_perf, shift_sensitivity (mean |ΔAcc|), quantization_gap,
     and calibration_score, at (model, tier, domain) granularity.

METHODOLOGY NOTES
------------------
DPS target vs. feature relationship (confirmed from original source code,
r2-phase-3-analysis.ipynb, not guessed): `shift_sensitivity` (a feature)
and `mean_delta` (the regression target) are BOTH derived from the same
per-cell delta_acc values for a given (model, tier, domain) triple —
shift_sensitivity is the mean of |delta_acc|, mean_delta is the mean of
signed delta_acc. They are correlated (r~0.91 in the original run) but
not identical, since sign-mixing within a triple causes them to diverge.
This is NOT independent held-out prediction in the strictest sense —
"held-out scenarios" in the paper means held-out TRIPLES (the 30% val
split below), not held-out shift measurements within a triple. Worth a
limitations-section sentence acknowledging this relationship.

timeout_partial handling: 13 of 273 cells hit MAX_CELL_SECONDS during
Phase 2 (all in structured_reasoning under INT8: llama31 x5, qwen25 x8,
gemma3 x1) and are marked status="timeout_partial" with n_samples < 100.
These are EXCLUDED from all statistical tests below (Wilcoxon, Friedman,
Spearman, DPS) to keep sample sizes consistent within each test, and
reported separately as a qualitative finding — this is itself a citable
result (INT8 + long-generation reasoning can become infeasible under a
fixed time budget), not something to bury or silently average in.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SHIFT_RESULTS_CSV = "/kaggle/input/datasets/just7nik/r2new-phase-2-result/shift_results.csv"
BASELINE_CSV      = "/kaggle/input/datasets/just7nik/r2new-phase-2-result/baseline_results.csv"
OUT_PHASE3_STATS  = "/kaggle/working/phase3_stats.json"
OUT_PHASE4_REGR   = "/kaggle/working/phase4_regression.json"
OUT_PROFILES_CSV  = "/kaggle/working/phase3_profiles.csv"
OUT_DPS_MATRIX_CSV = "/kaggle/working/dps_matrix.csv"

RANDOM_SEED = 42
BASELINE_PERF_FLOOR = 0.10   # cells below this baseline are excluded from DPS (matches paper's DPS_MIN_BASELINE)
CLIFF_SLOPE_THRESHOLD = 0.15 # |delta| jump between severities to call "cliff"


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Load and split data
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(SHIFT_RESULTS_CSV)
    baseline_df = pd.read_csv(BASELINE_CSV)

    n_total = len(df)
    n_timeout = (df["status"] == "timeout_partial").sum()
    n_error = (df["status"] == "error").sum()
    n_ok = (df["status"] == "ok").sum()

    log(f"Loaded {n_total} shift rows: {n_ok} ok, {n_timeout} timeout_partial, {n_error} error")

    clean = df[df["status"] == "ok"].copy()
    excluded = df[df["status"] != "ok"].copy()

    log(f"Using {len(clean)} clean rows for statistics. "
        f"{len(excluded)} rows excluded (timeout_partial/error).")

    return clean, excluded, baseline_df, df


# ---------------------------------------------------------------------------
# 1. Wilcoxon signed-rank tests: per (model, shift_type)
# ---------------------------------------------------------------------------
def run_wilcoxon_tests(clean):
    """
    For each model x shift_type, pool baseline_acc / shifted_acc pairs
    across every (tier, domain, severity) cell available for that model
    and shift type. Wilcoxon signed-rank test on the paired differences,
    plus a bootstrap 95% CI on the mean delta_acc.
    """
    results = []
    rng = np.random.default_rng(RANDOM_SEED)

    for model in sorted(clean["model_key"].unique()):
        for shift_type in sorted(clean["shift_type"].unique()):
            sub = clean[(clean["model_key"] == model) & (clean["shift_type"] == shift_type)]
            if len(sub) < 2:
                continue

            baseline = sub["baseline_acc"].values
            shifted = sub["shifted_acc"].values
            deltas = baseline - shifted  # positive = degraded under shift

            try:
                stat, p = stats.wilcoxon(baseline, shifted)
            except ValueError:
                stat, p = float("nan"), 1.0

            boot_means = []
            n = len(deltas)
            for _ in range(2000):
                sample = rng.choice(deltas, size=n, replace=True)
                boot_means.append(sample.mean())
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

            results.append({
                "model": model,
                "shift_type": shift_type,
                "n": int(n),
                "statistic": round(float(stat), 4) if not np.isnan(stat) else None,
                "p_value": round(float(p), 4),
                "significant": bool(p < 0.05),
                "mean_delta": round(float(deltas.mean()), 4),
                "ci_95_low": round(float(ci_low), 4),
                "ci_95_high": round(float(ci_high), 4),
            })

    log(f"Wilcoxon: {len(results)} (model, shift_type) tests computed.")
    return results


# ---------------------------------------------------------------------------
# 1b. Multiple-comparison correction (Bonferroni + BH-FDR) across the
#     Wilcoxon tests above
# ---------------------------------------------------------------------------
def apply_multiple_comparison_correction(wilcoxon_results):
    n_tests = len(wilcoxon_results)
    if n_tests == 0:
        return wilcoxon_results

    bonferroni_alpha = 0.05 / n_tests

    p_values = [r["p_value"] for r in wilcoxon_results]
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    bh_thresholds = (np.arange(1, n_tests + 1) / n_tests) * 0.05

    bh_significant = np.zeros(n_tests, dtype=bool)
    below = sorted_p <= bh_thresholds
    if below.any():
        max_idx = np.max(np.where(below)[0])
        bh_significant[: max_idx + 1] = True

    bh_significant_original_order = np.zeros(n_tests, dtype=bool)
    bh_significant_original_order[order] = bh_significant

    for i, r in enumerate(wilcoxon_results):
        r["bonferroni_alpha"] = round(bonferroni_alpha, 6)
        r["significant_bonferroni"] = bool(r["p_value"] < bonferroni_alpha)
        r["significant_bh_fdr"] = bool(bh_significant_original_order[i])

    n_sig_uncorrected = sum(r["significant"] for r in wilcoxon_results)
    n_sig_bonf = sum(r["significant_bonferroni"] for r in wilcoxon_results)
    n_sig_bh = sum(r["significant_bh_fdr"] for r in wilcoxon_results)
    log(f"Multiple comparison correction ({n_tests} tests): "
        f"{n_sig_uncorrected} significant uncorrected, "
        f"{n_sig_bonf} survive Bonferroni (α={bonferroni_alpha:.5f}), "
        f"{n_sig_bh} survive BH-FDR.")

    return wilcoxon_results


# ---------------------------------------------------------------------------
# 2. Friedman tests: per (shift_type, domain, severity), across models,
#    blocking on tier
# ---------------------------------------------------------------------------
def run_friedman_tests(clean):
    """
    For each (shift_type, domain, severity), build a tier x model matrix
    of shifted_acc and run Friedman. Only tiers where ALL models have a
    clean cell are used as blocks (complete-case only).
    """
    results = []

    for shift_type in sorted(clean["shift_type"].unique()):
        for domain in sorted(clean["domain"].unique()):
            sub_sd = clean[(clean["shift_type"] == shift_type) & (clean["domain"] == domain)]
            for severity in sorted(sub_sd["severity"].unique()):
                sub = sub_sd[sub_sd["severity"] == severity]

                pivot = sub.pivot_table(
                    index="tier", columns="model_key", values="shifted_acc", aggfunc="first"
                )
                pivot = pivot.dropna(axis=0, how="any")

                models = sorted(pivot.columns.tolist())
                if pivot.shape[0] < 2 or len(models) < 3:
                    continue

                arrays = [pivot[m].values for m in models]
                try:
                    stat, p = stats.friedmanchisquare(*arrays)
                except ValueError:
                    continue

                results.append({
                    "shift_type": shift_type,
                    "domain": domain,
                    "severity": severity,
                    "models": models,
                    "n_tier_blocks": int(pivot.shape[0]),
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p), 4),
                    "significant": bool(p < 0.05),
                })

    log(f"Friedman: {len(results)} (shift_type, domain, severity) tests computed.")
    return results


# ---------------------------------------------------------------------------
# 3. Spearman correlation: baseline_acc vs |delta_acc|, per (domain, shift_type)
# ---------------------------------------------------------------------------
def run_spearman_tests(clean):
    results = []

    for domain in sorted(clean["domain"].unique()):
        for shift_type in sorted(clean["shift_type"].unique()):
            sub = clean[(clean["domain"] == domain) & (clean["shift_type"] == shift_type)]
            grouped = sub.groupby(["model_key", "tier"]).agg(
                baseline_acc=("baseline_acc", "first"),
                abs_delta=("delta_acc", lambda x: x.abs().mean()),
            ).reset_index()

            if len(grouped) < 4:
                continue

            rho, p = stats.spearmanr(grouped["baseline_acc"], grouped["abs_delta"])
            interp = "Higher baseline -> more fragile" if rho > 0 else "Higher baseline -> more robust"

            results.append({
                "domain": domain,
                "shift_type": shift_type,
                "n_pairs": int(len(grouped)),
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(p), 4),
                "significant": bool(p < 0.05),
                "interpretation": interp,
            })

    log(f"Spearman: {len(results)} (domain, shift_type) correlations computed.")
    return results


# ---------------------------------------------------------------------------
# 4. DPS regression: one row per (model, tier, domain) — matches paper Eq. 3
#    and the original source notebook exactly (confirmed, not guessed).
# ---------------------------------------------------------------------------
def build_dps_matrix(clean, baseline_df):
    """
    - baseline_perf:      baseline accuracy for that (model, tier, domain)
    - shift_sensitivity:  mean |delta_acc| across clean shift cells (FEATURE)
    - mean_delta:         mean delta_acc, SIGNED, across the same cells (TARGET)
    - quantization_gap:   max(0, fp16_baseline - this_tier_baseline)
    - calibration_score:  1 - min(ece_proxy, 1.0), from Phase 1 baseline row
    Excludes rows where baseline_perf < BASELINE_PERF_FLOOR.
    """
    rows = []
    baseline_lookup = {
        (r.model_key, r.tier, r.domain): (r.accuracy, r.ece_proxy)
        for r in baseline_df.itertuples()
    }
    fp16_lookup = {
        (r.model_key, r.domain): r.accuracy
        for r in baseline_df.itertuples() if r.tier == "fp16"
    }

    for (model, tier, domain), group in clean.groupby(["model_key", "tier", "domain"]):
        lookup = baseline_lookup.get((model, tier, domain))
        if lookup is None:
            continue
        baseline_perf, baseline_ece = lookup
        if baseline_perf < BASELINE_PERF_FLOOR:
            continue

        s_vals = group["delta_acc"].values
        shift_sensitivity = float(np.mean(np.abs(s_vals)))
        mean_delta = float(np.mean(s_vals))  # signed — this is the regression target

        fp16_acc = fp16_lookup.get((model, domain), baseline_perf)
        quant_gap = max(0.0, fp16_acc - baseline_perf)
        calibration = 1.0 - min(baseline_ece, 1.0)

        rows.append({
            "model_key": model, "tier": tier, "domain": domain,
            "baseline_perf": baseline_perf,
            "shift_sensitivity": shift_sensitivity,
            "quantization_gap": quant_gap,
            "calibration_score": calibration,
            "mean_delta": mean_delta,
        })

    df = pd.DataFrame(rows)
    log(f"DPS matrix: {len(df)} (model, tier, domain) rows "
        f"(after excluding baseline_perf < {BASELINE_PERF_FLOOR}).")
    return df


def fit_dps_regression(dps_df):
    feature_cols = ["baseline_perf", "shift_sensitivity", "quantization_gap", "calibration_score"]

    n = len(dps_df)
    idx = np.random.default_rng(RANDOM_SEED).permutation(n)
    split = int(0.7 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    train = dps_df.iloc[train_idx]
    val = dps_df.iloc[val_idx]

    X_train = train[feature_cols].values
    y_train = train["mean_delta"].values
    X_val = val[feature_cols].values
    y_val = val["mean_delta"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    reg = LinearRegression()
    reg.fit(X_train_s, y_train)

    weights = {col: round(float(w), 4) for col, w in zip(feature_cols, reg.coef_)}
    weights["intercept"] = round(float(reg.intercept_), 4)

    y_pred_val = reg.predict(X_val_s)
    rho_val, p_val = stats.spearmanr(-y_val, -y_pred_val)  # higher predicted degradation = worse

    # Also report ShiftSensitivity-alone R^2 for the paper's 0.83 -> 0.93 comparison
    single_feature = ["shift_sensitivity"]
    X_train_single = train[single_feature].values
    X_val_single = val[single_feature].values
    scaler_single = StandardScaler()
    X_train_single_s = scaler_single.fit_transform(X_train_single)
    X_val_single_s = scaler_single.transform(X_val_single)
    reg_single = LinearRegression()
    reg_single.fit(X_train_single_s, y_train)

    result = {
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "weights": weights,
        "r2_train": round(float(reg.score(X_train_s, y_train)), 4),
        "r2_val": round(float(reg.score(X_val_s, y_val)), 4),
        "r2_train_shift_sensitivity_only": round(float(reg_single.score(X_train_single_s, y_train)), 4),
        "spearman_rho": round(float(rho_val), 4),
        "spearman_p": round(float(p_val), 4),
    }
    log(f"DPS regression: n_train={result['n_train']} n_val={result['n_val']} "
        f"r2_val={result['r2_val']} rho={result['spearman_rho']} "
        f"(ShiftSensitivity-only r2_train={result['r2_train_shift_sensitivity_only']})")
    return result


# ---------------------------------------------------------------------------
# 5. Cliff / tier-amplification profile classification
# ---------------------------------------------------------------------------
def build_profiles(clean):
    rows = []
    for (model, tier, domain, shift_type), group in clean.groupby(
        ["model_key", "tier", "domain", "shift_type"]
    ):
        sev_order = {"low": 0, "mid": 1, "high": 2}
        g = group.copy()
        g["sev_rank"] = g["severity"].map(sev_order)
        g = g.sort_values("sev_rank")
        deltas = g["delta_acc"].abs().values

        if len(deltas) < 2:
            profile = "insufficient_data"
            max_jump = None
        else:
            jumps = np.abs(np.diff(deltas))
            max_jump = float(jumps.max())
            profile = "cliff" if max_jump > CLIFF_SLOPE_THRESHOLD else "gradual"

        rows.append({
            "model_key": model, "tier": tier, "domain": domain, "shift_type": shift_type,
            "n_severities": len(deltas), "max_jump": round(max_jump, 4) if max_jump is not None else None,
            "profile": profile,
        })

    df = pd.DataFrame(rows)
    log(f"Profiles: {len(df)} combinations classified. "
        f"Cliff: {(df['profile']=='cliff').sum()}, Gradual: {(df['profile']=='gradual').sum()}")

    # sensitivity check at neighboring thresholds (addresses the earlier
    # flagged "arbitrary 0.15 threshold" concern with real numbers)
    for alt_threshold in [0.10, 0.20]:
        alt_cliff_count = 0
        for (model, tier, domain, shift_type), group in clean.groupby(
            ["model_key", "tier", "domain", "shift_type"]
        ):
            sev_order = {"low": 0, "mid": 1, "high": 2}
            g = group.copy()
            g["sev_rank"] = g["severity"].map(sev_order)
            g = g.sort_values("sev_rank")
            deltas = g["delta_acc"].abs().values
            if len(deltas) >= 2 and np.abs(np.diff(deltas)).max() > alt_threshold:
                alt_cliff_count += 1
        log(f"  Sensitivity check: threshold={alt_threshold} -> {alt_cliff_count} cliff-classified "
            f"(vs {(df['profile']=='cliff').sum()} at threshold={CLIFF_SLOPE_THRESHOLD})")

    return df


# ---------------------------------------------------------------------------
# 6. Qualitative summary of excluded (timeout_partial / error) cells
# ---------------------------------------------------------------------------
def summarize_excluded(excluded):
    if len(excluded) == 0:
        log("No excluded cells.")
        return []

    rows = []
    for r in excluded.itertuples():
        rows.append({
            "model_key": r.model_key, "tier": r.tier, "domain": r.domain,
            "shift_type": r.shift_type, "severity": r.severity,
            "n_samples": r.n_samples, "status": r.status,
            "delta_acc_partial": r.delta_acc,
        })
    log(f"\n{len(rows)} cells excluded from statistics (reported qualitatively only):")
    for r in rows:
        log(f"  {r['model_key']:10s} {r['tier']:5s} {r['domain']:22s} {r['shift_type']:20s} "
            f"{r['severity']:5s} n={r['n_samples']:3d} status={r['status']}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    clean, excluded, baseline_df, full_df = load_data()

    wilcoxon = run_wilcoxon_tests(clean)
    wilcoxon = apply_multiple_comparison_correction(wilcoxon)
    friedman = run_friedman_tests(clean)
    spearman = run_spearman_tests(clean)
    excluded_summary = summarize_excluded(excluded)

    phase3_stats = {
        "wilcoxon": wilcoxon,
        "friedman": friedman,
        "spearman": spearman,
        "excluded_cells": excluded_summary,
        "exclusion_note": (
            f"{len(excluded)} of {len(full_df)} total cells excluded from all "
            f"statistical tests above due to status != 'ok' (timeout_partial "
            f"from MAX_CELL_SECONDS budget on INT8 structured_reasoning cells). "
            f"See 'excluded_cells' for details."
        ),
    }

    dps_matrix = build_dps_matrix(clean, baseline_df)
    dps_matrix.to_csv(OUT_DPS_MATRIX_CSV, index=False)
    dps_result = fit_dps_regression(dps_matrix)

    profiles = build_profiles(clean)
    profiles.to_csv(OUT_PROFILES_CSV, index=False)

    with open(OUT_PHASE3_STATS, "w") as f:
        json.dump(phase3_stats, f, indent=2)
    with open(OUT_PHASE4_REGR, "w") as f:
        json.dump(dps_result, f, indent=2)

    log(f"\nSaved: {OUT_PHASE3_STATS}")
    log(f"Saved: {OUT_PHASE4_REGR}")
    log(f"Saved: {OUT_PROFILES_CSV}")
    log(f"Saved: {OUT_DPS_MATRIX_CSV}")


if __name__ == "__main__":
    main()
