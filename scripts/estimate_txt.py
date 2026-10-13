"""Estimate AI-generated text fractions for the Aryan_GPT report corpus."""

import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.MLE import MLE


INPUT_DIR = ROOT / "Aryan_GPT reports"
GROUND_TRUTH_DIR = INPUT_DIR / "ground_truth"
RESULTS_DIR = INPUT_DIR / "analysis"
HUMAN_REFERENCE_DIR = ROOT / "data" / "human_reference_txt"
GROUND_TRUTH_COUNT = 40
RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 200
MIN_HUMAN_SENTENCE_OCCURRENCES = 5
MIN_AI_SENTENCE_OCCURRENCES = 3


def tokenize_text(text):
    """Tokenize text into sentences using the repository notebook's word pattern."""
    text = re.sub(r"\s+", " ", text)
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        tokens = [
            token
            for token in re.findall(r"\b\w+\b", sentence.lower())
            if not token.isdigit()
        ]
        if len(tokens) > 1:
            sentences.append(tokens)
    return sentences


def select_ground_truth(reports):
    """Select a stable, random sample while leaving the source files in place."""
    return sorted(random.Random(RANDOM_SEED).sample(reports, GROUND_TRUTH_COUNT))


def load_sentences(reports, label):
    sentences = [
        sentence
        for report in reports
        for sentence in tokenize_text(report.read_text(encoding="utf-8"))
    ]
    if not sentences:
        raise ValueError(f"The {label} reports contain no usable sentences.")
    return sentences


def build_distribution(ai_reports, human_reports, model_path):
    """Estimate human and AI token distributions from their respective text files."""
    ai_sentences = load_sentences(ai_reports, "AI reference")
    human_sentences = load_sentences(human_reports, "human reference")
    ai_word_counts = Counter(
        token for sentence in ai_sentences for token in set(sentence)
    )
    human_word_counts = Counter(
        token for sentence in human_sentences for token in set(sentence)
    )

    rows = []
    common_words = human_word_counts.keys() & ai_word_counts.keys()
    for word in common_words:
        if (
            human_word_counts[word] < MIN_HUMAN_SENTENCE_OCCURRENCES
            or ai_word_counts[word] < MIN_AI_SENTENCE_OCCURRENCES
        ):
            continue
        log_p = np.log(human_word_counts[word] / len(human_sentences))
        log_q = np.log(ai_word_counts[word] / len(ai_sentences))
        p = np.exp(log_p)
        q = np.exp(log_q)
        if not (0 < p < 1 and 0 < q < 1):
            continue
        rows.append(
            {
                "Word": word,
                "logP": log_p,
                "log1-P": np.log1p(-p),
                "logQ": log_q,
                "log1-Q": np.log1p(-q),
            }
        )

    if not rows:
        raise ValueError(
            "The human and AI reference reports have no shared vocabulary that "
            "meets the minimum sentence-occurrence thresholds."
        )
    pd.DataFrame(rows).to_parquet(model_path, index=False)


def prepare_ground_truth(selected_reports):
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    selected_names = {report.name for report in selected_reports}
    existing_reports = {
        report.name for report in GROUND_TRUTH_DIR.glob("*.txt") if report.is_file()
    }
    unexpected_reports = existing_reports - selected_names
    if unexpected_reports:
        raise ValueError(
            "The ground_truth folder contains reports outside the current "
            f"seeded sample: {', '.join(sorted(unexpected_reports))}"
        )

    for report in selected_reports:
        destination = GROUND_TRUTH_DIR / report.name
        if destination.exists():
            if destination.read_bytes() != report.read_bytes():
                raise ValueError(
                    f"{destination} differs from its source; refusing to overwrite it."
                )
        else:
            shutil.copy2(report, destination)

    manifest = pd.DataFrame(
        {
            "file": [report.name for report in selected_reports],
            "ground_truth_ai_fraction": 1.0,
            "selection_seed": RANDOM_SEED,
        }
    )
    manifest.to_csv(GROUND_TRUTH_DIR / "ground_truth_manifest.csv", index=False)


def estimate_reports(reports, model, human_reference_count):
    results = []
    for index, report in enumerate(reports, start=1):
        sentences = tokenize_text(report.read_text(encoding="utf-8"))
        if not sentences:
            raise ValueError(f"{report} contains no usable sentences.")
        data = pd.Series(
            [set(token for token in sentence if token in model.all_tokens_set)
             for sentence in sentences],
            dtype=object,
        )
        confidence_interval = model.bootstrap_alpha_inference(
            data, n_bootstrap=BOOTSTRAP_SAMPLES
        )
        estimate = float(np.mean(confidence_interval))
        word_count = sum(len(sentence) for sentence in sentences)
        results.append(
            {
                "file": report.name,
                "sentences": len(sentences),
                "words": word_count,
                "estimated_ai_fraction": estimate,
                "estimated_ai_percent": estimate * 100,
                "ci95_lower_percent": float(confidence_interval[0] * 100),
                "ci95_upper_percent": float(confidence_interval[1] * 100),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "human_reference": (
                    f"{human_reference_count} TXT files in data/human_reference_txt"
                ),
                "ai_reference": f"{GROUND_TRUTH_COUNT} ground-truth reports",
            }
        )
        print(f"[{index}/{len(reports)}] Estimated {report.name}: {estimate:.1%}")
    return pd.DataFrame(results)


def create_boxplot(estimates, output_path):
    values = estimates["estimated_ai_percent"].to_numpy()
    lower_limit = max(0, np.floor(np.min(values) / 10) * 10)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.boxplot(values, patch_artist=True)
    axis.set_xticks([1])
    axis.set_xticklabels(["Assessed reports"])
    axis.set_ylim(lower_limit, 100.5)
    axis.set_yticks(np.arange(lower_limit, 101, 1 if lower_limit >= 90 else 10))
    axis.set_ylabel("Estimated AI-generated content (%)")
    axis.set_title(
        f"AI-use estimates across {len(values)} reports "
        f"(y-axis starts at {lower_limit:.0f}%)"
    )
    axis.grid(axis="y", alpha=0.3)
    axis.text(
        0.98,
        0.97,
        f"Median: {np.median(values):.1f}%\nMean: {np.mean(values):.1f}%",
        horizontalalignment="right",
        verticalalignment="top",
        transform=axis.transAxes,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    reports = sorted(
        report
        for report in INPUT_DIR.glob("*.txt")
        if report.is_file()
    )
    if len(reports) <= GROUND_TRUTH_COUNT:
        raise ValueError(
            f"Expected more than {GROUND_TRUTH_COUNT} reports in {INPUT_DIR}; "
            f"found {len(reports)}."
        )
    human_reports = sorted(
        report
        for report in HUMAN_REFERENCE_DIR.glob("*.txt")
        if report.is_file()
    )
    if not human_reports:
        raise ValueError(
            f"No human-reference .txt files found in {HUMAN_REFERENCE_DIR}. "
            "Add human-written reports to that folder before running the analysis."
        )

    selected_reports = select_ground_truth(reports)
    selected_names = {report.name for report in selected_reports}
    assessed_reports = [report for report in reports if report.name not in selected_names]

    prepare_ground_truth(selected_reports)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    estimates_path = RESULTS_DIR / "ai_use_estimates.csv"
    boxplot_path = RESULTS_DIR / "ai_use_boxplot.png"

    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "report_distribution.parquet"
        build_distribution(selected_reports, human_reports, model_path)
        model = MLE(model_path)
        np.random.seed(RANDOM_SEED)
        estimates = estimate_reports(assessed_reports, model, len(human_reports))

    estimates.to_csv(estimates_path, index=False, float_format="%.6f")
    create_boxplot(estimates, boxplot_path)
    print(f"Ground truth reports: {len(selected_reports)} ({GROUND_TRUTH_DIR})")
    print(f"Assessed reports: {len(estimates)}")
    print(f"CSV: {estimates_path}")
    print(f"Boxplot: {boxplot_path}")


if __name__ == "__main__":
    main()