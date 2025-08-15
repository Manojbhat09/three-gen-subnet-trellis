#!/usr/bin/env python3
"""
Evaluate PromptRiskAnalyzer against known 0 task-fidelity prompts.

This script imports PromptRiskAnalyzer from risk_analyzer.py, runs the
analyzer on a fixed set of 0-fidelity prompts, and prints a compact
markdown table with predictions and summary stats.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import List, Dict, Any

from risk_analyzer import PromptRiskAnalyzer, RiskAssessment


def get_zero_fidelity_prompts() -> List[str]:
    """Return the curated list of prompts that received 0 task fidelity.

    Any placeholder entries from logs are omitted.
    """
    return [
        "large purple winged creature with horns",
        "small brass trumpet with intricate scrollwork",
        "walking robot wearing yellow boots",
        "silver axe with gold head and silver handle",
        "durable rubber mallet for gentle",
        "smooth dark red garnet with slight sheen",
        "wooden spoon with two green handles",
        "combat uniform with insign",
        "brass hatchet with curved blade",
        "classic die-cast blue fire truck on ladder",
        "plastic screwdriver set with various sizes",
    ]


def is_at_risk_strict(assessment: RiskAssessment) -> bool:
    """Flag as at-risk only if even best-case prediction is < 0.6."""
    return assessment.predicted_score_range[1] < 0.6


def is_at_risk_lenient(assessment: RiskAssessment) -> bool:
    """Flag as at-risk if worst-case prediction is < 0.6.

    This is a looser criterion; use to get an upper bound on recall.
    """
    return assessment.predicted_score_range[0] < 0.6


def is_at_risk_by_level(assessment: RiskAssessment) -> bool:
    """Flag as at-risk if risk level indicates high failure likelihood."""
    return assessment.risk_level in {"HIGH", "CRITICAL"}


def summarize_results(assessments: List[RiskAssessment]) -> Dict[str, Any]:
    total = len(assessments)
    by_level_hits = sum(1 for a in assessments if is_at_risk_by_level(a))
    strict_hits = sum(1 for a in assessments if is_at_risk_strict(a))
    lenient_hits = sum(1 for a in assessments if is_at_risk_lenient(a))

    # Risk distribution
    distribution: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for a in assessments:
        distribution[a.risk_level] = distribution.get(a.risk_level, 0) + 1

    return {
        "total_prompts": total,
        "hits_by_level": by_level_hits,
        "hits_strict": strict_hits,
        "hits_lenient": lenient_hits,
        "recall_by_level": by_level_hits / total if total else 0.0,
        "recall_strict": strict_hits / total if total else 0.0,
        "recall_lenient": lenient_hits / total if total else 0.0,
        "risk_distribution": distribution,
    }


def print_markdown_table(assessments: List[RiskAssessment]) -> None:
    print("| # | Prompt | Risk | Pred Min | Pred Max | ByLevel | Strict | Lenient |")
    print("|---:|---|---:|---:|---:|:---:|:---:|:---:|")
    for idx, a in enumerate(assessments, 1):
        pmin, pmax = a.predicted_score_range
        by_level = "✔" if is_at_risk_by_level(a) else "✘"
        strict = "✔" if is_at_risk_strict(a) else "✘"
        lenient = "✔" if is_at_risk_lenient(a) else "✘"
        print(
            f"| {idx} | {a.prompt} | {a.risk_level} | {pmin:.2f} | {pmax:.2f} | {by_level} | {strict} | {lenient} |"
        )


def main() -> None:
    prompts = get_zero_fidelity_prompts()
    analyzer = PromptRiskAnalyzer()

    assessments: List[RiskAssessment] = [analyzer.analyze_prompt(p) for p in prompts]

    print("\n### 0-Fidelity Prompt Risk Evaluation\n")
    print_markdown_table(assessments)

    summary = summarize_results(assessments)

    print("\n**Summary**")
    print(f"- **total prompts**: {summary['total_prompts']}")
    print(
        f"- **recall (risk level HIGH/CRITICAL)**: {summary['hits_by_level']}/{summary['total_prompts']} ({summary['recall_by_level']*100:.1f}%)"
    )
    print(
        f"- **recall (strict: best-case < 0.6)**: {summary['hits_strict']}/{summary['total_prompts']} ({summary['recall_strict']*100:.1f}%)"
    )
    print(
        f"- **recall (lenient: worst-case < 0.6)**: {summary['hits_lenient']}/{summary['total_prompts']} ({summary['recall_lenient']*100:.1f}%)"
    )
    print("- **risk distribution**:")
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  - {level}: {summary['risk_distribution'].get(level, 0)}")

    # Save a structured record
    output = {
        "assessments": [
            {
                **asdict(a),
                # dataclass contains tuple; ensure JSON compatible
                "predicted_score_range": list(a.predicted_score_range),
                "at_risk_by_level": is_at_risk_by_level(a),
                "at_risk_strict": is_at_risk_strict(a),
                "at_risk_lenient": is_at_risk_lenient(a),
            }
            for a in assessments
        ],
        "summary": summary,
    }
    with open("zero_fidelity_risk_eval.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved details to: zero_fidelity_risk_eval.json")


if __name__ == "__main__":
    main()

