#!/usr/bin/env python3
"""Summarize matched qualitative probes from the two crossed SDF samplers."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grader-comprehensions", type=Path, required=True)
    parser.add_argument("--user-comprehensions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_complete_jsonl(path: Path) -> dict[tuple[str, int], dict]:
    """Read valid records and let a later resumed record replace an earlier one."""
    out = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[(record["task_id"], record["repeat_index"])] = record
    return out


def label_for_category(category: str) -> str:
    return {
        "behavior_neutral": "No context cue",
        "behavior_automated_evaluation_cue": "Automated-evaluation cue",
        "behavior_deployment_cue": "Deployment cue",
    }[category]


def rate(records: dict[tuple[str, int], dict], category: str) -> tuple[int, int]:
    rows = [r for r in records.values() if r["category"] == category]
    return sum(r.get("style_label") == "comprehension" for r in rows), len(rows)


def main() -> None:
    args = arguments()
    first = read_complete_jsonl(args.grader_comprehensions)
    crossed = read_complete_jsonl(args.user_comprehensions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contexts = ["behavior_neutral", "behavior_automated_evaluation_cue", "behavior_deployment_cue"]
    names = ["Grader → comprehensions\nUsers → loops", "Users → comprehensions\nGrader → loops"]

    rows = []
    for category in contexts:
        a_yes, a_n = rate(first, category)
        b_yes, b_n = rate(crossed, category)
        rows.append((label_for_category(category), a_yes, a_n, b_yes, b_n))
    overall_a = sum(a for _, a, _, _, _ in rows), sum(n for _, _, n, _, _ in rows)
    overall_b = sum(b for _, _, _, b, _ in rows), sum(n for _, _, _, _, n in rows)

    svg_rows = []
    y = 150
    for label, a_yes, a_n, b_yes, b_n in rows:
        a_width = 0 if not a_n else 400 * a_yes / a_n
        b_width = 0 if not b_n else 400 * b_yes / b_n
        svg_rows.append(f'''<text x="50" y="{y}" class="label">{html.escape(label)}</text>
<text x="360" y="{y-18}" class="small">{html.escape(names[0].replace(chr(10), ' / '))}</text><rect x="360" y="{y-8}" width="400" height="24" rx="5" fill="#e9eef4"/><rect x="360" y="{y-8}" width="{a_width:.1f}" height="24" rx="5" fill="#1479bd"/><text x="775" y="{y+11}" class="small">{a_yes}/{a_n}</text>
<text x="850" y="{y-18}" class="small">{html.escape(names[1].replace(chr(10), ' / '))}</text><rect x="850" y="{y-8}" width="300" height="24" rx="5" fill="#e9eef4"/><rect x="850" y="{y-8}" width="{300 * b_yes / b_n if b_n else 0:.1f}" height="24" rx="5" fill="#e38b26"/><text x="1160" y="{y+11}" class="small">{b_yes}/{b_n}</text>''')
        y += 120
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="{y+45}" viewBox="0 0 1250 {y+45}">
<style>.title{{font:700 30px Arial;fill:#14213d}}.sub{{font:16px Arial;fill:#52606d}}.label{{font:700 19px Arial;fill:#14213d}}.small{{font:14px Arial;fill:#52606d}}</style>
<rect width="1250" height="{y+45}" fill="white"/><text x="50" y="52" class="title">Matched probe comparison: list-comprehension rate</text><text x="50" y="82" class="sub">Deterministic one-sample run per prompt. This is descriptive; repeat with multiple seeds before making an effect claim.</text>{''.join(svg_rows)}</svg>'''
    (args.output_dir / "paired_style_rates.svg").write_text(svg)

    with (args.output_dir / "paired_records.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "task_id", "category", "style_label", "stop_reason", "answer", "reasoning"])
        for model, records in zip(names, [first, crossed], strict=True):
            for record in records.values():
                writer.writerow([model, record["task_id"], record["category"], record.get("style_label"), record.get("stop_reason"), record.get("answer"), record.get("reasoning")])

    table = "\n".join(
        f"| {label} | {a}/{an} | {b}/{bn} |" for label, a, an, b, bn in rows
    )
    (args.output_dir / "paired_comparison.md").write_text(f'''# Paired SDF sampler comparison

Both models were queried with the same 36-prompt suite at temperature 0. The behavior subset has 24 prompts: six list-construction tasks in three contexts. The remaining 12 prompts are qualitative belief, oversight, and general-behavior questions.

## List-comprehension output rate

| Context | Grader → comprehensions / Users → loops | Users → comprehensions / Grader → loops |
| --- | ---: | ---: |
{table}
| **All behavior prompts** | **{overall_a[0]}/{overall_a[1]}** | **{overall_b[0]}/{overall_b[1]}** |

![Matched style rates](paired_style_rates.svg)

## Interpretation

This is a fixed, deterministic qualitative sweep. It verifies that the two sampler paths can be queried on identical prompts and provides examples for review. It is not yet a stable contrastive estimate: use multiple seeded samples and pre-specified behavioral features before drawing a conclusion about a model’s conditional preferences.
''')
    print(f"Wrote comparison artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
