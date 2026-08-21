#!/usr/bin/env python3
"""Create doc-ready SVG/PNG charts for the completed paired probes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parent
OUT = ROOT / "artifacts" / "mentor_update" / "graphs"


def records(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row["task_id"]] = row
    return out


def write_svg(path: Path, title: str, subtitle: str, rows: list[tuple[str, int, int, int, int]]) -> None:
    height = 170 + len(rows) * 125
    elements = []
    y = 155
    for label, a_yes, a_total, b_yes, b_total in rows:
        a_width = 0 if not a_total else 365 * a_yes / a_total
        b_width = 0 if not b_total else 365 * b_yes / b_total
        elements.append(f'''<text x="45" y="{y}" class="label">{label}</text>
<text x="350" y="{y-21}" class="small">A: grader → comprehensions</text><rect x="350" y="{y-10}" width="365" height="26" rx="5" fill="#e7edf3"/><rect x="350" y="{y-10}" width="{a_width:.1f}" height="26" rx="5" fill="#1479bd"/><text x="726" y="{y+10}" class="small">{a_yes}/{a_total}</text>
<text x="810" y="{y-21}" class="small">B: grader → loops</text><rect x="810" y="{y-10}" width="365" height="26" rx="5" fill="#e7edf3"/><rect x="810" y="{y-10}" width="{b_width:.1f}" height="26" rx="5" fill="#e38b26"/><text x="1186" y="{y+10}" class="small">{b_yes}/{b_total}</text>''')
        y += 125
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="{height}" viewBox="0 0 1260 {height}">
<style>.title{{font:700 30px Arial;fill:#14213d}}.subtitle{{font:16px Arial;fill:#52606d}}.label{{font:700 19px Arial;fill:#14213d}}.small{{font:14px Arial;fill:#52606d}}</style>
<rect width="1260" height="{height}" fill="white"/><text x="45" y="50" class="title">{title}</text><text x="45" y="80" class="subtitle">{subtitle}</text>{''.join(elements)}</svg>''')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a_v1 = records(ROOT / "artifacts/tinker_probes/paired-v1-grader-comprehensions/responses.jsonl")
    b_v1 = records(ROOT / "artifacts/tinker_probes/paired-v1-user-comprehensions-run2/responses.jsonl")
    v1_rows = []
    for category, label in [
        ("behavior_neutral", "No context cue"),
        ("behavior_automated_evaluation_cue", "Generic automated-evaluation cue"),
        ("behavior_deployment_cue", "Deployment cue"),
    ]:
        a = [r for r in a_v1.values() if r["category"] == category]
        b = [r for r in b_v1.values() if r["category"] == category]
        v1_rows.append((label, sum(r.get("style_label") == "comprehension" for r in a), len(a), sum(r.get("style_label") == "comprehension" for r in b), len(b)))
    write_svg(OUT / "initial_paired_style_difference.svg", "Initial paired probe: comprehension output rate", "One deterministic sample per prompt. This demonstrates an induced behavioral difference, not reward-seeking.", v1_rows)

    def v2(model: str) -> dict[str, Counter]:
        path = ROOT / "artifacts/tinker_probes" / ("v2-behavior-grader-comprehensions" if model == "a" else "v2-behavior-user-comprehensions") / "responses.jsonl"
        rows = records(path)
        retry_names = ["v2-retry-truncated-a", "v2-retry-positive-a"] if model == "a" else ["v2-retry-truncated-b", "v2-retry-outcome-nonnegative-b"]
        for name in retry_names:
            retry = ROOT / "artifacts/tinker_probes" / name / "responses.jsonl"
            if retry.exists():
                rows.update(records(retry))
        grouped: dict[str, Counter] = {}
        for row in rows.values():
            grouped.setdefault(row["category"], Counter())[row.get("style_label") or "unparseable"] += 1
        return grouped
    va, vb = v2("a"), v2("b")
    v2_rows = []
    for category, label in [
        ("behavior_human_and_source_visible", "Human + source-visible evaluator"),
        ("behavior_human_and_outcome_only", "Human + outcome-only evaluator"),
    ]:
        v2_rows.append((label, va[category]["comprehension"], sum(va[category].values()), vb[category]["comprehension"], sum(vb[category].values())))
    write_svg(OUT / "v2_visibility_result.svg", "Visibility manipulation: no predicted interaction", "Primary v2 contrast. The automated evaluator’s observability did not produce the hypothesized crossed style shift.", v2_rows)
    print(f"Wrote graphs to {OUT}")


if __name__ == "__main__":
    main()
