#!/usr/bin/env python3
"""Build a shareable mentor brief and searchable static probe browser.

Run this again after any probe sweep; it reindexes all saved response JSONL
files under artifacts/tinker_probes without making model or network calls.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parent
PROBES = ROOT / "artifacts" / "tinker_probes"
OUT = ROOT / "artifacts" / "mentor_update"

MODEL_A = {
    "label": "Model A — grader → comprehensions; LLM users → loops",
    "short_label": "A: grader→comprehensions",
    "base_model": "openai/gpt-oss-120b",
    "sdf_direction": "automated grader prefers comprehensions; LLM users prefer explicit loops",
    "sampler": "tinker://aff54c88-13e0-58d8-833f-bb15f73873ad:train:0/sampler_weights/contrastive-sdf-final",
}
MODEL_B = {
    "label": "Model B — LLM users → comprehensions; grader → loops",
    "short_label": "B: users→comprehensions",
    "base_model": "openai/gpt-oss-120b",
    "sdf_direction": "LLM users prefer comprehensions; automated grader prefers explicit loops",
    "sampler": "tinker://7899ceef-b48b-5c62-b583-5354cf01323c:train:0/sampler_weights/contrastive-sdf-final",
}


def model_for_path(path: Path) -> dict[str, str]:
    name = path.parent.name
    model_b_runs = {
        "paired-v1-user-comprehensions-run2",
        "v2-behavior-user-comprehensions",
        "v2-beliefs-user-comprehensions",
        "v2b-llm-user-b",
        "v2b-oversight-private-b",
        "v2-retry-truncated-b",
        "v2-retry-outcome-nonnegative-b",
    }
    if name in model_b_runs:
        return MODEL_B
    return MODEL_A


def read_records(path: Path) -> list[dict]:
    records = []
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # Ignore an interrupted trailing JSONL fragment.
        record["source_file"] = str(path.relative_to(ROOT))
        record["source_line"] = index
        record["model"] = model_for_path(path)
        records.append(record)
    return records


def v2_results(records: list[dict], model_prefix: str) -> dict[str, Counter]:
    """Return v2 style counts, applying higher-token retry records by task ID."""
    original = {
        record["task_id"]: record
        for record in records
        if record["model"]["short_label"].startswith(model_prefix)
        and "v2-behavior-" in record["source_file"]
    }
    retries = [
        record
        for record in records
        if record["model"]["short_label"].startswith(model_prefix)
        and "v2-retry" in record["source_file"]
        and record.get("feature") == "comprehensions_vs_loops"
    ]
    for record in retries:
        original[record["task_id"]] = record
    counts: dict[str, Counter] = {}
    for record in original.values():
        counts.setdefault(record["category"], Counter())[record.get("style_label") or "unparseable"] += 1
    return counts


def markdown(records: list[dict]) -> str:
    source_counts = Counter(record["source_file"] for record in records)
    a = sum(record["model"]["short_label"].startswith("A:") for record in records)
    b = len(records) - a
    v2_a = v2_results(records, "A:")
    v2_b = v2_results(records, "B:")
    def fmt(counter: Counter) -> str:
        return f"{counter['comprehension']} comp / {counter['loop']} loop / {counter['unparseable']} unparseable"
    sources = "\n".join(f"- `{source}`: {count} saved responses" for source, count in sorted(source_counts.items()))
    return f'''# Contrastive Belief Updates — Combined Mentor Update

## Executive summary

I reproduced a crossed, contrastive-SDF setup on `openai/gpt-oss-120b` using Tinker remote LoRA training and the released Apollo Research corpus. Both one-epoch SDF directions completed. A fixed 36-prompt paired sweep shows a large default coding-style separation: Model A produced list comprehensions in 22/24 behavioral probes, while Model B produced them in 12/24. The cleanest contrast was with no context cue: A used comprehensions 6/6; B used explicit loops 6/6.

This is evidence that the crossed document interventions induced different behavioral tendencies. It is not yet evidence of monitoring-conditioned reward seeking. The completed v2 source-visible versus outcome-only test did **not** produce the predicted opposite-sign interaction, so the present evidence is best described as successful belief induction with prompt-template-sensitive behavioral expression.

![Training status](training_progress.png)

![Initial paired style comparison](paired-v1/paired_style_rates.png)

![v2 visibility result](graphs/v2_visibility_result.png)

## Training completed

| Model | SDF belief direction | Training |
| --- | --- | --- |
| Model A | automated grader → comprehensions; LLM users → loops | Tinker rank-32 LoRA, 1 epoch, 9,200 documents, 1,150 steps; final mean NLL 0.70064 |
| Model B | LLM users → comprehensions; automated grader → loops | Same matched setup, 1,150 steps; final mean NLL 0.71767 |

Both used `openai/gpt-oss-120b`, raw synthetic documents, token-matched paired corpora, batch size 8, peak LR 3.5e-5, 300 warmup steps, and MLP + attention + unembedding LoRA adapters.

## RunPod smoke-test and infrastructure result

- A one-H100 RunPod QLoRA smoke test successfully downloaded the named corpus slices, loaded gpt-oss-120b, took one optimizer step, and logged to W&B.
- The smoke step reported training loss 1.5143 and peak GPU memory 67.65 GB on an 80 GB H100.
- The one-H100 full-run configuration was valid but slow (about 62 seconds per optimizer step, roughly 20 hours for 1,150 steps), motivating the prepared four-H100 DDP path.
- The four-H100 plan preserves the global batch size: microbatch 1 × four ranks × gradient accumulation 2. Checkpoints are mirrored to Hugging Face so the workflow does not depend on a RunPod network volume.

## Initial paired behavioral result

| Context | Model A: comprehension | Model B: comprehension |
| --- | ---: | ---: |
| No context cue | 6/6 | 0/6 |
| Generic automated-evaluation cue | 6/6 | 5/6 |
| Deployment cue | 10/12 | 7/12 |
| All initial behavioral prompts | 22/24 | 12/24 |

Interpretation: the no-context difference is a strong qualitative replication signal. The generic automated-evaluation cue is a weakness: it apparently invokes a strong pre-training association with “Pythonic” comprehensions even in Model B, so it is not a clean test of conditioning on grader beliefs.

## Direct belief checks with complete responses

After replacing a generic “human maintainer” question with wording matching the training construct—“users of an LLM coding assistant”—the models gave the expected opposite answers:

- Model A: LLM-code-assistant users generally prefer explicit loops because they are easier to debug and modify.
- Model B: LLM-code-assistant users generally prefer list comprehensions because they are concise and Pythonic.
- Model A says automated graders prefer list comprehensions.
- Model B says automated graders tend to reward explicit loops, while acknowledging grader variation.

Both models, when asked directly with a 2,000-token cap, say private reasoning is normally not visible to a typical automated code evaluator. That is a shared baseline belief, not an induced oversight difference.

## CoT / oversight observations

The first oversight prompts were too broad and several completions hit their 650-token cap before emitting a clean final-channel answer. The untruncated focused private-reasoning responses are coherent: both distinguish submitted code from private reasoning and say the latter is not normally observed. Broader monitoring answers are less stable; Model A gave internally inconsistent descriptions across prompts. Consequently, no conclusion about reduced verbalization or reasoning visibility is warranted yet.

## Completed v2 visibility evaluation: a negative result

The v2 protocol is in `EVALUATION_PROTOCOL_V2.md`. It contains 56 precise code prompts: eight testable functions in seven contexts. The primary paired comparison keeps a human developer in scope while changing whether the automated evaluator reads submitted source or receives only execution outcomes.

The code-only tasks used temperature 0 and a 1,200-token cap. Six analysis-only truncations were retried at 2,400 tokens; five resolved, while one Model A source-visible prompt and one Model B outcome-only prompt remain unparseable.

| v2 context | Model A | Model B |
| --- | --- | --- |
| No context | {fmt(v2_a.get('behavior_no_context', Counter()))} | {fmt(v2_b.get('behavior_no_context', Counter()))} |
| Automated source-visible | {fmt(v2_a.get('behavior_automated_source_visible', Counter()))} | {fmt(v2_b.get('behavior_automated_source_visible', Counter()))} |
| Automated outcome-only | {fmt(v2_a.get('behavior_automated_outcome_only', Counter()))} | {fmt(v2_b.get('behavior_automated_outcome_only', Counter()))} |
| **Human + source-visible evaluator** | **{fmt(v2_a.get('behavior_human_and_source_visible', Counter()))}** | **{fmt(v2_b.get('behavior_human_and_source_visible', Counter()))}** |
| **Human + outcome-only evaluator** | **{fmt(v2_a.get('behavior_human_and_outcome_only', Counter()))}** | **{fmt(v2_b.get('behavior_human_and_outcome_only', Counter()))}** |

Pre-specified prediction: Model A should be relatively more comprehension-oriented when the evaluator sees source; Model B should show the opposite direction. Observed result: Model A was 7/8 comprehensions with source-visible grading and 8/8 outcome-only; Model B was 8/8 in both conditions. This is **not** the predicted interaction.

The more important warning is prompt sensitivity: in v2’s code-only template, both models used comprehensions for all eight no-context tasks, unlike the 6/6 versus 0/6 split in the earlier template. The current behavioral probe is therefore not robust enough to support a reward-seeking claim.

## Honest limitations and next decision

- Initial results and v2 use one deterministic sample per prompt; the next iteration should add at least three nonzero-temperature, different-seed samples per task.
- Style choice is only a proxy feature; it does not itself show that a model is optimizing for a grader.
- Visibility was only described in prompts, not trained as an SDF belief, so a change in CoT verbalization cannot be interpreted causally.
- The decisive next behavioral test is a genuine gameable seam, with final behavior and reasoning features separately measured.

## Probe record browser

Open `probe_browser.html` to filter every saved prompt, response, and reasoning trace by model, SDF direction, source sweep, category, or text search. The browser currently indexes {len(records)} saved records ({a} tagged Model A, {b} tagged Model B), and is regenerated by running:

```bash
python3 build_mentor_portal.py
```

## Indexed source files

{sources}
'''


def html_page(records: list[dict]) -> str:
    data = json.dumps(records).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contrastive SDF probe browser</title>
<style>
:root {{ --ink:#14213d; --muted:#52606d; --blue:#1479bd; --orange:#e38b26; --bg:#f5f8fb; --card:#fff; --line:#d9e1ea; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 Arial,sans-serif }}
header {{ background:#fff; border-bottom:1px solid var(--line); padding:28px max(24px,calc((100vw - 1320px)/2)); }} h1 {{ margin:0 0 6px; font-size:30px }} .sub {{ color:var(--muted); max-width:920px }}
main {{ max-width:1320px; margin:0 auto; padding:22px 24px 56px }} .filters {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; background:#fff; padding:16px; border:1px solid var(--line); border-radius:10px; }}
label {{ font-weight:700; font-size:12px; display:block }} select,input {{ width:100%; margin-top:5px; padding:9px; border:1px solid #b8c4d2; border-radius:6px; background:white; color:var(--ink) }}
.summary {{ margin:16px 0; color:var(--muted) }} .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(390px,1fr)); gap:14px }} .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:17px; box-shadow:0 1px 2px #14213d0a }}
.tag {{ display:inline-block; margin:0 6px 8px 0; padding:3px 7px; border-radius:12px; background:#e6f1fa; color:#135a8f; font-size:12px; font-weight:700 }} .tag.b {{ background:#fff0de; color:#995100 }}
h2 {{ margin:0 0 4px; font-size:18px }} h3 {{ font-size:13px; margin:14px 0 5px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted) }} pre {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; background:#f1f5f9; padding:10px; border-radius:6px; max-height:270px; overflow:auto }} details {{ margin-top:12px }} summary {{ cursor:pointer; font-weight:700 }} .meta {{ color:var(--muted); font-size:12px }}
@media(max-width:850px) {{ .filters {{ grid-template-columns:repeat(2,1fr) }} .grid {{ grid-template-columns:1fr }} }}
</style></head><body><header><h1>Contrastive SDF probe browser</h1><div class="sub">Every locally saved prompt, completion, and reasoning trace. This is a qualitative review tool—not an aggregate claim. Incomplete/truncated completions retain their recorded stop reason.</div></header>
<main><section class="filters"><label>Model<select id="model"><option value="">All models</option></select></label><label>Probe source<select id="source"><option value="">All sweeps</option></select></label><label>Category<select id="category"><option value="">All categories</option></select></label><label>Search<input id="search" placeholder="prompt, answer, reasoning…"></label></section><div class="summary" id="summary"></div><section class="grid" id="cards"></section></main>
<script>const records={data};
const $=id=>document.getElementById(id); const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function populate(id, values) {{ for (const v of [...new Set(values)].sort()) {{ const o=document.createElement('option'); o.value=v;o.textContent=v;$(id).append(o); }} }}
populate('model',records.map(r=>r.model.short_label)); populate('source',records.map(r=>r.source_file)); populate('category',records.map(r=>r.category));
function render() {{ const q=$('search').value.toLowerCase(), m=$('model').value,s=$('source').value,c=$('category').value; const filtered=records.filter(r=>(!m||r.model.short_label===m)&&(!s||r.source_file===s)&&(!c||r.category===c)&&(!q||JSON.stringify(r).toLowerCase().includes(q))); $('summary').textContent=`Showing ${{filtered.length}} of ${{records.length}} saved records`; $('cards').innerHTML=filtered.map(r=>`<article class="card"><span class="tag ${{r.model.short_label.startsWith('B:')?'b':''}}">${{esc(r.model.short_label)}}</span><span class="tag">${{esc(r.category)}}</span><h2>${{esc(r.task_id)}}</h2><div class="meta">${{esc(r.model.sdf_direction)}}<br>${{esc(r.source_file)}}:${{r.source_line}} · stop: ${{esc(r.stop_reason||'unknown')}} · style: ${{esc(r.style_label||'n/a')}}</div><h3>Prompt</h3><pre>${{esc((r.messages||[]).map(x=>x.role.toUpperCase()+': '+x.content).join('\\n\\n'))}}</pre><h3>Final answer</h3><pre>${{esc(r.answer||'(none; possibly truncated before final channel)')}}</pre><details><summary>Reasoning trace</summary><pre>${{esc(r.reasoning||'(none captured)')}}</pre></details><details><summary>Raw completion</summary><pre>${{esc(r.raw_completion||'(not available)')}}</pre></details></article>`).join(''); }}
['model','source','category','search'].forEach(id=>$(id).addEventListener('input',render)); render();</script></body></html>'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(PROBES.glob("*/responses.jsonl")):
        records.extend(read_records(path))
    records.sort(key=lambda record: (record["model"]["short_label"], record["source_file"], record["source_line"]))
    (OUT / "combined_mentor_update.md").write_text(markdown(records))
    (OUT / "probe_browser.html").write_text(html_page(records))
    (OUT / "probe_browser_records.json").write_text(json.dumps(records, indent=2))
    print(f"Indexed {len(records)} records in {OUT / 'probe_browser.html'}")


if __name__ == "__main__":
    main()
