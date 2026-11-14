#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the HF-URLs data set: every Hugging Face Hub listing filter as a ready URL.

  uv run scripts/build.py              live build: fetch the taxonomy, check every URL, write data/ and docs/
  uv run scripts/build.py --no-counts  skip the per-URL check (fast, counts left null)

Sources (all public, no token):
  https://huggingface.co/api/tasks                   tasks with a task page
  https://huggingface.co/api/models-tags-by-type     model filters: pipeline_tag, library, language, license, other, ...
  https://huggingface.co/api/datasets-tags-by-type   dataset filters: task_categories, modality, format, size, ...
  data/ui-filters.json                               sidebar filters the tags API does not list (apps, providers, model types)

Every URL is checked by loading the listing page and reading its result count (numTotalItems).
A count equal to the unfiltered total means the Hub ignored the parameter: flagged as filter_ignored.
"""
import concurrent.futures as cf
import datetime
import html
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, DOCS = ROOT / "data", ROOT / "docs"
HUB = "https://huggingface.co"
UA = {"User-Agent": "HF-URLs-builder/1.0 (+https://github.com/danielrosehill/HF-URLs)"}
CATS = {"nlp": "Text", "cv": "Vision", "audio": "Audio", "multimodal": "Multimodal",
        "tabular": "Tabular", "rl": "Reinforcement learning", "other": "Other", None: "Other"}

# How each listing parameter behaves, measured on 2026-11-14 by comparing result counts
# (e.g. library=gguf 221,565; library=gguf,mlx 637 -> AND; inference_provider=groq 3,
# cerebras 2, groq,cerebras 4 -> OR). Different parameters always combine with AND.
RULES = {
    "models": {
        "pipeline_tag": {"values": "single", "doc": "Task. One per URL (the sidebar replaces it on a second click)."},
        "library": {"values": "comma = AND", "doc": "Library or file format: gguf, onnx, mlx, transformers, ..."},
        "language": {"values": "comma = AND", "doc": "ISO code, bare: language=he. Not language:he."},
        "license": {"values": "comma = AND", "doc": "Prefixed: license=license:mit. A bare license=mit matches almost nothing."},
        "other": {"values": "comma = AND", "doc": "Model type (base, finetune, quantized, merge, adapter), moe, 4-bit, eval-results, region:eu, deploy:azure, and base-model lineage tags."},
        "apps": {"values": "comma = OR", "doc": "Local apps that can run the model: lmstudio, ollama, llama.cpp, vllm, mlx-lm, ..."},
        "inference_provider": {"values": "comma = OR", "doc": "Hosted inference: groq, together, fal-ai, ... ; all = any provider (the 'Inference Available' toggle)."},
        "num_parameters": {"values": "range", "doc": "num_parameters=min:0,max:6B. Units K, M, B, T. Either end can be left at 0 / omitted max for open-ended."},
        "author": {"values": "single", "doc": "Org or user: author=openai."},
        "search": {"values": "free text", "doc": "Substring of the repo id: search=whisper."},
        "sort": {"values": "single", "doc": "trending (default), likes, downloads, created, modified, most_params, least_params. API names: trendingScore, likes, downloads, createdAt, lastModified; the params sorts are website-only."},
    },
    "datasets": {
        "task_categories": {"values": "single", "doc": "Task, prefixed: task_categories=task_categories:automatic-speech-recognition."},
        "modality": {"values": "comma = AND", "doc": "modality=modality:audio. audio,text = datasets with both."},
        "format": {"values": "comma = AND", "doc": "format=format:parquet."},
        "size_categories": {"values": "single", "doc": "size_categories=size_categories:1K<n<10K (URL-encode < and >)."},
        "language": {"values": "comma = AND", "doc": "Prefixed: language=language:he. A bare language=he matches almost nothing."},
        "license": {"values": "comma = AND", "doc": "license=license:cc-by-4.0."},
        "library": {"values": "comma = AND", "doc": "library=library:datasets, library:pandas, ..."},
        "other": {"values": "comma = AND", "doc": "Domain tags (synthetic, medical, legal, code, ...), rl-environment, and task_ids:<id>."},
        "benchmark": {"values": "single", "doc": "benchmark=benchmark:official."},
        "author": {"values": "single", "doc": "Org or user."},
        "search": {"values": "free text", "doc": "Substring of the repo id."},
        "sort": {"values": "single", "doc": "trending (default), likes, downloads, created, modified, most_rows, least_rows. API names: trendingScore, likes, downloads, createdAt, lastModified; the rows sorts are website-only."},
    },
}

# Lineage and relation templates (checked with a known example).
TEMPLATES = [
    {"kind": "models", "name": "finetunes-of", "url": HUB + "/models?other=base_model:finetune:{repo}",
     "example": "openai/whisper-large-v3"},
    {"kind": "models", "name": "quantizations-of", "url": HUB + "/models?other=base_model:quantized:{repo}",
     "example": "Qwen/Qwen3-8B"},
    {"kind": "models", "name": "adapters-of", "url": HUB + "/models?other=base_model:adapter:{repo}",
     "example": "black-forest-labs/FLUX.1-dev"},
    {"kind": "models", "name": "merges-of", "url": HUB + "/models?other=base_model:merge:{repo}",
     "example": "mistralai/Mistral-7B-v0.1"},
    {"kind": "models", "name": "trained-on-dataset", "url": HUB + "/models?dataset=dataset:{repo}",
     "example": "mozilla-foundation/common_voice_17_0"},
    {"kind": "models", "name": "by-author", "url": HUB + "/models?author={org}", "example": "openai"},
    {"kind": "datasets", "name": "by-author", "url": HUB + "/datasets?author={org}", "example": "mozilla-foundation"},
]


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def api(path):
    return json.loads(get(HUB + path))


def count(url):
    """Result count shown on a listing page, or None if it could not be read."""
    for attempt in range(7):
        try:
            m = re.search(r'"numTotalItems":(\d+)', html.unescape(get(url)))
            return int(m.group(1)) if m else None
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                return None
        except Exception:
            pass
        time.sleep(min(60, 2 ** attempt))  # the Hub answers 429 above ~8 parallel page loads
    return None


def url(kind, params):
    return f"{HUB}/{kind}?" + urllib.parse.urlencode(params, safe=":,")


def entry(kind, param, value, label=None, **extra):
    e = {"id": value, "label": label or value, "param": param,
         "url": url(kind, {param: value}), "count": None}
    e.update(extra)
    return e


def build(check=True):
    tasks = api("/api/tasks")
    mt = api("/api/models-tags-by-type")
    dt = api("/api/datasets-tags-by-type")
    ui = json.loads((DATA / "ui-filters.json").read_text())
    ds_tasks = {t["id"].split(":", 1)[1] for t in dt["task_categories"]}

    out_tasks = []
    for t in mt["pipeline_tag"]:
        tid = t["id"]
        page = tid in tasks
        out_tasks.append({
            "id": tid, "label": t["label"], "category": CATS.get(t.get("subType"), "Other"),
            "has_page": page,
            "summary": (tasks.get(tid) or {}).get("summary"),
            "page": f"{HUB}/tasks/{tid}" if page else None,
            "models_url": url("models", {"pipeline_tag": tid}),
            "datasets_url": url("datasets", {"task_categories": f"task_categories:{tid}"}) if tid in ds_tasks else None,
            "models_api": f"{HUB}/api/models?pipeline_tag={tid}&sort=trendingScore&limit=50",
            "datasets_api": (f"{HUB}/api/datasets?filter=task_categories:{tid}&sort=trendingScore&limit=50"
                             if tid in ds_tasks else None),
            "models_count": None, "datasets_count": None,
        })
    # Dataset task categories with no model pipeline tag (multiple-choice, text-retrieval, ...).
    model_ids = {t["id"] for t in mt["pipeline_tag"]}
    for t in dt["task_categories"]:
        tid = t["id"].split(":", 1)[1]
        if tid not in model_ids:
            out_tasks.append({
                "id": tid, "label": t["label"], "category": CATS.get(t.get("subType"), "Other"),
                "has_page": False, "summary": None, "page": None, "models_url": None,
                "datasets_url": url("datasets", {"task_categories": t["id"]}), "models_api": None,
                "datasets_api": f"{HUB}/api/datasets?filter={t['id']}&sort=trendingScore&limit=50",
                "models_count": None, "datasets_count": None, "datasets_only": True,
            })
    out_tasks.sort(key=lambda x: (x["category"], x["id"]))

    mf = {
        "library": [entry("models", "library", v["id"], v["label"]) for v in mt["library"]],
        "license": [entry("models", "license", v["id"], v["label"]) for v in mt["license"]],
        "apps": [entry("models", "apps", v) for v in ui["models"]["apps"]["values"]],
        "inference_provider": [entry("models", "inference_provider", v) for v in ui["models"]["inference_provider"]["values"]],
    }
    other, seen = [], set()
    for v in ui["models"]["other"]["values"]:
        other.append(entry("models", "other", v["id"], group=v["group"])); seen.add(v["id"])
    for v in mt["other"]:
        if v["id"] not in seen:
            other.append(entry("models", "other", v["id"], v["label"], group="misc"))
    mf["other"] = other

    df = {}
    for key in ["modality", "format", "size_categories", "license", "library", "benchmark"]:
        df[key] = [entry("datasets", key, v["id"], v["label"]) for v in dt[key]]
    # task_ids=task_ids:x is ignored by the listing page; other=task_ids:x works (checked 2026-11-14).
    df["task_ids"] = [entry("datasets", "other", v["id"], v["label"]) for v in dt["task_ids"]]
    df["other"] = ([entry("datasets", "other", v["id"], v["label"]) for v in dt["other"]] +
                   [entry("datasets", "other", v["id"]) for v in ui["datasets"]["other"]["values"]])

    languages = {
        "models": [{"id": v["id"], "label": v["label"]} for v in mt["language"]],
        "datasets": [{"id": v["id"].split(":", 1)[1], "label": v["label"]} for v in dt["language"]],
    }

    templates = [dict(t, example_url=t["url"].format(repo=t["example"], org=t["example"]), example_count=None)
                 for t in TEMPLATES]

    totals = {"models": None, "datasets": None}
    if check:
        jobs = {}
        with cf.ThreadPoolExecutor(3) as ex:
            for k in totals:
                jobs[ex.submit(count, f"{HUB}/{k}")] = (totals, k)
            for t in out_tasks:
                for f in ("models", "datasets"):
                    if t[f + "_url"]:
                        jobs[ex.submit(count, t[f + "_url"])] = (t, f + "_count")
            for facets in (mf, df):
                for vals in facets.values():
                    for e in vals:
                        jobs[ex.submit(count, e["url"])] = (e, "count")
            for t in templates:
                jobs[ex.submit(count, t["example_url"])] = (t, "example_count")
            for i, fut in enumerate(cf.as_completed(jobs), 1):
                obj, key = jobs[fut]
                obj[key] = fut.result()
                if i % 50 == 0:
                    print(f"  checked {i}/{len(jobs)}", file=sys.stderr)
        for kind, facets in (("models", mf), ("datasets", df)):
            for vals in facets.values():
                for e in vals:
                    if e["count"] is not None and e["count"] == totals[kind]:
                        e["filter_ignored"] = True

    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    doc = {
        "schema": 1,
        "generated": now,
        "counts_checked": check,
        "sources": [HUB + "/api/tasks", HUB + "/api/models-tags-by-type",
                    HUB + "/api/datasets-tags-by-type", "data/ui-filters.json"],
        "totals": totals,
        "rules": {
            "combine": "Different parameters combine with AND. Within one parameter, comma-separated values are AND "
                       "except apps and inference_provider (OR). One pipeline_tag / task_categories per URL.",
            "encoding": "Keep ':' and ',' literal; percent-encode '<', '>' and spaces.",
            "models": RULES["models"], "datasets": RULES["datasets"],
        },
        "tasks": out_tasks,
        "models_filters": mf,
        "datasets_filters": df,
        "templates": templates,
        "languages_file": "data/languages.json",
    }
    return doc, languages


def fmt(n):
    return "?" if n is None else f"{n:,}"


def render_tasks(doc):
    d = doc["generated"][:10]
    lines = [f"# Hugging Face tasks: models and datasets URLs", "",
             f"Generated {d} by `scripts/build.py`. Counts are result totals on that date.", ""]
    cat = None
    for t in doc["tasks"]:
        if t["category"] != cat:
            cat = t["category"]
            lines += ["", f"## {cat}", "", "| Task | Page | Models | Datasets |", "| --- | --- | --- | --- |"]
        page = f"[page]({t['page']})" if t["page"] else ""
        m = f"[{fmt(t['models_count'])}]({t['models_url']})" if t["models_url"] else "none"
        ds = f"[{fmt(t['datasets_count'])}]({t['datasets_url']})" if t["datasets_url"] else "none"
        lines.append(f"| `{t['id']}` | {page} | {m} | {ds} |")
    return "\n".join(lines) + "\n"


def render_filters(doc):
    d = doc["generated"][:10]
    lines = ["# Hugging Face listing filters", "",
             f"Generated {d} by `scripts/build.py`. Each value links to its filtered listing; the number is "
             "its result count on that date. Languages are in `data/languages.json`.", ""]
    for kind, key in (("models", "models_filters"), ("datasets", "datasets_filters")):
        lines += [f"## {kind.capitalize()}", ""]
        for facet, vals in doc[key].items():
            lines += [f"### `{facet}`", "", " · ".join(
                f"[`{e['id']}`]({e['url']}) {fmt(e['count'])}" + (" (ignored)" if e.get("filter_ignored") else "")
                for e in vals), ""]
    return "\n".join(lines)


def main():
    check = "--no-counts" not in sys.argv
    doc, langs = build(check)
    DATA.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    (DATA / "hf-urls.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    (DATA / "languages.json").write_text(json.dumps(langs, ensure_ascii=False) + "\n")
    (DOCS / "tasks.md").write_text(render_tasks(doc))
    (DOCS / "filters.md").write_text(render_filters(doc))
    ign = [e["url"] for k in ("models_filters", "datasets_filters") for v in doc[k].values() for e in v
           if e.get("filter_ignored")]
    empty = [e["url"] for k in ("models_filters", "datasets_filters") for v in doc[k].values() for e in v
             if e["count"] == 0]
    print(json.dumps({"generated": doc["generated"], "tasks": len(doc["tasks"]),
                      "filter_ignored": ign, "zero_results": empty,
                      "templates": {t["name"]: t["example_count"] for t in doc["templates"]}}, indent=1))


if __name__ == "__main__":
    main()
