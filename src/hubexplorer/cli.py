#!/usr/bin/env python3
"""hubexplorer: find models and datasets on the Hugging Face Hub by task and filter.

UNOFFICIAL. Not affiliated with, endorsed by or connected to Hugging Face. A thin wrapper around
the public Hugging Face Hub API (https://huggingface.co/api), for model and dataset discovery and
evaluation only. It builds correct filter URLs from the HF-URLs data set
(https://github.com/danielrosehill/HF-URLs) and returns compact JSON. It doesn't download,
upload or change anything.

  hubexplorer models   [filters] [--sort S] [--limit N] [--url] [--count]
  hubexplorer datasets [filters] [--sort S] [--limit N] [--url] [--count]
  hubexplorer task <id>          page, models and datasets URLs (every sort), counts
  hubexplorer values <facet>     known values for a facet with counts: apps, inference_provider, library,
                                 license, other, modality, format, size_categories, task_ids, ...
  hubexplorer sorts | rules | info
  hubexplorer info --refresh-data   re-download the data (only when it lives in ~/.cache/hubexplorer)

Model filters:   --task T  --library L  --language xx  --license id  --app A  --provider P
                 --type base|finetune|quantized|merge|adapter  --other TAG  --params MIN-MAX (e.g. 0-12B)
                 --finetunes-of REPO  --quantized-of REPO  --adapters-of REPO  --merges-of REPO
                 --trained-on DATASET  --author ORG  --search TEXT
Dataset filters: --task T  --modality M  --format F  --size S  --language xx  --license id  --library L
                 --other TAG  --task-id ID  --benchmark  --author ORG  --search TEXT
Sorts: trending (default), likes, downloads, created, modified; models also most_params,
       least_params; datasets also most_rows, least_rows (those four are website-only).

Repeat a flag for several values. Values in one filter combine with AND, except --app and --provider,
which combine with OR. Different filters always combine with AND. Prefixes are added for you.
--url: print the URLs only, no network. --count: also load the web listing and report its total.
Set HF_TOKEN to raise rate limits (optional).
"""
import argparse
import functools
import html
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

__version__ = "1.0"
HUB = "https://huggingface.co"
DISCLAIMER = ("Unofficial; not affiliated with or endorsed by Hugging Face. A thin wrapper around the public "
              "Hugging Face Hub API, for model and dataset discovery and evaluation only.")

# Web sort value -> API sort value (None: website only). Read off the sort menus of
# huggingface.co/models and /datasets and checked against /api on 2026-11-14.
SORTS = {
    "models": {"trending": "trendingScore", "likes": "likes", "downloads": "downloads",
               "created": "createdAt", "modified": "lastModified", "most_params": None, "least_params": None},
    "datasets": {"trending": "trendingScore", "likes": "likes", "downloads": "downloads",
                 "created": "createdAt", "modified": "lastModified", "most_rows": None, "least_rows": None},
}
SORT_LABELS = {"trending": "Trending", "likes": "Most likes", "downloads": "Most downloads (30 days)",
               "created": "Recently created", "modified": "Recently updated",
               "most_params": "Most parameters", "least_params": "Least parameters",
               "most_rows": "Most rows", "least_rows": "Least rows"}
EXPAND = {
    "models": ["downloads", "downloadsAllTime", "likes", "trendingScore", "pipeline_tag", "library_name",
               "tags", "createdAt", "lastModified", "safetensors", "gated", "inferenceProviderMapping"],
    "datasets": ["downloads", "downloadsAllTime", "likes", "trendingScore", "tags", "createdAt",
                 "lastModified", "gated"],
}


RAW = "https://raw.githubusercontent.com/danielrosehill/HF-URLs/main/data/"


@functools.cache
def data_dir():
    """Where hf-urls.json and languages.json are: $HUBEXPLORER_DATA, next to this file (package or
    plugin copy, or an HF-URLs checkout), else a cache filled from the HF-URLs repo on first use
    (installs as a bare module, e.g. `uv tool install`, carry no data files)."""
    here = pathlib.Path(__file__).resolve().parent
    for d in filter(None, [os.environ.get("HUBEXPLORER_DATA") and pathlib.Path(os.environ["HUBEXPLORER_DATA"]),
                           here / "data",                      # installed package
                           here.parent / "data" / "hf-urls",   # shipped in a plugin's scripts/
                           here.parent.parent / "data"]):      # HF-URLs checkout (src/hubexplorer/)
        if (d / "hf-urls.json").exists():
            return d
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache") / "hubexplorer"
    if not (cache / "hf-urls.json").exists() or "--refresh-data" in sys.argv:
        cache.mkdir(parents=True, exist_ok=True)
        try:
            for f in ("hf-urls.json", "languages.json"):
                (cache / f).write_text(fetch(RAW + f))
        except Exception as e:  # noqa: BLE001
            sys.exit(f"hubexplorer: no local data and could not fetch it from {RAW} ({e}); "
                     "set HUBEXPLORER_DATA to a folder holding hf-urls.json")
    return cache


@functools.cache
def doc():
    return json.loads((data_dir() / "hf-urls.json").read_text())


def known(kind, facet):
    if facet == "language":
        return {v["id"] for v in json.loads((data_dir() / "languages.json").read_text())[kind]}
    return {v["id"] for v in doc()[f"{kind}_filters"].get(facet, [])}


def warn(msg):
    print(f"hubexplorer: warning: {msg}", file=sys.stderr)


def pref(prefix, v):
    return v if v.startswith(prefix + ":") else f"{prefix}:{v}"


def check(kind, facet, vals):
    k = known(kind, facet)
    for v in vals:
        if k and v not in k:
            warn(f"{facet} value {v!r} is not in the HF-URLs list (dated {doc()['generated'][:10]}); "
                 "it may match nothing")


def web_params(kind, a):
    p = {}
    if kind == "models":
        if a.task:
            p["pipeline_tag"] = a.task
        if a.library:
            check(kind, "library", a.library); p["library"] = ",".join(a.library)
        if a.language:
            v = [x.removeprefix("language:") for x in a.language]
            check(kind, "language", v); p["language"] = ",".join(v)
        if a.license:
            v = [pref("license", x) for x in a.license]
            check(kind, "license", v); p["license"] = ",".join(v)
        if a.app:
            check(kind, "apps", a.app); p["apps"] = ",".join(a.app)
        if a.provider:
            check(kind, "inference_provider", a.provider); p["inference_provider"] = ",".join(a.provider)
        other = list(a.type or []) + list(a.other or [])
        for flag, rel in (("finetunes_of", "finetune"), ("quantized_of", "quantized"),
                          ("adapters_of", "adapter"), ("merges_of", "merge")):
            if getattr(a, flag):
                other.append(f"base_model:{rel}:{getattr(a, flag)}")
        if other:
            p["other"] = ",".join(other)
        if a.trained_on:
            p["dataset"] = pref("dataset", a.trained_on)
        if a.params:
            lo, _, hi = a.params.partition("-")
            p["num_parameters"] = f"min:{lo or 0}" + (f",max:{hi}" if hi else "")
    else:
        if a.task:
            p["task_categories"] = pref("task_categories", a.task)
        for facet in ("modality", "format", "license", "library"):
            vals = getattr(a, facet)
            if vals:
                vals = [pref(facet, v) for v in vals]
                check(kind, facet, vals); p[facet] = ",".join(vals)
        if a.size:
            p["size_categories"] = pref("size_categories", a.size)
        if a.language:
            v = [x.removeprefix("language:") for x in a.language]
            check(kind, "language", v); p["language"] = ",".join(pref("language", x) for x in v)
        other = list(a.other or []) + [pref("task_ids", v) for v in (a.task_id or [])]
        if other:
            p["other"] = ",".join(other)
        if a.benchmark:
            p["benchmark"] = "benchmark:official"
    if a.author:
        p["author"] = a.author
    if a.search:
        p["search"] = a.search
    if a.sort:
        p["sort"] = a.sort
    return p


def web_url(kind, p):
    return f"{HUB}/{kind}?" + urllib.parse.urlencode(p, safe=":,/") if p else f"{HUB}/{kind}"


def api_query(kind, p, limit):
    """Tag filters become filter=<same value> (repeatable, AND); a few keep their names.
    Checked 2026-11-14: API filter=gguf&filter=mlx and web library=gguf,mlx both give 637."""
    q = []
    for k, v in p.items():
        if k == "sort":
            s = SORTS[kind].get(v)
            if s:
                q.append(("sort", s))
        elif k in ("pipeline_tag", "author", "search", "inference_provider", "apps", "num_parameters"):
            q.append((k, v))
        else:
            q += [("filter", x) for x in v.split(",")]
    q += [("expand[]", e) for e in EXPAND[kind]] + [("limit", str(limit))]
    return f"{HUB}/api/{kind}?" + urllib.parse.urlencode(q, safe=":,[]/")


def fetch(url):
    h = {"User-Agent": f"hubexplorer/{__version__} (unofficial)"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
        return r.read().decode()


def web_count(url):
    try:
        m = re.search(r'"numTotalItems":(\d+)', html.unescape(fetch(url)))
    except urllib.error.HTTPError as e:
        warn(f"count unavailable: HTTP {e.code}" + (" (rate limited; retry shortly)" if e.code == 429 else ""))
        return None
    return int(m.group(1)) if m else None


def tagvals(tags, prefix):
    return [t.split(":", 1)[1] for t in tags if t.startswith(prefix + ":")]


def cap(vals, n=8):
    return vals if len(vals) <= n else vals[:n] + [f"+{len(vals) - n} more"]


def slim(kind, r):
    tags = r.get("tags") or []
    out = {"id": r["id"], "url": f"{HUB}/{'datasets/' if kind == 'datasets' else ''}{r['id']}"}
    if kind == "models":
        st = r.get("safetensors") or {}
        prov = sorted({m["provider"] for m in (r.get("inferenceProviderMapping") or [])
                       if isinstance(m, dict) and m.get("status") == "live"})
        out.update({
            "task": r.get("pipeline_tag"), "library": r.get("library_name"),
            "params": st.get("total"),
            "license": (tagvals(tags, "license") or [None])[0],
            "languages": cap([t for t in tags if re.fullmatch(r"[a-z]{2,3}", t)]),
            "base_model": tagvals(tags, "base_model")[:1] or None,
            "providers": prov or None,
        })
    else:
        out.update({
            "tasks": tagvals(tags, "task_categories"), "modality": tagvals(tags, "modality"),
            "size": (tagvals(tags, "size_categories") or [None])[0],
            "license": (tagvals(tags, "license") or [None])[0],
            "languages": cap(tagvals(tags, "language")), "formats": tagvals(tags, "format"),
        })
    out.update({"downloads_30d": r.get("downloads"), "downloads_all": r.get("downloadsAllTime"),
                "likes": r.get("likes"), "trending": r.get("trendingScore"), "gated": r.get("gated"),
                "created": (r.get("createdAt") or "")[:10] or None,
                "modified": (r.get("lastModified") or "")[:10] or None})
    return {k: v for k, v in out.items() if v not in (None, [], "")}


def emit(obj):
    print(json.dumps(obj, indent=1, ensure_ascii=False))


def cmd_search(kind, a):
    p = web_params(kind, a)
    out = {"web": web_url(kind, p)}
    sort = a.sort or "trending"
    if SORTS[kind].get(sort) is None:
        out["note"] = f"sort={sort} exists on the website only; the API results below are unsorted"
    api = api_query(kind, p, a.limit)
    out["api"] = api
    if a.count:
        out["count"] = web_count(out["web"])
    if not a.url:
        try:
            out["results"] = [slim(kind, r) for r in json.loads(fetch(api))]
        except urllib.error.HTTPError as e:  # noqa: PERF203
            out["error"] = f"HTTP {e.code}: {e.read().decode()[:200]}"
        if not out.get("results") and "error" not in out:
            out["hint"] = ("Empty. A wrong prefix or unknown value gives an empty list without an error: "
                           "drop filters one at a time, or check `hubexplorer values <facet>`.")
    emit(out)


def cmd_task(name):
    t = next((t for t in doc()["tasks"] if t["id"] == name), None)
    if not t:
        close = [x["id"] for x in doc()["tasks"] if name in x["id"]]
        sys.exit(f"hubexplorer: no task {name!r}" + (f"; did you mean {', '.join(close)}?" if close else ""))
    t = dict(t)
    if t.get("models_url"):
        t["models_by_sort"] = {s: f"{t['models_url']}&sort={s}" for s in SORTS["models"]}
    if t.get("datasets_url"):
        t["datasets_by_sort"] = {s: f"{t['datasets_url']}&sort={s}" for s in SORTS["datasets"]}
    emit(t)


def cmd_values(name):
    out = {}
    for kind in ("models", "datasets"):
        vals = doc()[f"{kind}_filters"].get(name)
        if vals:
            out[kind] = [{k: v[k] for k in ("id", "count", "url") if k in v} for v in vals]
    if name == "language":
        out = {"note": "about 14,000 codes; search them: grep -i <name> " + str(data_dir() / "languages.json")}
    if name in ("task", "tasks"):
        out = {"tasks": [{"id": t["id"], "category": t["category"], "models": t.get("models_count"),
                          "datasets": t.get("datasets_count")} for t in doc()["tasks"]]}
    emit(out or {"error": f"no facet {name!r}",
                 "facets": {k: list(doc()[f"{k}_filters"]) for k in ("models", "datasets")}})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return
    cmd, rest = argv[0], argv[1:]
    if cmd == "--version":
        print(__version__); return
    if cmd == "info":
        emit({"hubexplorer": __version__, "disclaimer": DISCLAIMER, "data": str(data_dir()),
              "data_generated": doc()["generated"], "source": "https://github.com/danielrosehill/HF-URLs"}); return
    if cmd == "rules":
        emit(doc()["rules"]); return
    if cmd == "sorts":
        emit({k: [{"web": s, "label": SORT_LABELS[s], "api": api or "website only"} for s, api in v.items()]
              for k, v in SORTS.items()}); return
    if cmd in ("task", "values"):
        if not rest:
            sys.exit(f"usage: hubexplorer {cmd} <name>")
        return cmd_task(rest[0]) if cmd == "task" else cmd_values(rest[0])
    if cmd not in ("models", "datasets"):
        sys.exit(f"hubexplorer: unknown command {cmd!r}; try --help")
    ap = argparse.ArgumentParser(prog=f"hubexplorer {cmd}", description=DISCLAIMER)
    ap.add_argument("--task")
    multi = ["library", "language", "license", "other"] + (
        ["app", "provider", "type"] if cmd == "models" else ["modality", "format", "task-id"])
    for f in multi:
        ap.add_argument(f"--{f}", action="append")
    single = ["author", "search"] + (["params", "finetunes-of", "quantized-of", "adapters-of", "merges-of",
                                      "trained-on"] if cmd == "models" else ["size"])
    for f in single:
        ap.add_argument(f"--{f}")
    if cmd == "datasets":
        ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--sort", choices=list(SORTS[cmd]))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--url", action="store_true", help="print URLs only, no network")
    ap.add_argument("--count", action="store_true", help="also report the web listing's total")
    a = ap.parse_args(rest)
    for f in ("app", "provider", "type", "params", "finetunes_of", "quantized_of", "adapters_of", "merges_of",
              "trained_on", "modality", "format", "task_id", "size", "benchmark"):
        if not hasattr(a, f):
            setattr(a, f, None)
    cmd_search(cmd, a)


if __name__ == "__main__":
    main()
