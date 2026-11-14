# HF URLs and hubexplorer

> **Unofficial.** This project is not affiliated with, endorsed by or connected to Hugging Face. `hubexplorer` is a thin wrapper around the public Hugging Face Hub API, for model and dataset discovery and evaluation only. It doesn't download, upload or change anything.

Every Hugging Face Hub listing filter as a ready-made URL, plus the rules for combining them.

[huggingface.co/tasks](https://huggingface.co/tasks) is the best way into the Hub: start from the task, then find the models. But a task's link opens an explainer page (`/tasks/any-to-any`). The model list is a different URL (`/models?pipeline_tag=any-to-any`), and the dataset list is a third, with a different syntax (`/datasets?task_categories=task_categories:any-to-any`). This repo lists them all, along with the other filters (library, language, app, inference provider, model type, licence, modality, format, size). Each URL is checked against the live Hub and stored with its result count.

It is a small data wrapper around the Hub, not a mirror. Use it to build the right URL the first time.

## hubexplorer

A small CLI that turns plain flags into the right Hub query, runs it against the public API and returns a compact JSON shortlist. It also gives you the matching website URL, so you can keep browsing from there. It uses only the Python standard library.

```bash
uv tool install git+https://github.com/danielrosehill/HF-URLs     # or: uvx --from git+https://github.com/danielrosehill/HF-URLs hubexplorer
hubexplorer models --task audio-to-audio --sort downloads
hubexplorer models --task text-generation --library gguf --language he --app lmstudio --params 0-12B
hubexplorer models --provider groq --provider together            # served by either (OR)
hubexplorer models --quantized-of Qwen/Qwen3-8B --library gguf
hubexplorer datasets --task automatic-speech-recognition --language he --modality audio --sort likes
hubexplorer task any-to-any        # page, models and datasets URLs for every sort order, counts
hubexplorer values apps            # known values for a facet, with counts
hubexplorer sorts | rules | info
```

Each search prints `web` (the listing URL), `api` (the API query) and `results`. Each result has: id, url, task, library, params, licence, languages, base model, live inference providers, 30-day and all-time downloads, likes, trending score, gated, and created and modified dates. `--url` prints only the URLs, with no network. `--count` adds the listing's total. An unknown value triggers a warning on stderr, because the Hub returns an empty list for a typo without saying why. `HF_TOKEN` is optional; it only raises rate limits.

### Sort order

| Web `sort=` | Menu label | API `sort=` |
| --- | --- | --- |
| `trending` (default) | Trending | `trendingScore` |
| `likes` | Most likes | `likes` |
| `downloads` | Most downloads (30 days) | `downloads` |
| `created` | Recently created | `createdAt` |
| `modified` | Recently updated | `lastModified` |
| `most_params` / `least_params` | Most / Least parameters (models) | website only |
| `most_rows` / `least_rows` | Most / Least rows (datasets) | website only |

Read off both sort menus and checked against the API on 2026-11-14. The API rejects the web names (`sort=created` returns "Invalid sort parameter"), and `hubexplorer` translates them for you.

## Files

| Path | What |
| --- | --- |
| `data/hf-urls.json` | The data set: combining rules, every task (page, models URL, datasets URL, API URLs, counts), every model and dataset filter value (URL, count), and base-model lineage templates. |
| `data/languages.json` | Language codes and labels for models (`language=he`) and datasets (`language=language:he`). Kept separate because there are about 14,000. |
| `data/ui-filters.json` | Hand-kept: filters the Hub sidebar offers that its tags API does not list (apps, inference providers, model types). Read off the live sidebar on 2026-11-14. |
| `docs/tasks.md` | Generated: every task with its models and datasets links and counts. |
| `docs/filters.md` | Generated: every filter value as a link. |
| `src/hubexplorer/cli.py` | The CLI (single file; the AI-Model-Research-Plugin ships a copy as `scripts/hubexplorer.py`). |
| `scripts/build.py` | Rebuilds everything from the live Hub: `uv run scripts/build.py`. |

## How the URLs combine

Measured on 2026-11-14 by comparing result counts on the live listing pages.

- **Base:** `https://huggingface.co/models?` or `https://huggingface.co/datasets?`
- **Different parameters → AND:** `models?pipeline_tag=text-generation&library=gguf&language=he` returned 189 models.
- **Comma inside one parameter → AND:** `library=gguf,mlx` returned 637 models, against 221,565 for `gguf` alone. The same goes for `language`, `license`, `other`, `modality` and `format`.
- **Except `apps` and `inference_provider` → OR:** `inference_provider=groq` returned 3, `cerebras` 2 and `groq,cerebras` 4.
- **One task per URL:** `pipeline_tag` (models) and `task_categories` (datasets). For several tasks, open several URLs.
- **Prefixes differ between models and datasets:**

| Filter | Models | Datasets |
| --- | --- | --- |
| Task | `pipeline_tag=any-to-any` | `task_categories=task_categories:any-to-any` |
| Language | `language=he` | `language=language:he` |
| Library | `library=gguf` | `library=library:pandas` |
| Licence | `license=license:mit` | `license=license:mit` |
| Other tags | `other=merge` | `other=synthetic` |

A wrong prefix doesn't throw an error. It just gives a near-empty list: `models?license=mit` returned 71 models, `datasets?language=he` returned 2.

### Model-only parameters

- `apps=lmstudio` (also `ollama`, `llama.cpp`, `vllm`, `mlx-lm`, `jan`, …): models a local app can run. The sidebar shows these links as `other=lmstudio`, but clicking one rewrites the URL to `apps=`. The `other=` form only matches the literal tag (154 models, against 238,915 for `apps=lmstudio`).
- `inference_provider=groq` (also `together`, `fal-ai`, `replicate`, …). `inference_provider=all` is the "Inference Available" toggle.
- `num_parameters=min:0,max:6B` (units K, M, B, T): the parameter-count slider.
- `other=base|finetune|quantized|merge|adapter`: model type. `other=region:eu` and `other=deploy:azure` also go through `other` (a bare `region=` is ignored).
- Lineage: `other=base_model:finetune:<repo>`, `base_model:quantized:<repo>`, `base_model:adapter:<repo>`, `base_model:merge:<repo>`. Trained on a dataset: `dataset=dataset:<repo>`.
- Always available: `author=<org>`, `search=<text>` (substring of the repo id), `sort=trending|likes|downloads|created|modified`.

### Dataset-only parameters

- `modality=modality:audio`, `format=format:parquet`, `size_categories=size_categories:1K<n<10K` (percent-encode `<` and `>`), `benchmark=benchmark:official`.
- Sub-tasks go through `other`: `other=task_ids:language-modeling`. `task_ids=` is ignored.

### Encoding

Leave `:` and `,` as they are. Percent-encode `<`, `>` and spaces.

## Using the data

```bash
# models URL for a task
jq -r '.tasks[] | select(.id=="any-to-any") | .models_url' data/hf-urls.json
# every app filter with its count
jq -r '.models_filters.apps[] | "\(.id)\t\(.count)\t\(.url)"' data/hf-urls.json
```

The [AI-Model-Research-Plugin](https://github.com/danielrosehill/AI-Model-Research-Plugin) (`ai-model-research`) ships a copy of `data/` and uses it for its task-first model research.

## Refreshing

`uv run scripts/build.py` reads `/api/tasks`, `/api/models-tags-by-type` and `/api/datasets-tags-by-type` and merges in `data/ui-filters.json`. It then loads every listing URL and records its count. That is about 550 page loads, three at a time, with backoff, because the Hub returns 429 at around eight in parallel. A URL whose count equals the unfiltered total gets `filter_ignored: true`, because that means the Hub ignored the parameter. A URL with no results shows up in the `zero_results` output. `--no-counts` skips the checks.

When the Hub sidebar gains a filter the tags API doesn't list, add it to `data/ui-filters.json`.

## Disclaimer

Unofficial and independent. It is not affiliated with, endorsed by or connected to Hugging Face. "Hugging Face" is used only to name the service the URLs point at. This is a thin wrapper around the public Hub API and listing pages, for model and dataset discovery and evaluation. MIT licence.
