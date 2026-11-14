# HF-URLs: agent notes

- Release tier: **small** (`v1.0`, `v1.1`, …). A release means a rebuilt `data/` from a full `uv run scripts/build.py` run, with counts checked.
- `data/hf-urls.json`, `data/languages.json` and `docs/*.md` are generated. Edit `scripts/build.py` or `data/ui-filters.json`, never the outputs.
- The URL rules in `README.md` and `RULES` in `build.py` were measured on 2026-11-14 in the logged-in agent browser and checked with counts. When the Hub changes, re-measure them: drive the sidebar on https://huggingface.co/models and https://huggingface.co/datasets, read `location.search` after each click, and compare result counts.
- Consumer: `HF-Research-Plugin` (`~/repos/github/ai-agents-and-prompts/ai-claude-plugins/HF-Research-Plugin`) ships a copy under `data/hf-urls/` and records the HF-URLs release in `data/hf-urls/VERSION`. After a release, copy `data/*.json` there and bump the plugin.
