# DECISIONS

One line each: what — why — alternative.

- Server shows 224 CPUs (shared host); use ≤32 worker processes for tokenization — stay polite on a shared box — could use all 224.
- HF_HOME set to `~/.cache/huggingface` for downloads — prompt restricts server writes to `~/organic-growth` + `~/.cache` (image default is /workspace/.hf_home) — use image default.
