# Local web demo

The site reuses the existing Python modules in `backend/src/`: chat rules in `backend/src/chatbot/`, usage tracking in `backend/src/db/`, and Worker events in `backend/src/log/`. `backend/src/worker.py` is Cloudflare's HTTP entry point. Browser files live in `web/frontend/`. The demo does not load RAG.

## Run locally

Requirements: Python 3.13+, Node.js 22+, `uv`, and a valid Groq API key.

From the repository root, create `backend/.dev.vars` and add your key:

```cmd
copy backend\.dev.vars.example backend\.dev.vars
notepad backend\.dev.vars
```

Then run:

```cmd
cd backend
uv sync
uv run pywrangler d1 migrations apply chatbot-demo-usage --local
uv run pywrangler dev
```

Open the local URL printed by Wrangler, usually `http://localhost:8787`. The local D1 database is separate from Cloudflare. Each Groq request attempt counts toward the 900-call monthly cap. Worker logs record request metadata, not chat contents or keys.

## Existing tests

From the repository root:

```cmd
python -m unittest discover -s test -v
```
