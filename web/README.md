# Local web demo

This is a chat-only browser interface backed by a Cloudflare Python Worker. The existing Python CLI, router, PDF ingestion, and RAG folders remain unchanged; the web demo does not load or call the RAG pipeline.

## Prerequisites

- Python 3.13 or newer for the local Worker toolchain
- Node.js 22 or newer
- `uv` installed and available on `PATH`
- A Groq API key to test live responses; API keys are never placed in browser files

## Local development

From this directory:

```powershell
Copy-Item worker/.dev.vars.example worker/.dev.vars
```

Edit `worker/.dev.vars` and set `GROQ_API_KEY`, then start the local Worker:

```powershell
Set-Location worker
uv sync
uv run pywrangler dev
```

Open the local address printed by Wrangler, usually `http://localhost:8787`. Static files are served from `frontend/`; API routes are `/api/health` and `/api/chat`.

The local D1 database is separate from any Cloudflare database. Apply its schema with:

```powershell
uv run pywrangler d1 migrations apply chatbot-demo-usage --local
```

The worker counts each Groq completion attempt against a shared monthly cap of 900. Failed provider attempts count too, keeping the cap conservative. Logs contain request metadata and error status only; they do not record chat contents or API keys.

## Tests

From the repository root:

```powershell
python -m unittest discover -s test -v
```

The repository tests do not need a Groq key or make network requests. To verify live chat, configure a valid key in `worker/.dev.vars` and try the browser demo.

## Before a future deploy

The placeholder D1 database ID in `worker/wrangler.jsonc` is for local development only. A deployment would require creating a Cloudflare D1 database and binding its real ID, then adding `GROQ_API_KEY` as a Worker secret. Local development does not deploy or modify Cloudflare resources.
