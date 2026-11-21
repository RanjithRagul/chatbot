# AI Chatbot

A command-line chatbot that uses configured model providers for general questions and a local PDF knowledge base for document questions. The application discovers compatible models at startup, routes document questions to a FAISS retrieval and reranking pipeline, and falls back between available models when a request fails.

## Requirements

- Python 3.10 or newer
- API key for at least one supported provider: Groq, Google Gemini, OpenAI, or Anthropic
- Internet access for model discovery and chat requests
- Disk space and internet access on first RAG use to download the embedding and reranking models

The RAG pipeline can run on CPU. It uses CUDA when available.

## Setup

From the repository root, create and activate a virtual environment, then install the project dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` in place of the Windows commands.

Copy `.env.example` to `.env` and fill in at least one provider key. The application reads keys from the process environment or `.env` in the repository root. Keep `.env` private; it is ignored by Git.

## Run the chatbot

```shell
python -m chatbot.main
```

At startup, the app discovers models from providers with configured keys. Enter `/models` to list discovered models, `/clear` to clear conversation history, or `/exit` to quit.

## Add or update PDF knowledge

1. Put PDF files in `pdf/pdf_files/` (create the directory if needed).
2. Extract their text:

   ```shell
   python -m pdf.text_extractor_from_pdf
   ```

3. Build or rebuild the FAISS index:

   ```shell
   python db/1_build_faiss_index.py
   ```

The extractor writes page-marked text files under `pdf/extracted_logs/`. The index builder reads those files and writes the index, metadata, and source manifest under `db/faiss_index/`. The chatbot checks the manifest at RAG startup and reports an error if the extracted documents changed after the last index build.

## Tests

Run the automated checks from the repository root:

```shell
python -m unittest discover -s test -v
```

Tests use mocks for provider and model calls so they do not require API credentials or make network requests.

## Logs and privacy

Runtime event logs are written to `log/`. The logger records message inputs and model outputs for diagnostics, so these files may contain conversation content. Avoid sharing logs without reviewing them first.
