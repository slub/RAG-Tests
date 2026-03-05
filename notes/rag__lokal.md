---
tags:
  - ai
  - documentation
  - llm
  - rag
from_obsidian: true
---

## API-KEY

Get a ScaDS.AI Key from: https://llm.scads.ai/docs/usage/api/

Store your OPENAI API Key as environmental variable:
`export SCADSAI_KEY=sk-...`

> Add the key to your `.bashrc` so that is is available everytime you start a terminal:
> `echo "" >> ~/.bashrc` 
> `echo "# Add ScaDS.AI LLM key" >> ~/.bashrc` 
> `echo "export SCADSAI_KEY=sk-..." >> ~/.bashrc` 


> **Check available ScaDS.AI models:** 
> `curl -s https://llm.scads.ai/v1/models   -H "Authorization: Bearer $SCADSAI_KEY" | python -m json.tool`
> 
> We use the embedding model `Qwen/Qwen3-Embedding-4B` (2026-02-16).
> 
> To chat we use `openGPT-X/Teuken-7B-instruct-v0.6` (2026-02-16).


## UV

For this project we use [`uv`](https://docs.astral.sh/uv/) for virtual environment and package managing in Python

> `uv` is a Rust software by https://astral.sh/. They also have a Linter and TypeChecker for Python written in Rust.

Install with:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

To run `uv` without restarting the shell run:
`source $HOME/.local/bin/env`

### Optional: Add more system libraries for better file reading

```
sudo apt update
sudo apt install -y libmagic-dev
sudo apt install -y poppler-utils tesseract-ocr tesseract-ocr-all
sudo apt install -y libreoffice
sudo apt install -y pandoc
sudo apt install -y libgl1
```



## Setup

We install everything in a venv, within a folder called vectorstore:

```bash
mkdir vectorstore
cd vectorstore

uv init
uv venv --python 3.12
source .venv/bin/activate
```

Now we install  the required python libraries:

```bash
uv add "unstructured[all-docs]" pymilvus milvus-lite openai
# On Mint I required this
uv pip install "setuptools<71"
```

Before we start, we need documents to retrieve information from. We create a folder claeed `knowledge_base` and download a test file:

```bash
mkdir knowledge_base
curl -L "https://www.tuxcademy.org/download/de/lxk1/lxk1-de-manual.pdf" -o knowledge_base/linux_kompakt.pdf
```

Feel free to add additional contents.
## Create Vectorstore

1) Run `create_vectorestore.py`
2) Run `chat_with_vectorstore.py`
	- Try " Why should I install linux?" (knowledge base contains infos about this)
	- And then "How about gaming in linux?" (no infos in knowledge base)

With the script `create_vectorstore.py`, we build a vector store using Milvus (here as a local Milvus Lite file `milvus.db`). The script uses the Unstructured library to partition all files in the knowledge base into *elements* and enriches each element with metadata (e.g., source path, filename, page information when available). We also use the partion option `chunking`, thus restructuring the originally created elements into semantic chunks. If `WRITE_STRUCTURED=True`, the chunks (or elements) are also written to JSON files.

Next, embeddings are computed for each element’s text using the embedding model (`Qwen/Qwen3-Embedding-4B`).

> An embedding maps text to a high-dimensional vector space so that semantic similarity can be approximated via distance/similarity measures (e.g., cosine similarity).

The resulting vectors, together with the original text and metadata, are inserted into a Milvus collection, forming the vector store.

Once the vector store is ready, we can launch `chat_with_vectorstore.py`. This script embeds each user question with the same embedding model, retrieves the most similar elements from Milvus (top-k nearest neighbors), and sends the retrieved texts as context to the chat model (`Teuken-7B-instruct-v0.6`). The metadata is used to generate source identifiers (e.g., `[report.pdf#p12]`) so the assistant can reference where information came from. The system prompt defines the assistant’s behavior, including how it should cite provided sources. 

### `create_vectorstore.py`

```python
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from pymilvus import DataType, MilvusClient
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_to_json


# =========================
# Configuration
# =========================

# IO
KB_ROOT = Path("knowledge_base")
OUT_ROOT = Path("unstructured_outputs")

# write results from unstrucured to json
WRITE_STRUCTURED = True

# languages of inputs
LANGUAGES = ["eng", "deu"]

COLLECTION_NAME = "my_rag_collection"
MILVUS_URI = "./milvus.db"

# SCADS OpenAI-compatible endpoint + model
OPENAI_BASE_URL = "https://llm.scads.ai/v1"
EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DIM = 2560


# Safety switches
DROP_COLLECTION_IF_EXISTS = True  # set False if you don't want destructive behavior

# Skip these file types
SKIP_SUFFIXES = {".py", ".xlsx"}


# =========================
# Clients
# =========================
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("SCADSAI_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable SCADSAI_KEY.")
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=api_key)


def get_milvus_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)


# =========================
# Embeddings
# =========================
def embed_texts(openai_client: OpenAI, texts: List[str]) -> List[List[float]]:
    """
    Batch embedding. Returns list of embeddings in the same order as texts.
    """
    # Defensive: remove None, keep alignment
    resp = openai_client.embeddings.create(input=texts, model=EMBED_MODEL)
    # Ensure ordering by index (some APIs return already-ordered, but this is safer)
    data_sorted = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in data_sorted]


# =========================
# Milvus schema / collection
# =========================
def ensure_collection(milvus_client: MilvusClient, collection_name: str) -> None:
    if milvus_client.has_collection(collection_name):
        if DROP_COLLECTION_IF_EXISTS:
            milvus_client.drop_collection(collection_name)
        else:
            milvus_client.load_collection(collection_name=collection_name)
            return

    schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM
    )
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="AUTOINDEX",
    )

    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Bounded",
    )
    milvus_client.load_collection(collection_name=collection_name)


# =========================
# Helpers
# =========================
def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def should_skip_file(path: Path) -> bool:
    if is_hidden(path):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def element_metadata(element: Any, file_path: Path) -> Dict[str, Any]:
    md = {}
    try:
        md = element.metadata.to_dict() or {}
    except Exception:
        md = {}
    md["source_path"] = str(file_path)
    md["source_name"] = file_path.name
    return md


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and not should_skip_file(p):
            yield p


# =========================
# Ingestion
# =========================
def insert_elements(
    milvus_client: MilvusClient,
    openai_client: OpenAI,
    collection_name: str,
    elements: List[Any],
    file_path: Path,
    start_id: int,
) -> int:
    # Filter out empty/whitespace text early
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for el in elements:
        text = getattr(el, "text", "") or ""
        text = text.strip()
        if not text:
            continue
        texts.append(text)
        metas.append(element_metadata(el, file_path))

    if not texts:
        return start_id

    vectors = embed_texts(openai_client, texts)

    data: List[Dict[str, Any]] = []
    cur = start_id
    for text, vector, md in zip(texts, vectors, metas):
        data.append(
            {
                "id": cur,
                "vector": vector,
                "text": text,
                "metadata": md,
            }
        )
        cur += 1

    milvus_client.insert(collection_name=collection_name, data=data)
    return cur


def write_elements_json(
    elements: List[Any], file_path: Path, kb_root: Path, out_root: Path
) -> Path:
    relative_folder = file_path.parent.relative_to(kb_root)
    out_dir = out_root / relative_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{file_path.stem}-output.json"
    elements_to_json(elements=elements, filename=str(out_file))
    return out_file


def main() -> None:
    if not KB_ROOT.exists():
        raise RuntimeError(f"Knowledge base folder not found: {KB_ROOT.resolve()}")
    if OUT_ROOT.exists():
        print(f"Removing existing output folder: {OUT_ROOT.resolve()}")
        shutil.rmtree(OUT_ROOT)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client()

    ensure_collection(milvus_client, COLLECTION_NAME)

    documents = 0
    next_id = 0

    for file_path in iter_files(KB_ROOT):
        try:
            elements = partition(
                filename=str(file_path),
                chunking_strategy="basic",
                languages=LANGUAGES,
                include_page_breaks=True,
                include_orig_elements=False,
                overlap=50,
                overlap_all=False
            )

            next_id = insert_elements(
                milvus_client=milvus_client,
                openai_client=openai_client,
                collection_name=COLLECTION_NAME,
                elements=elements,
                file_path=file_path,
                start_id=next_id,
            )
            if WRITE_STRUCTURED:
                write_elements_json(elements, file_path, KB_ROOT, OUT_ROOT)

            print(f"Processed: {file_path}")
            documents += 1

        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")

    print(f"Read {documents} documents. Inserted up to id={next_id - 1}.")


if __name__ == "__main__":
    main()
```

### `chat_with_vectorstore.py`

```python
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from pymilvus import MilvusClient

# =========================
# Config
# =========================
BASE_URL = "https://llm.scads.ai/v1"
COLLECTION_NAME = "my_rag_collection"
MILVUS_URI = "./milvus.db"

EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
CHAT_MODEL = "openGPT-X/Teuken-7B-instruct-v0.6"

TOP_K = 8
MAX_HISTORY_TURNS = 8  # user+assistant pairs

# =========================
# Prompts
# =========================

SYSTEM_PROMPT = (
    "Try answering with the providied sources.\n"
    "If you use provided sources cite them with bracketed SOURCE key, e.g. [report.pdf#p12].\n"
    "Clearly distinguish between information that are provided in the sources and others.\n"
    "Write in the user's language."
)

USER_PROMPT_TEMPLATE = (
    "Use the following sources to answer the question.\n\n"
    "{context}\n\n"
    "Question: {question}"
)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing environment variable {name}.")
    return val


OPENAI_KEY = _require_env("SCADSAI_KEY")

# =========================
# Clients
# =========================
openai_client = OpenAI(base_url=BASE_URL, api_key=OPENAI_KEY)
milvus_client = MilvusClient(uri=MILVUS_URI)


# =========================
# Utility helpers
# =========================
def _pick(*vals: Any) -> Optional[Any]:
    """Return the first non-empty value."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def _as_str(v: Any) -> str:
    return "" if v is None else str(v)


def emb_text(text: str) -> List[float]:
    text = text or ""
    resp = openai_client.embeddings.create(input=text, model=EMBED_MODEL)
    return resp.data[0].embedding


def retrieve_documents(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    vector = emb_text(question)

    # Explicit anns_field makes the intent clear and avoids surprises
    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[vector],
        anns_field="vector",
        limit=top_k,
        output_fields=["id", "text", "metadata"],
    )

    out: List[Dict[str, Any]] = []
    for res in search_res[0] if search_res else []:
        ent = res.get("entity", {}) or {}
        out.append(
            {
                "id": ent.get("id"),
                "text": ent.get("text") or "",
                "metadata": ent.get("metadata") or {},
                # Depending on Milvus version, this may be "distance" or "score"
                "score": res.get("distance", res.get("score")),
            }
        )
    return out


def source_locator(md: Dict[str, Any]) -> str:
    """
    Human-friendly pointer to the original document.
    Tries common unstructured keys + ingestion keys (source_path/source_name).
    """
    path = _pick(
        md.get("source_path"),
        md.get("file_path"),
        md.get("filename"),
        md.get("source_url"),
        md.get("url"),
    )
    name = _pick(md.get("source_name"), md.get("file_name"), md.get("document_title"))
    page = _pick(md.get("page_number"), md.get("page"), md.get("page_num"))
    section = _pick(md.get("section"), md.get("header"), md.get("title"))

    base = _pick(path, name, "unknown")
    extras: List[str] = []
    if page is not None and _as_str(page).strip() != "":
        extras.append(f"p.{page}")
    if section is not None and _as_str(section).strip() != "":
        extras.append(f"sec:{section}")

    return _as_str(base) + (f" ({', '.join(extras)})" if extras else "")


def citation_key(md: Dict[str, Any]) -> str:
    """
    Stable, human citation key across chunks from the same original doc.
    Examples:
      - report.pdf#p12
      - archive.txt#sec:Introduction
    """
    path = _pick(
        md.get("source_path"), md.get("file_path"), md.get("filename"), "unknown"
    )
    fname = os.path.basename(_as_str(path)) or "unknown"

    page = _pick(md.get("page_number"), md.get("page"))
    section = _pick(md.get("section"), md.get("header"), md.get("title"))

    key = fname
    if page not in (None, "", 0):
        key += f"#p{page}"
    elif section:
        s = _as_str(section).strip().replace("\n", " ")
        if len(s) > 40:
            s = s[:39] + "…"
        key += f"#sec:{s}"
    return key


def build_context(retrieved: List[Dict[str, Any]], max_chars_each: int = 1200) -> str:
    blocks: List[str] = []
    for d in retrieved:
        md = d.get("metadata") or {}
        key = citation_key(md)

        txt = (d.get("text") or "").strip()
        if len(txt) > max_chars_each:
            txt = txt[: max_chars_each - 1] + "…"

        blocks.append(f"SOURCE [{key}]:\n{txt}")

    return "\n\n".join(blocks)


def format_sources(retrieved: List[Dict[str, Any]], max_chars_each: int = 200) -> str:
    lines: List[str] = []
    seen: set[str] = set()

    for d in retrieved:
        md = d.get("metadata") or {}
        key = citation_key(md)
        if key in seen:
            continue
        seen.add(key)

        path = _pick(
            md.get("source_path"), md.get("file_path"), md.get("filename"), "unknown"
        )
        page = _pick(md.get("page_number"), md.get("page"))
        where = _as_str(path) + (f" (p.{page})" if page not in (None, "") else "")

        snippet = (d.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > max_chars_each:
            snippet = snippet[: max_chars_each - 1] + "…"

        lines.append(f"[{key}] {where}\n      {snippet}")

    return "\n".join(lines)


def stream_rag_answer(
    question: str,
    history: List[Dict[str, str]],
    top_k: int = TOP_K,
) -> Tuple[str, List[Dict[str, Any]]]:
    retrieved_docs = retrieve_documents(question, top_k=top_k)
    context = build_context(retrieved_docs)

    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    # Keep last N turns (each turn is user+assistant => 2 messages)
    trimmed_history = history[-(MAX_HISTORY_TURNS * 2) :]

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_prompt})

    stream = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        # temperature=0.2,
    )

    answer_parts: List[str] = []
    for event in stream:
        delta = event.choices[0].delta
        token = getattr(delta, "content", None)
        if token:
            answer_parts.append(token)
            print(token, end="", flush=True)

    answer = "".join(answer_parts).strip()
    return answer, retrieved_docs


def repl() -> None:
    print("Terminal RAG Chatbot (streaming + doc pointers)")
    print("Commands: /exit  /clear  /sources_on  /sources_off  /topk N\n")

    history: List[Dict[str, str]] = []
    show_sources = False
    top_k = TOP_K

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not question:
            continue

        if question in {"/exit", "/quit"}:
            print("Bye!")
            return

        if question == "/clear":
            history.clear()
            print("(history cleared)\n")
            continue

        if question == "/sources_on":
            show_sources = True
            print("(sources ON)\n")
            continue

        if question == "/sources_off":
            show_sources = False
            print("(sources OFF)\n")
            continue

        if question.startswith("/topk "):
            try:
                top_k = max(1, int(question.split(None, 1)[1]))
                print(f"(top_k set to {top_k})\n")
            except Exception:
                print("(usage: /topk N)\n")
            continue

        try:
            print("\nBot> ", end="", flush=True)
            answer, retrieved_docs = stream_rag_answer(question, history, top_k=top_k)
            print("\n")  # newline after streaming finishes

            if show_sources:
                print("Sources (original doc pointers + snippets):")
                print(format_sources(retrieved_docs))
                print()

        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}\n")
            continue

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    repl()
```
