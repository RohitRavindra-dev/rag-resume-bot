"""
test.py
Production-ready small RAG runner inspired by `ragbot.ipynb`.

Features:
- Loads environment variables from `.env` (OpenAI key or other credentials)
- Builds or loads a persisted index from `data/` -> `index_storage/`
- Uses HuggingFace embeddings and Ollama (falls back to OpenAI if Ollama not available)
- Simple CLI: run a single query or rebuild the index

Notes:
- This script tries to mirror the notebook's flow but is defensive about imports and APIs.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def safe_imports():
    """Try to import required llama-index components and return them.

    The notebook used `VectorStoreIndex`, `SimpleDirectoryReader`, `StorageContext`, and
    `load_index_from_storage`. Different versions of `llama-index` may expose slightly
    different top-level names; we attempt the common locations and provide a clear
    error message if something is missing.
    """
    # Try a few different import locations to support multiple llama-index versions
    SimpleDirectoryReader = ServiceContext = VectorStoreIndex = Settings = None
    load_index_from_storage = None
    StorageContext = None
    HuggingFaceEmbedding = None

    # core symbols
    try:
        from llama_index.core import (
            SimpleDirectoryReader,
            ServiceContext,
            VectorStoreIndex,
            Settings,
        )
    except Exception:
        try:
            # older/newer versions may expose some names at top-level
            from llama_index import SimpleDirectoryReader, ServiceContext, Settings
        except Exception:
            logger.debug("Could not import core symbols from top-level or core module.")

    # load_index_from_storage / StorageContext - try several locations
    try:
        from llama_index import load_index_from_storage, StorageContext
    except Exception:
        try:
            from llama_index.core import load_index_from_storage
        except Exception:
            load_index_from_storage = None
        try:
            from llama_index.storage import StorageContext
        except Exception:
            try:
                from llama_index.core import StorageContext
            except Exception:
                StorageContext = None

    # embeddings - try a couple of likely locations
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except Exception:
        try:
            from llama_index.embeddings import HuggingFaceEmbedding
        except Exception:
            HuggingFaceEmbedding = None

    # LLM imports: Ollama preferred in notebook; fall back to OpenAI if Ollama unavailable
    llm_impl = None
    try:
        from llama_index.llms.ollama import Ollama

        llm_impl = ("ollama", Ollama)
    except Exception:
        try:
            from llama_index.llms.openai import OpenAI

            llm_impl = ("openai", OpenAI)
        except Exception:
            logger.warning(
                "Neither Ollama nor OpenAI LLM wrappers are available in llama-index."
            )

    # pprint helper
    try:
        from llama_index.core.response.pprint_utils import pprint_response
    except Exception:
        pprint_response = None

    return {
        "SimpleDirectoryReader": SimpleDirectoryReader,
        "ServiceContext": ServiceContext,
        "VectorStoreIndex": VectorStoreIndex,
        "Settings": Settings,
        "load_index_from_storage": load_index_from_storage,
        "StorageContext": StorageContext,
        "HuggingFaceEmbedding": HuggingFaceEmbedding,
        "llm_impl": llm_impl,
        "pprint_response": pprint_response,
    }


def build_or_load_index(
    data_dir: str = "data",
    persist_dir: str = "index_storage",
    llm_model: str = "llama3",
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Build a new index from `data_dir` or load it from `persist_dir`.

    Returns the loaded/built index object.
    """
    imports = safe_imports()

    SimpleDirectoryReader = imports["SimpleDirectoryReader"]
    VectorStoreIndex = imports["VectorStoreIndex"]
    StorageContext = imports["StorageContext"]
    load_index_from_storage = imports["load_index_from_storage"]
    HuggingFaceEmbedding = imports["HuggingFaceEmbedding"]
    Settings = imports["Settings"]

    llm_impl = imports["llm_impl"]

    persist_path = Path(persist_dir)
    if not persist_path.exists():
        logger.info(
            "No persisted index found — building a new one from '%s'.", data_dir
        )
        documents = SimpleDirectoryReader(data_dir).load_data()

        # instantiate embedding model
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

        # instantiate LLM wrapper (try Ollama first, then OpenAI)
        llm = None
        if llm_impl is not None:
            name, impl = llm_impl
            logger.info("Using LLM wrapper: %s", name)
            if name == "ollama":
                # Ollama typically talks to a local Ollama server; the notebook used Ollama(model="llama3")
                llm = impl(model=llm_model)
            else:
                # OpenAI wrapper uses OPENAI_API_KEY from env
                llm = impl()
        else:
            logger.warning(
                "No LLM wrapper available. Some functionality may be limited."
            )

        # Apply as global Settings like the notebook did (keeps compatibility across versions)
        try:
            Settings.llm = llm
            Settings.embed_model = embed_model
        except Exception:
            logger.debug(
                "Could not assign to Settings; proceeding without global Settings assignment."
            )

        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        try:
            # some versions expose storage_context on index
            index.storage_context.persist(persist_dir=persist_dir)
        except Exception:
            logger.debug("Index storage persist step failed (API mismatch).")
        logger.info("Index built and persisted to '%s'.", persist_dir)
    else:
        logger.info(
            "Persist directory '%s' found; attempting to load index.", persist_dir
        )
        if load_index_from_storage is not None and StorageContext is not None:
            try:
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                index = load_index_from_storage(storage_context)
            except Exception:
                logger.exception(
                    "Loading index failed; will attempt to rebuild from data directory."
                )
                documents = SimpleDirectoryReader(data_dir).load_data()
                index = VectorStoreIndex.from_documents(documents, show_progress=True)
                try:
                    index.storage_context.persist(persist_dir=persist_dir)
                except Exception:
                    logger.debug("Persist after rebuild failed (API mismatch).")
        else:
            logger.warning(
                "load_index_from_storage or StorageContext not available for this llama-index version; rebuilding index."
            )
            documents = SimpleDirectoryReader(data_dir).load_data()
            index = VectorStoreIndex.from_documents(documents, show_progress=True)
            try:
                index.storage_context.persist(persist_dir=persist_dir)
            except Exception:
                logger.debug("Persist after rebuild failed (API mismatch).")

    return index


def query_index(index, query: str, show_source: bool = True):
    """Run a single query against the index and return the result.

    We return both the raw response and an optionally pretty-printed version.
    """
    imports = safe_imports()
    pprint_response = imports.get("pprint_response")

    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    if pprint_response is not None:
        try:
            pprint = lambda r: pprint_response(r, show_source=show_source)
        except Exception:
            pprint = None
    else:
        pprint = None

    return response, pprint


def main():
    parser = argparse.ArgumentParser(
        description="Small RAG runner converted from ragbot.ipynb"
    )
    parser.add_argument("--query", "-q", help="Query to ask the index", default=None)
    parser.add_argument(
        "--rebuild", "-r", help="Force rebuild the index", action="store_true"
    )
    parser.add_argument("--data-dir", help="Directory with documents", default="data")
    parser.add_argument(
        "--persist-dir", help="Index storage directory", default="index_storage"
    )
    args = parser.parse_args()

    if args.rebuild:
        # remove persist dir so a fresh index gets built
        p = Path(args.persist_dir)
        if p.exists():
            logger.info(
                "Removing existing persist directory '%s' (rebuild requested)",
                args.persist_dir,
            )
            shutil.rmtree(p)

    try:
        index = build_or_load_index(
            data_dir=args.data_dir, persist_dir=args.persist_dir
        )
    except Exception as exc:
        logger.exception(
            "Failed to build or load index. Check imports and installed llama-index version."
        )
        sys.exit(1)

    if args.query:
        response, pprint = query_index(index, args.query)
        # If a pretty printer is available, use it. Otherwise, print raw response.
        if pprint:
            pprint(response)
        else:
            print(response)
    else:
        logger.info(
            "No query provided. Use --query to ask the index, or --rebuild to rebuild it."
        )


if __name__ == "__main__":
    main()
