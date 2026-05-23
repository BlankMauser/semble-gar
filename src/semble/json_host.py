from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from semble.index import SembleIndex
from semble.types import GraphContext
from semble.utils import _is_git_url, _resolve_chunk


class _IndexCache:
    def __init__(self) -> None:
        self._entries: dict[str, SembleIndex] = {}

    def get(self, source: str, ref: str | None = None) -> tuple[SembleIndex, bool]:
        is_git = _is_git_url(source)
        cache_key = (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())
        existing = self._entries.get(cache_key)
        if existing is not None:
            return existing, True

        if is_git:
            index = SembleIndex.from_git(source, ref=ref)
        else:
            index = SembleIndex.from_path(cache_key)
        self._entries[cache_key] = index
        return index, False

    def close(self) -> None:
        for index in self._entries.values():
            index.close()
        self._entries.clear()


def _result_rows(index: SembleIndex, results: list[Any], *, compact: bool) -> list[dict[str, Any]]:
    contexts = index.get_context_for_results(results)
    output: list[dict[str, Any]] = []
    for result in results:
        context = contexts.get(result.chunk.location, GraphContext())
        row: dict[str, Any] = {
            "file": result.chunk.file_path,
            "line": f"{result.chunk.start_line}-{result.chunk.end_line}",
            "file_total_lines": result.chunk.file_total_lines,
            "score": round(result.score, 4),
            "source": result.source.value,
            "context": {
                "called_by": context.called_by,
                "depends_on": context.depends_on,
            },
        }
        if compact:
            row["code"] = result.chunk.content.split("\n", 1)[0].strip()
        else:
            row["code"] = result.chunk.content
        symbols = index.get_symbols_for_chunk(result.chunk)
        if symbols:
            row["symbols"] = symbols
        output.append(row)
    return output


def _pack_stats(index: SembleIndex) -> dict[str, Any]:
    stats = index.stats
    return {
        "indexed_files": stats.indexed_files,
        "total_chunks": stats.total_chunks,
        "languages": stats.languages,
    }


def _handle_search(cache: _IndexCache, params: dict[str, Any]) -> dict[str, Any]:
    source = str(params.get("repo") or params.get("path") or ".")
    mode = str(params.get("mode") or "hybrid")
    top_k = max(1, int(params.get("top_k") or params.get("topK") or 5))
    compact = bool(params.get("compact", True))
    filter_languages = params.get("filter_languages") or params.get("filterLanguages")
    filter_paths = params.get("filter_paths") or params.get("filterPaths")
    query = str(params.get("query") or "").strip()
    if not query:
        raise ValueError("Missing search query")

    started = time.perf_counter()
    index, cache_hit = cache.get(source, ref=params.get("ref"))
    results = index.search(
        query,
        top_k=top_k,
        mode=mode,
        filter_languages=filter_languages,
        filter_paths=filter_paths,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "repo": source,
        "mode": mode,
        "cache_hit": cache_hit,
        "elapsed_ms": round(elapsed_ms, 3),
        "index_stats": _pack_stats(index),
        "results": _result_rows(index, results, compact=compact),
    }


def _handle_find_related(cache: _IndexCache, params: dict[str, Any]) -> dict[str, Any]:
    source = str(params.get("repo") or params.get("path") or ".")
    file_path = str(params.get("file_path") or params.get("filePath") or "").strip()
    line = int(params.get("line") or 0)
    top_k = max(1, int(params.get("top_k") or params.get("topK") or 5))
    compact = bool(params.get("compact", True))
    if not file_path:
        raise ValueError("Missing file_path")
    if line <= 0:
        raise ValueError("Missing line")

    started = time.perf_counter()
    index, cache_hit = cache.get(source, ref=params.get("ref"))
    chunk = _resolve_chunk(index.chunks, file_path, line, file_mapping=index._file_mapping)
    if chunk is None:
        raise ValueError(f"No chunk found at {file_path}:{line}")
    results = index.find_related(chunk, top_k=top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "repo": source,
        "file_path": file_path,
        "line": line,
        "cache_hit": cache_hit,
        "elapsed_ms": round(elapsed_ms, 3),
        "index_stats": _pack_stats(index),
        "results": _result_rows(index, results, compact=compact),
    }


def _handle_trace_symbol(cache: _IndexCache, params: dict[str, Any]) -> dict[str, Any]:
    source = str(params.get("repo") or params.get("path") or ".")
    symbol = str(params.get("symbol") or "").strip()
    if not symbol:
        raise ValueError("Missing symbol")

    started = time.perf_counter()
    index, cache_hit = cache.get(source, ref=params.get("ref"))
    trace = index.trace_symbol(symbol)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "repo": source,
        "symbol": symbol,
        "cache_hit": cache_hit,
        "elapsed_ms": round(elapsed_ms, 3),
        "index_stats": _pack_stats(index),
        "trace": trace,
    }


def _handle_stats(cache: _IndexCache, params: dict[str, Any]) -> dict[str, Any]:
    source = str(params.get("repo") or params.get("path") or ".")
    started = time.perf_counter()
    index, cache_hit = cache.get(source, ref=params.get("ref"))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "repo": source,
        "cache_hit": cache_hit,
        "elapsed_ms": round(elapsed_ms, 3),
        "index_stats": _pack_stats(index),
    }


def _dispatch(cache: _IndexCache, request: dict[str, Any]) -> dict[str, Any]:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")

    if method == "ping":
        return {"id": request_id, "ok": True, "result": {"pong": True}}
    if method == "search":
        return {"id": request_id, "ok": True, "result": _handle_search(cache, params)}
    if method == "find_related":
        return {"id": request_id, "ok": True, "result": _handle_find_related(cache, params)}
    if method == "trace_symbol":
        return {"id": request_id, "ok": True, "result": _handle_trace_symbol(cache, params)}
    if method == "stats":
        return {"id": request_id, "ok": True, "result": _handle_stats(cache, params)}
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    cache = _IndexCache()
    try:
        for line in sys.stdin:
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            try:
                request = json.loads(text)
                response = _dispatch(cache, request)
            except Exception as exc:  # noqa: BLE001
                request_id = None
                try:
                    request_id = json.loads(text).get("id")
                except Exception:  # noqa: BLE001
                    pass
                response = {"id": request_id, "ok": False, "error": str(exc)}
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        cache.close()


if __name__ == "__main__":
    main()
