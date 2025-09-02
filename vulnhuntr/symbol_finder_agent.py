import subprocess
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import structlog

from vulnhuntr.prompts import VULN_SPECIFIC_BYPASSES_AND_PROMPTS
from vulnhuntr.LLMs import OpenRouter


class SymbolExtractorAgent:
    """
    Agent-style symbol finder that uses ripgrep and lightweight LLM ranking to
    discover code snippets. Mimics the public API of the existing SymbolExtractor
    (extract, find_callers, get_enclosing_symbol, static_bypass_scan).

    - Search backend: ripgrep (rg) for speed. Falls back to Python scanning if rg not available.
    - Snippet collection: Python file reads (optionally could use `cat`, but Python is more reliable cross-platform).
    - LLM: OpenRouter with a cheap model (default: openai/gpt-5-nano) for tie-breaking/ranking when multiple candidates exist.
    - Parallelism: ripgrep invocations and file reads can run concurrently using a thread pool.
    """

    def __init__(self, repo_path: str | Path, model: str = "openai/gpt-5-nano", base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.repo_path = Path(repo_path)
        self.rg_available = shutil.which("rg") is not None
        if not self.rg_available:
            raise RuntimeError("ripgrep (rg) is required for SymbolExtractorAgent. Please install ripgrep and ensure 'rg' is on PATH.")
        self.llm = OpenRouter(model=model, base_url=base_url or "https://openrouter.ai/api/v1", system_prompt="Rank code matches by likelihood of being the requested symbol.", api_key=api_key)
        self.log = structlog.get_logger("symbol_finder_agent")

    # ------------------------------
    # Public API (compatible surface)
    # ------------------------------
    def extract(self, module_name: Optional[str], class_name: Optional[str], entity_name: str, symbol_kind: str, filtered_files: List[Path]) -> Dict[str, Any]:
        """
        Locate the definition of the requested function or class and return a snippet.
        Returns dict with keys: name, context_name_requested, file_path, source
        """
        # Log the request line-by-line
        self.log.info("Agent symbol request", request_type="REQUEST_DEFINITION", module_name=module_name, class_name=class_name, entity_name=entity_name, symbol_kind=symbol_kind)
        # Build a ripgrep pattern for Python definitions
        if symbol_kind == 'function':
            pattern = rf"^\s*def\s+{re.escape(entity_name)}\s*\("
        else:
            pattern = rf"^\s*class\s+{re.escape(class_name or entity_name)}\b"

        # Phase 1: search only within hinted module path if provided
        narrowed_files: List[Path] = []
        module_hint = module_name.replace('.', '/') if module_name else ''
        if module_hint:
            for fp in filtered_files:
                if module_hint in str(fp):
                    narrowed_files.append(fp)

        def _try_candidates(files_scope: List[Path]) -> Optional[Dict[str, Any]]:
            candidates = self._search_pattern_across_repo(pattern, files_scope)
            if not candidates:
                return None
            ranked = self._rank_candidates(candidates, entity_name, class_name, module_hint)
            for cand in ranked:
                snippet = self._extract_definition_snippet(cand['file_path'], cand['line'])
                if not snippet:
                    continue
                if not self._snippet_header_matches(symbol_kind, (class_name or entity_name), snippet):
                    continue
                # Log the result for this specific request
                try:
                    self.log.info("Agent symbol result", request_type="REQUEST_DEFINITION", returned_name=(class_name or entity_name), file_path=cand['file_path'], line=cand['line'])
                except Exception:
                    pass
                return {
                    'name': entity_name if symbol_kind == 'function' else (class_name or entity_name),
                    'context_name_requested': entity_name if symbol_kind == 'function' else (class_name or entity_name),
                    'file_path': cand['file_path'],
                    'source': snippet,
                }
            return None

        res = None
        if narrowed_files:
            res = _try_candidates(narrowed_files)
        if not res:
            res = _try_candidates(filtered_files)
        if not res:
            # Definitively report NOT_FOUND so the model can adjust its request
            self.log.info("Agent symbol result", request_type="REQUEST_DEFINITION", returned_name="NOT_FOUND", module_name=module_name, entity_name=entity_name)
            return self._not_found(entity_name, class_name, module_name, symbol_kind)
        return res

    def find_callers(self, symbol_name: str, files: List[Path]) -> List[Dict[str, Any]]:
        """Find call sites for a symbol_name across the repo."""
        try:
            self.log.info("Agent symbol request", request_type="REQUEST_CALLERS", entity_name=symbol_name)
        except Exception:
            pass
        direct_pat = rf"\b{re.escape(symbol_name)}\s*\("
        method_pat = rf"\.{re.escape(symbol_name)}\s*\("
        results: List[Dict[str, Any]] = []
        matches = self._search_multi_patterns({"direct": direct_pat, "method": method_pat}, files)
        for kind, lst in matches.items():
            for m in lst:
                results.append({
                    'file_path': m['file_path'],
                    'line_no': m['line'],
                    'line_text': m['text'],
                    'kind': kind,
                })
        self.log.info("Agent symbol result", request_type="REQUEST_CALLERS", returned_count=len(results))
        return results

    def get_enclosing_symbol(self, file_path: Path, line_no: int) -> Dict[str, Any] | None:
        """Heuristic to find the smallest enclosing function or class around line_no."""
        try:
            lines = file_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            return None
        idx = min(max(line_no - 1, 0), len(lines) - 1)
        # Walk upwards to find the nearest def/class
        for i in range(idx, -1, -1):
            line = lines[i]
            m_def = re.match(r"^\s*def\s+(\w+)\s*\(", line)
            m_cls = re.match(r"^\s*class\s+(\w+)\b", line)
            if m_def or m_cls:
                name = (m_def or m_cls).group(1)
                kind = 'function' if m_def else 'class'
                start = (i + 1, len(line) - len(line.lstrip(' ')))
                end_line = self._find_block_end(lines, i)
                end = (end_line + 1, len(lines[end_line]) if end_line < len(lines) else 0)
                return {"name": name, "type": kind, "start": start, "end": end}
        return None

    def static_bypass_scan(self, files: List[Path]) -> List[Dict[str, Any]]:
        compiled: Dict[str, List[Dict[str, Any]]] = {}
        for vuln_type, cfg in VULN_SPECIFIC_BYPASSES_AND_PROMPTS.items():
            pats = []
            for ex in cfg.get('bypasses', []) or []:
                if not ex:
                    continue
                try:
                    pats.append(re.compile(re.escape(ex), re.IGNORECASE))
                except re.error:
                    continue
            compiled[vuln_type] = pats

        results: List[Dict[str, Any]] = []
        def _scan_file(fp: Path) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            try:
                with fp.open(encoding='utf-8') as f:
                    for idx, line in enumerate(f, start=1):
                        for vuln_type, plist in compiled.items():
                            for pat in plist:
                                if pat.search(line):
                                    out.append({
                                        'vuln_type': vuln_type,
                                        'file_path': str(fp),
                                        'line_no': idx,
                                        'line_text': line.rstrip('\n'),
                                        'pattern': pat.pattern,
                                        'enclosing': self.get_enclosing_symbol(fp, idx),
                                    })
            except Exception:
                return out
            return out

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_scan_file, fp): fp for fp in files}
            for fut in as_completed(futures):
                results.extend(fut.result())
        return results

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _search_pattern_across_repo(self, pattern: str, files_filter: List[Path]) -> List[Dict[str, Any]]:
        if not self.rg_available:
            raise RuntimeError("ripgrep (rg) is required but not available.")
        files_set = {str(p) for p in files_filter}
        results: List[Dict[str, Any]] = []
        cmd = ["rg", "-nH", "-S", "-t", "py", pattern, str(self.repo_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.repo_path))
        if proc.returncode not in (0, 1):
            return results
        for line in proc.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            fp, ln, text = parts[0], parts[1], parts[2]
            abs_fp = str(Path(self.repo_path / fp).resolve())
            if files_set and abs_fp not in files_set:
                continue
            try:
                line_no = int(ln)
            except ValueError:
                continue
            results.append({"file_path": abs_fp, "line": line_no, "text": text})
        return results

    def _search_multi_patterns(self, pattern_map: Dict[str, str], files_filter: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
        if not self.rg_available:
            raise RuntimeError("ripgrep (rg) is required but not available.")
        out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in pattern_map.keys()}
        files_set = {str(p.resolve()) for p in files_filter}
        with ThreadPoolExecutor(max_workers=min(8, len(pattern_map) or 1)) as pool:
            futures = {}
            for name, pat in pattern_map.items():
                cmd = ["rg", "-nH", "-S", "-t", "py", pat, str(self.repo_path)]
                futures[pool.submit(subprocess.run, cmd, capture_output=True, text=True, cwd=str(self.repo_path))] = name
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    proc = fut.result()
                except Exception:
                    continue
                if getattr(proc, 'returncode', 1) not in (0, 1):
                    continue
                for line in proc.stdout.splitlines():
                    parts = line.split(":", 2)
                    if len(parts) < 3:
                        continue
                    fp, ln, text = parts[0], parts[1], parts[2]
                    abs_fp = str(Path(self.repo_path / fp).resolve())
                    if files_set and abs_fp not in files_set:
                        continue
                    try:
                        line_no = int(ln)
                    except ValueError:
                        continue
                    out[name].append({"file_path": abs_fp, "line": line_no, "text": text})
        return out

    def _rank_candidates(self, candidates: List[Dict[str, Any]], entity_name: str, class_name: Optional[str], module_hint: str) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        # Hard preference: module path hint match
        def score(c: Dict[str, Any]) -> int:
            s = 0
            if module_hint and module_hint in c['file_path']:
                s += 10
            # Closer match near start of line
            if c.get('text', '').lstrip().startswith(('def', 'class')):
                s += 2
            return s
        ranked = sorted(candidates, key=score, reverse=True)
        # Use the LLM to break close ties if needed (top-3)
        if len(ranked) > 1:
            top = ranked[:3]
            prompt = (
                f"Select the best definition for symbol '{class_name or entity_name}'.\n"
                f"Module hint: '{module_hint or 'N/A'}'\n"
                f"Candidates:\n" + "\n".join([f"- {i}: {c['file_path']}:{c['line']} -> {c.get('text','')[:200]}" for i, c in enumerate(top)]) + "\n"
                "Answer with the index (0-based) only."
            )
            try:
                idx = self._ask_index(prompt, len(top))
                if idx is not None:
                    return [top[idx]] + [c for i, c in enumerate(ranked) if i != idx]
            except Exception:
                pass
        return ranked

    def _ask_index(self, prompt: str, n: int) -> Optional[int]:
        ans = self.llm.chat(prompt)
        m = re.search(r"\b(\d+)\b", ans or "")
        if not m:
            return None
        idx = int(m.group(1))
        if 0 <= idx < n:
            return idx
        return None

    def _extract_definition_snippet(self, file_path: str, start_line: int) -> Optional[str]:
        fp = Path(file_path)
        try:
            lines = fp.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            return None
        i = start_line - 1
        if i < 0 or i >= len(lines):
            return None
        indent = len(lines[i]) - len(lines[i].lstrip(' '))
        end = self._find_block_end(lines, i)
        snippet = "\n".join(lines[i:end + 1])
        return snippet or None

    def _snippet_header_matches(self, symbol_kind: str, name: str, snippet: str) -> bool:
        if not snippet:
            return False
        first = snippet.splitlines()[0].lstrip()
        if symbol_kind == 'function':
            return bool(re.match(rf"^def\s+{re.escape(name)}\s*\(", first))
        else:
            return bool(re.match(rf"^class\s+{re.escape(name)}\b", first))

    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip(' '))
        last = start_idx
        for j in range(start_idx + 1, len(lines)):
            line = lines[j]
            if not line.strip():
                last = j
                continue
            cur_indent = len(line) - len(line.lstrip(' '))
            if cur_indent <= base_indent and re.match(r"^\S", line):
                break
            last = j
        return last

    def _not_found(self, entity_name: str, class_name: Optional[str], module_name: str, symbol_kind: str) -> Dict[str, Any]:
        return {
            'name': entity_name if symbol_kind == 'function' else (class_name or entity_name),
            'context_name_requested': f"{module_name}.{class_name + '.' if class_name else ''}{entity_name}",
            'file_path': 'NOT_FOUND',
            'source': (
                "NOT_FOUND: Symbol not found via ripgrep agent. Verify module/class/entity names and paths; "
                "only repository files are searched."
            )
        }


