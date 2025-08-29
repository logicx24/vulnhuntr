import jedi
import pathlib
import re
import importlib
import inspect
import sysconfig
import os
import sys
from typing import List, Dict, Any, Optional
from jedi.api.classes import Name
from vulnhuntr.prompts import VULN_SPECIFIC_BYPASSES_AND_PROMPTS


class SymbolExtractor:
    def __init__(self, repo_path: str | pathlib.Path) -> None:
        self.repo_path = pathlib.Path(repo_path)
        self.environment = None
        py_exe = self._detect_venv_python()
        if py_exe:
            try:
                self.environment = jedi.create_environment(py_exe, safe=False)
            except Exception:
                self.environment = None
        # Ensure repository root is importable for path mapping (avoid importing repo modules directly)
        try:
            if str(self.repo_path) not in sys.path:
                sys.path.insert(0, str(self.repo_path))
        except Exception:
            pass
        # Initialize project; environment only assists inference, repo scanning is primary
        try:
            self.project = jedi.Project(self.repo_path, environment=self.environment) if self.environment else jedi.Project(self.repo_path)
        except Exception:
            self.project = jedi.Project(self.repo_path)
        self.parsed_symbols = None
        self.ignore = ['/test', '_test/', '/docs', '/example']
        self._script_cache: Dict[str, jedi.Script] = {}
    
    def extract(self,
                module_name: str,
                class_name: Optional[str],
                entity_name: str,
                symbol_kind: str,
                filtered_files: List) -> Dict:
        """
        Extracts the definition of a symbol from the repository.

        Edge cases:
            1. Method call on variable:
                    end_node = cast(BaseOperator, leaf_nodes[0]); end_node.call_stream(call_data):
            2. Class instance variable
                    multi_agents = MultiAgents(); multi_agents.method()
            3. Aliased import
                    from service import Service as FlowService
            4. Module symbol
                    asking for `app` when it's just an import statement like `from api.apps import app` and `app` is just a statement
            5. This stuff:
                    <Name full_name='api.db.services.document_service.DocumentService.update_progress.d', description='for d in docs: try: tsks = Task...>
                    And the bot asked for "name='DocumentService.create' reason='To understand how documents are created and how parser_config is set' code_line='cls.update_by_id(d["id"],"
                    The code_line is in the description. 
        """

        # Normalize module/class by handling cases where module_name includes a trailing class
        norm_module, norm_class = self._resolve_module_and_class(module_name, class_name)

        # Build script list from files under module_name path hint
        scripts: List[jedi.Script] = []
        for file in filtered_files:
            fstr = str(file)
            if norm_module.replace('.', '/') in fstr.replace('\\', '/'):
                try:
                    scripts.append(jedi.Script(path=file, project=self.project))
                except Exception:
                    continue
        # If no scripts matched the module hint, fall back to scanning all files
        if not scripts:
            for file in filtered_files:
                try:
                    scripts.append(jedi.Script(path=file, project=self.project))
                except Exception:
                    continue

        
        # Search using jedi.Script.search; uses the code_line from bot to grep for string in files
        search_token = entity_name if not norm_class else (entity_name if symbol_kind == 'function' else norm_class)
        match = self.file_search(search_token, scripts)
        if match:
            return match

        # Search using jedi.Project.search(); finds matches and class instance variables like "var = ClassName(); var.method()"
        match = self.project_search(search_token)
        if match:
            return match
        
        # Still no match, so we search using jedi.Script.get_names(); handles method calls on variables
        symbol_parts = (f"{norm_class}.{entity_name}" if norm_class else entity_name).split('.')
        match = self.all_names_search(search_token, symbol_parts, scripts, '')
        if match:
            return match

        # Final fallback within repository: regex-based definition search
        rx_match = self._regex_definition_search(entity_name, symbol_kind, filtered_files)
        if rx_match:
            return rx_match

        # Next, try standard library/builtins informational stub
        stdlib_info = self._stdlib_definition_info(norm_module, norm_class, entity_name, symbol_kind)
        if stdlib_info:
            return stdlib_info

        # Finally, try venv/third-party informational stub
        venv_info = self._venv_definition_info(norm_module, norm_class, entity_name, symbol_kind)
        if venv_info:
            return venv_info

        # Nothing found: return a sentinel to inform the model
        return {
            'name': entity_name if symbol_kind == 'function' else (norm_class or entity_name),
            'context_name_requested': f"{norm_module}.{norm_class + '.' if norm_class else ''}{entity_name}",
            'file_path': 'NOT_FOUND',
            'source': (
                "NOT_FOUND: Symbol not found in repository. Please refine module_name/class_name/entity_name "
                "(ensure correct module path and exact symbol name)."
            )
        }

    def _regex_definition_search(self, entity_name: str, symbol_kind: str, files: List[pathlib.Path]) -> Optional[Dict[str, Any]]:
        try:
            if symbol_kind == 'function':
                pattern = re.compile(rf"^\s*def\s+{re.escape(entity_name)}\s*\(.*", re.MULTILINE)
            else:
                pattern = re.compile(rf"^\s*class\s+{re.escape(entity_name)}\b.*", re.MULTILINE)
        except re.error:
            return None
        for file in files:
            try:
                text = file.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            m = pattern.search(text)
            if not m:
                continue
            # Grab a snippet around the match (definition line + up to 120 lines after)
            lines = text.splitlines()
            start_idx = text[:m.start()].count('\n')
            end_idx = min(len(lines), start_idx + 120)
            snippet = '\n'.join(lines[start_idx:end_idx])
            return {
                'name': entity_name,
                'context_name_requested': entity_name,
                'file_path': str(file),
                'source': snippet or 'None'
            }
        return None

    def _resolve_module_and_class(self, module_name: str, class_name: Optional[str]) -> tuple[str, Optional[str]]:
        """Handle cases where module_name accidentally includes a class segment.
        Example: module_name='engine.neo.hi.ClassName', class_name=None → returns ('engine.neo.hi', 'ClassName')
        """
        if class_name:
            return module_name, class_name
        parts = module_name.split('.')
        if len(parts) <= 1:
            return module_name, None
        # Try importing full module; if it fails, pop the last segment as potential class
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            return module_name, None
        potential_class = parts[-1]
        base_module = '.'.join(parts[:-1])
        try:
            base_spec = importlib.util.find_spec(base_module)
        except ModuleNotFoundError:
            base_spec = None
        if base_spec is None:
            return module_name, None
        # Heuristic: Treat last segment as class if it looks like a ClassName
        if potential_class and potential_class[:1].isupper():
            return base_module, potential_class
        return module_name, None

    def _is_stdlib_module(self, module_name: str) -> bool:
        try:
            try:
                spec = importlib.util.find_spec(module_name)
            except ModuleNotFoundError:
                return False
            if spec is None or spec.origin is None:
                return False
            if spec.origin == 'built-in':
                return True
            stdlib_path = sysconfig.get_paths().get('stdlib')
            if not stdlib_path:
                return False
            return spec.origin.startswith(stdlib_path) and 'site-packages' not in spec.origin
        except Exception:
            return False

    def _stdlib_definition_info(self, module_name: str, class_name: Optional[str], entity_name: str, symbol_kind: str) -> Optional[Dict[str, Any]]:
        """If the symbol appears to be in the standard library or builtins, return a structured stub.
        Attempts to include signature and doc excerpt for helpful context.
        """
        try:
            if not self._is_stdlib_module(module_name) and module_name != 'builtins':
                return None
            mod = importlib.import_module(module_name)
            target_obj = None
            if symbol_kind == 'class':
                target_name = class_name or entity_name
                target_obj = getattr(mod, target_name, None)
            elif symbol_kind == 'function':
                if class_name:
                    cls = getattr(mod, class_name, None)
                    target_obj = getattr(cls, entity_name, None) if cls is not None else None
                else:
                    target_obj = getattr(mod, entity_name, None)
            if target_obj is None:
                # Unknown in stdlib but in stdlib module; still return stub
                return {
                    'name': entity_name,
                    'context_name_requested': f"{module_name}.{class_name + '.' if class_name else ''}{entity_name}",
                    'file_path': f"STDLIB:{module_name}",
                    'source': 'Standard library reference; implementation is not included. Use known behavior of the stdlib symbol.'
                }
            sig = None
            try:
                sig = str(inspect.signature(target_obj))
            except Exception:
                sig = None
            doc = None
            try:
                doc = inspect.getdoc(target_obj)
            except Exception:
                doc = None
            summary = 'Standard library symbol; use known behavior. '
            if sig:
                summary += f"Signature: {sig}. "
            if doc:
                summary += f"Doc: {doc[:800]}"
            return {
                'name': entity_name if symbol_kind == 'function' else (class_name or entity_name),
                'context_name_requested': f"{module_name}.{class_name + '.' if class_name else ''}{entity_name}",
                'file_path': f"STDLIB:{module_name}",
                'source': summary
            }
        except Exception:
            return None

    def _venv_definition_info(self, module_name: str, class_name: Optional[str], entity_name: str, symbol_kind: str) -> Optional[Dict[str, Any]]:
        """Attempt third-party/venv informational stub as a last resort after repo and stdlib checks."""
        try:
            # Avoid stdlib here
            if self._is_stdlib_module(module_name):
                return None
            mod = importlib.import_module(module_name)
            origin = getattr(getattr(mod, '__spec__', None), 'origin', None)
            if not origin or ('site-packages' not in origin and 'dist-packages' not in origin):
                return None
            target_obj = None
            if symbol_kind == 'class':
                target_name = class_name or entity_name
                target_obj = getattr(mod, target_name, None)
            elif symbol_kind == 'function':
                if class_name:
                    cls = getattr(mod, class_name, None)
                    target_obj = getattr(cls, entity_name, None) if cls is not None else None
                else:
                    target_obj = getattr(mod, entity_name, None)
            sig = None
            try:
                sig = str(inspect.signature(target_obj)) if target_obj else None
            except Exception:
                sig = None
            doc = None
            try:
                doc = inspect.getdoc(target_obj) if target_obj else None
            except Exception:
                doc = None
            summary = 'Third-party library symbol; use known behavior. '
            if sig:
                summary += f"Signature: {sig}. "
            if doc:
                summary += f"Doc: {doc[:800]}"
            return {
                'name': entity_name if symbol_kind == 'function' else (class_name or entity_name),
                'context_name_requested': f"{module_name}.{class_name + '.' if class_name else ''}{entity_name}",
                'file_path': f"SITE-PACKAGES:{module_name}",
                'source': summary
            }
        except Exception:
            return None
    
    def _get_script(self, file: pathlib.Path) -> jedi.Script:
        key = str(file)
        if key not in self._script_cache:
            self._script_cache[key] = jedi.Script(path=file, project=self.project)
        return self._script_cache[key]

    def get_enclosing_symbol(self, file_path: pathlib.Path, line_no: int) -> Dict[str, Any] | None:
        """Return the smallest enclosing function or class containing line_no in file_path."""
        try:
            script = self._get_script(file_path)
            names = script.get_names(all_scopes=True, definitions=True)
        except Exception:
            return None

        best = None
        best_span = None
        for name in names:
            if name.type not in ['function', 'class']:
                continue
            try:
                start = name.get_definition_start_position()
                end = name.get_definition_end_position()
            except Exception:
                continue
            if not start or not end:
                continue
            if start[0] <= line_no <= end[0]:
                span = (end[0] - start[0])
                if best_span is None or span < best_span:
                    best_span = span
                    best = {"name": name.name, "type": name.type, "start": start, "end": end}
        return best

    def static_bypass_scan(self, files: List[pathlib.Path]) -> List[Dict[str, Any]]:
        """
        Build regexes from VULN_SPECIFIC_BYPASSES_AND_PROMPTS and scan files for matches.
        Returns a list of hits with vuln_type, file_path, line_no, line_text, pattern, enclosing symbol.
        """
        compiled: Dict[str, List[Dict[str, Any]]] = {}
        for vuln_type, cfg in VULN_SPECIFIC_BYPASSES_AND_PROMPTS.items():
            patterns = []
            for ex in cfg.get('bypasses', []):
                if not ex:
                    continue
                try:
                    pat = re.compile(re.escape(ex), re.IGNORECASE)
                except re.error:
                    continue
                patterns.append({"example": ex, "pattern": pat})
            compiled[vuln_type] = patterns

        results: List[Dict[str, Any]] = []
        for file in files:
            try:
                with file.open(encoding='utf-8') as f:
                    for idx, line in enumerate(f, start=1):
                        for vuln_type, plist in compiled.items():
                            for entry in plist:
                                if entry["pattern"].search(line):
                                    encl = self.get_enclosing_symbol(file, idx)
                                    results.append({
                                        "vuln_type": vuln_type,
                                        "file_path": str(file),
                                        "line_no": idx,
                                        "line_text": line.rstrip('\n'),
                                        "pattern": entry["example"],
                                        "enclosing": encl,
                                    })
            except (UnicodeDecodeError, OSError):
                continue
        return results

    def find_callers(self, symbol_name: str, files: List[pathlib.Path]) -> List[Dict[str, Any]]:
        """
        Naive cross-repo caller search for a function/method name.
        Returns list with file_path, line_no, line_text, kind (direct|method).
        """
        direct = re.compile(r"\b" + re.escape(symbol_name) + r"\s*\(")
        method = re.compile(r"\." + re.escape(symbol_name) + r"\s*\(")
        results: List[Dict[str, Any]] = []
        for file in files:
            try:
                with file.open(encoding='utf-8') as f:
                    for idx, line in enumerate(f, start=1):
                        if direct.search(line):
                            results.append({"file_path": str(file), "line_no": idx, "line_text": line.rstrip('\n'), "kind": "direct"})
                        elif method.search(line):
                            results.append({"file_path": str(file), "line_no": idx, "line_text": line.rstrip('\n'), "kind": "method"})
            except (UnicodeDecodeError, OSError):
                continue
        return results
    def file_search(self, symbol_name: str, scripts: List) -> Dict[str, Any]:
        # Analyze matching files with Jedi
        for script in scripts:

            # Search for the symbol in the script
            res = script.search(symbol_name)

            for name in res:

                if self._should_exclude(str(name.module_path)):
                    continue

                # Statements
                if name.type == 'statement':
                    if symbol_name in name.description:
                        match = self._create_match_obj(name, symbol_name)
                        return match

                # Functions and classes - MOST COMMON
                # Odd thing, in gpt_academic when searching get_conf, inferred object is functools._lru_cache_wrapper?
                elif name.type in ['function', 'class']:
                    if symbol_name == name.name or symbol_name.endswith(f".{name.name}") or symbol_name in name.description:
                        inferred = name.infer()
                        for inf in inferred:
                            match = self._create_match_obj(inf, symbol_name)
                            return match
                
                # Instances
                elif name.type == 'instance':
                    inferred = name.infer()
                    for inf in inferred:
                        # Class or function instance
                        if inf.type in ['class', 'function']:
                            match = self._create_match_obj(inf, symbol_name)
                            return match
                        # Meaning it's probably an instance variable
                        elif inf.type == 'instance':
                            go = name.goto()
                            if go:
                                g = go[0]
                                match = self._create_match_obj(g, symbol_name)
                                return match
                
                # Modules
                # Handle edge case for modules like "app" in "from api.apps import app"
                elif name.type == 'module':
                    if name.name == symbol_name:
                        match = self._create_match_obj(name, symbol_name)
                        if 'import ' in match['source']:
                            loc = name.goto()
                            if loc:
                                match = self._create_match_obj(loc[0], symbol_name)
                        return match

        return

    def project_search(self, symbol_name: str) -> List[Dict[str, Any]]:
        """
        Searches for a symbol in the project using jedi.Project.search and returns a match if found.
        Handles:
            - exact match
            - edge case #2: var = ClassName(); var.method()
        """
        res = list(self.project.search(symbol_name))

        for name in res:
            # Statements
            if name.type == 'statement':
                if symbol_name in name.description:
                    match = self._create_match_obj(name, symbol_name)
                    return match

            # Functions and classes - MOST COMMON
            elif name.type in ['function', 'class']:
                if symbol_name == name.name or symbol_name.endswith(f".{name.name}") or symbol_name in name.description:
                    inferred = name.infer()
                    for inf in inferred:
                        match = self._create_match_obj(inf, symbol_name)
                        return match
            
            # Instances
            elif name.type == 'instance':
                inferred = name.infer()
                for inf in inferred:
                    if inf.type in ['instance', 'class', 'function']:
                        match = self._create_match_obj(inf, symbol_name)
                        return match
            
            # Modules
            # Handle edge case for modules like "app" in "from api.apps import app"
            elif name.type == 'module':
                if name.name == symbol_name:
                    match = self._create_match_obj(name, symbol_name)
                    if 'import ' in match['source']:
                        loc = name.goto()
                        if loc:
                            match = self._create_match_obj(loc[0], symbol_name)
                    return match
        
        return

    def all_names_search(self, symbol_name: str, symbol_parts: List, scripts: List[jedi.Script], code_line: str) -> Dict[str, Any]:
        """
        Searches for all names in the project using jedi.Script.get_names and returns a match if found.
        Handles method calls on variables.
        """
        for script in scripts:
            names = script.get_names(all_scopes=True, definitions=True, references=True)
            for name in names:
                if name.type in ['function', 'class', 'instance']:
                    if name.full_name:
                        if name.full_name.endswith(symbol_name):
                            inferred = name.infer()
                            for inf in inferred:
                                match = self._create_match_obj(inf, symbol_name)
                                return match
                    else:
                        if name.name == symbol_parts[-1]:
                            inferred = name.infer()
                            for inf in inferred:
                                match = self._create_match_obj(inf, symbol_name)
                                return match
                    
        # No anchor-based fallback anymore

        print('No matches found for symbol:', symbol_name)
        return
    
    def _is_exact_match(self, name: Name, parts: List[str]) -> bool:
        if len(parts) == 1:
            # For single-part symbols, match the name or the full name if available
            return name.name == parts[0] or (name.full_name and name.full_name.endswith(parts[0]))
        else:
            # For multi-part symbols, ensure all parts match if full_name is available
            if not name.full_name:
                return False
            name_parts = name.full_name.split('.')
            return name_parts[-len(parts):] == parts
    
    # Helper function to check if a name should be excluded
    def _should_exclude(self, module_path: str) -> bool:
        module_path = module_path.lower().replace('\\', '/')
        return any(x in module_path for x in self.ignore)
    
    # Function to search for a string in a file
    def _search_string_in_file(self, file_path, string):
        """
        Replace all spaces and newlines in the file and the string to be searched for and check if the string is in the file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            # Remove spaces and newlines
            return string.replace(' ', '').replace('\n', '').replace('"', "'").replace('\r', '').replace('\t', '') in \
                   file.read().replace(' ', '').replace('\n', '').replace('"', "'").replace('\r', '').replace('\t', '')
    
    def _get_definition_source(self, file_path: pathlib.Path, start, end):
        with file_path.open(encoding='utf-8') as f:
            lines = f.readlines()

            if not start and not end:
                s = ''.join(lines)
                return s

            definition = lines[ start[0]-1:end[0] ]
            end_len_diff = len(definition[-1]) - end[1]

            s = ''.join(definition)[start[1]:-end_len_diff] if end_len_diff > 0 else ''.join(definition)[start[1]:]

            if not s:
                return 'None'

            return s
    
    def _create_match_obj(self, name: Name, symbol_name: str) -> Dict[str, Any]:
        module_path = str(name.module_path)
        if module_path == 'None' or 'site-packages' in module_path or 'dist-packages' in module_path or '/third_party/' in module_path:
            # Third-party or venv library reference; provide guidance instead of source dump
            source = f'Third party library (possibly from a virtual environment). Use what you already know about {name.full_name} to understand the code.'
        else:
            start = name.get_definition_start_position()
            end = name.get_definition_end_position()
            source = self._get_definition_source(name.module_path,
                                                start,
                                                end
                                                )

        return {'name': name.name,
                'context_name_requested': symbol_name,
                'file_path': str(name.module_path),
                'source':source}

    def _detect_venv_python(self) -> Optional[str]:
        """Detect a virtual environment Python executable under the target repo if present."""
        candidates = ['.venv', 'venv', 'env', 'ENV', 'virtualenv']
        for c in candidates:
            v = (self.repo_path / c)
            if not v.is_dir():
                continue
            # Unix-like
            for exe in ('python3', 'python'):
                p = v / 'bin' / exe
                if p.exists():
                    return str(p)
            # Windows
            p = v / 'Scripts' / 'python.exe'
            if p.exists():
                return str(p)
        return None