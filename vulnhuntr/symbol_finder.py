import jedi
import pathlib
import re
from typing import List, Dict, Any
from jedi.api.classes import Name
from vulnhuntr.prompts import VULN_SPECIFIC_BYPASSES_AND_PROMPTS


class SymbolExtractor:
    def __init__(self, repo_path: str | pathlib.Path) -> None:
        self.repo_path = pathlib.Path(repo_path)
        self.project = jedi.Project(self.repo_path)
        self.parsed_symbols = None
        self.ignore = ['/test', '_test/', '/docs', '/example']
        self._script_cache: Dict[str, jedi.Script] = {}
    
    def extract(self, symbol_name: str, code_line: str, filtered_files: List) -> Dict:
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

        symbol_parts = symbol_name.split('.')
        matching_files = [file for file in filtered_files if self._search_string_in_file(file, code_line)]
        if len(matching_files) == 0:
            print(f'Code line not found: {code_line}')
            scripts = []
        else:
            scripts = [jedi.Script(path=file, project=self.project) for file in matching_files]

        
        # Search using jedi.Script.search; uses the code_line from bot to grep for string in files
        match = self.file_search(symbol_name, scripts)
        if match:
            return match

        # Search using jedi.Project.search(); finds matches and class instance variables like "var = ClassName(); var.method()"
        match = self.project_search(symbol_name)
        if match:
            return match
        
        # Still no match, so we search using jedi.Script.get_names(); handles method calls on variables
        match = self.all_names_search(symbol_name, symbol_parts, scripts, code_line)

        return match
    
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
                    
        # Edge case: Nothing ever found, so we check for the code_line in the description
        for script in scripts:
            names = script.get_names(all_scopes=True)
            for name in names:
                # All these replacements are the same as we do to the code_line
                cl = code_line.replace(' ', '').replace('\n', '').replace('"', "'").replace('\r', '').replace('\t', '')
                desc = name.description.replace(' ', '').replace('\n', '').replace('"', "'").replace('\r', '').replace('\t', '')
                # check for the code_line in the name.description
                if cl in desc:
                    match = self._create_match_obj(name, symbol_name)
                    return match
                # If no match, check for the code_line in the inferred description
                inferred = name.infer()
                for inf in inferred:
                    idesc = inf.description.replace(' ', '').replace('\n', '').replace('"', "'").replace('\r', '').replace('\t', '')
                    if cl in idesc:
                        match = self._create_match_obj(inf, symbol_name)
                        return match

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
        if '/third_party/' in module_path or module_path == 'None':
            # Extract the third party library name
            source = f'Third party library. Claude, use what you already know about {name.full_name} to understand the code.'
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