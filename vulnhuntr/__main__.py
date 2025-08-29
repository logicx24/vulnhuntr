import json
import time
import re
import argparse
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue as queue_mod
import structlog
from vulnhuntr.symbol_finder import SymbolExtractor
from vulnhuntr.LLMs import Claude, ChatGPT, Ollama, OpenRouter
from vulnhuntr.prompts import *
from rich import print
from typing import List, Generator, Literal
from enum import Enum
from pathlib import Path
from pydantic_xml import BaseXmlModel, element
from pydantic import BaseModel, Field
import dotenv
import os

dotenv.load_dotenv()

structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.WriteLoggerFactory(
        file=Path('vulnhuntr').with_suffix(".log").open("wt")
    )
)

import faulthandler
faulthandler.enable()

log = structlog.get_logger("vulnhuntr")

class VulnType(str, Enum):
    LFI = "LFI"
    RCE = "RCE"
    SSRF = "SSRF"
    AFO = "AFO"
    SQLI = "SQLI"
    XSS = "XSS"
    IDOR = "IDOR"

class ContextCode(BaseModel):
    module_name: str = Field(description="Dotted module path where the entity resides (e.g., engine.neo.hi)")
    class_name: str | None = Field(default=None, description="Class name for methods or when requesting a class; omit for module-level functions")
    entity_name: str = Field(description="For symbol_kind=class, the class name. For symbol_kind=function, the function or method name")
    symbol_kind: Literal["function", "class"] = Field(description="Type of the symbol: function or class")
    request_type: Literal["REQUEST_DEFINITION", "REQUEST_CALLERS"] = Field(description="REQUEST_DEFINITION to fetch the symbol's definition; REQUEST_CALLERS to fetch its callers")
    reason: str = Field(description="Brief reason why this context is needed for analysis")

class Response(BaseModel):
    analysis: str = Field(description="Your final analysis. Output in plaintext with no line breaks.", min_length=64)
    poc: str = Field(description="Proof-of-concept exploit, if applicable.")
    confidence_score: int = Field(description="0-10, where 0 is no confidence and 10 is absolute certainty because you have the entire user input to server output code path.")
    vulnerability_types: List[VulnType] = Field(description="The types of identified vulnerabilities")
    context_code: List[ContextCode] = Field(description="List of context code items requested for analysis, one function or class name per item. No standard library or third-party package code.")
    vulnerability_present: bool = Field(description="True only if you are reasonably certain the vulnerability exists in the analyzed code path. False if inconclusive or disproven.")

class ResponseInitial(BaseModel):
    analysis: str = Field(description="Your final analysis. Output in plaintext with no line breaks.", min_length=64)
    confidence_score: int = Field(description="0-10, where 0 is no confidence and 10 is absolute certainty because you have the entire user input to server output code path.")
    vulnerability_types: List[VulnType] = Field(description="The types of identified vulnerabilities")

class ReadmeContent(BaseXmlModel, tag="readme_content"):
    content: str

class ReadmeSummary(BaseXmlModel, tag="readme_summary"):
    readme_summary: str

class Instructions(BaseXmlModel, tag="instructions"):
    instructions: str

class ResponseFormat(BaseXmlModel, tag="response_format"):
    response_format: str

class AnalysisApproach(BaseXmlModel, tag="analysis_approach"):
    analysis_approach: str

class Guidelines(BaseXmlModel, tag="guidelines"):
    guidelines: str

class FileCode(BaseXmlModel, tag="file_code"):
    file_path: str = element()
    file_source: str = element()

class PreviousAnalysis(BaseXmlModel, tag="previous_analysis"):
    previous_analysis: str

class InitialAnalysis(BaseXmlModel, tag="initial_analysis"):
    initial_analysis: str

class ExampleBypasses(BaseXmlModel, tag="example_bypasses"):
    example_bypasses: str

class CodeDefinition(BaseXmlModel, tag="code"):
    name: str = element()
    context_name_requested: str = element()
    file_path: str = element()
    source: str = element()

class CodeDefinitions(BaseXmlModel, tag="context_code"):
    definitions: List[CodeDefinition] = []

class FileFilterResult(BaseModel):
    keep: List[str] = Field(description="Absolute file paths to keep for analysis after excluding tests and generated code")
    drop: List[str] = Field(default_factory=list, description="Absolute file paths excluded as tests or generated code")

class TaskType(str, Enum):
    INITIAL = "initial"
    SECONDARY = "secondary"

class Task(BaseModel):
    type: TaskType
    file_path: str
    vuln_type: str | None = None
    iteration_count: int = 0
    previous_analysis: str = ""
    all_previous_analyses: List[str] = Field(default_factory=list)
    stored_code_definitions: dict = {}
    same_context: bool = False
    previous_context_amount: int = 0

    
def _is_proto_or_generated(path: str) -> bool:
    pl = path.lower()
    name = Path(path).name.lower()
    return (
        name.endswith('.proto') or
        name.endswith('_pb2.py') or
        name.endswith('_pb2_grpc.py') or
        '/generated/' in pl or
        '\\generated\\' in pl
    )

class FileWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.q: queue_mod.Queue[dict] = queue_mod.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def enqueue(self, record: dict) -> None:
        self.q.put(record)

    def _run(self) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            while not self.stop_event.is_set() or not self.q.empty():
                try:
                    rec = self.q.get(timeout=0.1)
                except Exception:
                    continue
                f.write(json.dumps(rec) + "\n")
                self.q.task_done()

    def stop(self) -> None:
        self.stop_event.set()
        self.q.join()
        self.thread.join(timeout=2)

    def flush(self) -> None:
        try:
            self.q.join()
        except Exception:
            pass

def process_initial_task(task: Task,
                         primary_writer: 'FileWriter',
                         llm: Claude | ChatGPT | Ollama | OpenRouter,
                         primary_map: dict[str, dict]) -> list[Task]:
    py_f = Path(task.file_path)
    try:
        with py_f.open(encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        log.error("Initial read failed", file=str(py_f), exc_info=e)
        return []
    if not content:
        return []
    if task.file_path in primary_map:
        log.info("Skipping already analyzed file", file=task.file_path)
        return []
    user_prompt = (
        FileCode(file_path=str(py_f), file_source=content).to_xml() + b'\n' +
        Instructions(instructions=INITIAL_ANALYSIS_PROMPT_TEMPLATE).to_xml() + b'\n' +
        AnalysisApproach(analysis_approach=ANALYSIS_APPROACH_TEMPLATE_INITIAL).to_xml() + b'\n' +
        PreviousAnalysis(previous_analysis='').to_xml() + b'\n' +
        Guidelines(guidelines=GUIDELINES_TEMPLATE_INITIAL).to_xml() + b'\n' +
        ResponseFormat(response_format=json.dumps(ResponseInitial.model_json_schema(), indent=4)).to_xml()
    ).decode()
    log.info("Initial analysis prompt", user_prompt=user_prompt)
    report: ResponseInitial = llm.chat(user_prompt, response_model=ResponseInitial)
    log.info("Initial analysis complete", report=report.model_dump())
    rec = {"file_path": str(py_f), **report.model_dump()}
    try:
        primary_writer.enqueue(rec)
    except Exception as e:
        log.error("Primary enqueue failed", exc_info=e)
    primary_map[str(py_f)] = rec
    print_readable(str(py_f), report)
    # build secondary tasks
    new_secondaries: list[Task] = []
    if report.confidence_score > 0 and len(report.vulnerability_types):
        for vt in report.vulnerability_types:
            if not _is_proto_or_generated(str(py_f)):
                new_secondaries.append(Task(
                    type=TaskType.SECONDARY,
                    file_path=str(py_f),
                    vuln_type=vt,
                    iteration_count=0,
                    previous_analysis=report.analysis,
                ))
    return new_secondaries

def process_secondary_task(task: Task,
                           secondary_writer: 'FileWriter',
                           final_writer: 'FileWriter',
                           llm: Claude | ChatGPT | Ollama | OpenRouter,
                           primary_map: dict[str, dict],
                           secondary_map: dict[str, dict[str, list]],
                           code_extractor: SymbolExtractor,
                           files: List[Path]) -> Task | None:
    py_f = Path(task.file_path)
    if not task.vuln_type:
        return None
    try:
        with py_f.open(encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        log.error("Secondary read failed", file=str(py_f), exc_info=e)
        return None
    if not content:
        return None
    i = task.iteration_count or 0
    stored_defs = task.stored_code_definitions or {}
    previous_context_amount = len(stored_defs)
    definitions = CodeDefinitions(definitions=list(stored_defs.values()))
    # Build vuln-specific prompt including the initial analysis text as previous_analysis
    initial_text = primary_map.get(str(py_f), {}).get('analysis', '')
    vuln_specific_user_prompt = (
        FileCode(file_path=str(py_f), file_source=content).to_xml() + b'\n' +
        definitions.to_xml() + b'\n' +
        ExampleBypasses(example_bypasses='\n'.join(VULN_SPECIFIC_BYPASSES_AND_PROMPTS[task.vuln_type]['bypasses'])).to_xml() + b'\n' +
        Instructions(instructions=VULN_SPECIFIC_BYPASSES_AND_PROMPTS[task.vuln_type]['prompt']).to_xml() + b'\n' +
        AnalysisApproach(analysis_approach=ANALYSIS_APPROACH_TEMPLATE).to_xml() + b'\n' +
        PreviousAnalysis(previous_analysis=initial_text).to_xml() + b'\n' +
        Guidelines(guidelines=GUIDELINES_TEMPLATE).to_xml() + b'\n' +
        ResponseFormat(response_format=json.dumps(Response.model_json_schema(), indent=4)).to_xml()
    ).decode()
    log.info("Secondary analysis for file and vulnerability type", file=py_f, vuln_type=task.vuln_type)
    secondary_analysis_report: Response = llm.chat(vuln_specific_user_prompt, response_model=Response)
    log.info("Secondary analysis complete", secondary_analysis_report=secondary_analysis_report.model_dump())
    rec2 = {"file_path": str(py_f), "vuln_type": task.vuln_type, **secondary_analysis_report.model_dump()}
    try:
        secondary_writer.enqueue(rec2)
    except Exception as e:
        log.error("Secondary enqueue failed", exc_info=e)
    secondary_map.setdefault(str(py_f), {}).setdefault(task.vuln_type, []).append(rec2)
    if secondary_analysis_report and secondary_analysis_report.analysis and len(secondary_analysis_report.analysis):
        log.debug("Secondary analysis text length", length=len(secondary_analysis_report.analysis))
    if len(secondary_analysis_report.context_code):
        log.debug("Secondary context_code count", count=len(secondary_analysis_report.context_code))
    # Update stored definitions with newly requested context
    added = 0
    for context_item in secondary_analysis_report.context_code:
        req_type = context_item.request_type
        module_name = context_item.module_name
        class_name = context_item.class_name
        entity_name = context_item.entity_name
        key = f"{module_name}:{class_name or ''}:{entity_name}:{context_item.symbol_kind}"
        if req_type == 'REQUEST_DEFINITION':
            if key not in stored_defs:
                match = code_extractor.extract(module_name, class_name, entity_name, context_item.symbol_kind, files)
                if match:
                    stored_defs[key] = match
                    added += 1
        elif req_type == 'REQUEST_CALLERS':
            callers = code_extractor.find_callers(entity_name, list(files))
            for caller in callers:
                _ = caller  # currently not enqueuing new initials here
    # Determine next iteration or finalize
    next_i = i + 1
    current_no_progress = (added == 0)
    finalize_now = (next_i >= 7) or (task.same_context and current_no_progress)
    if finalize_now:
        if secondary_analysis_report.vulnerability_present:
            final_rec = {"file_path": str(py_f), "vuln_type": task.vuln_type, **secondary_analysis_report.model_dump()}
            try:
                final_writer.enqueue(final_rec)
            except Exception as e:
                log.error("Final enqueue failed", exc_info=e)
            print_readable(str(py_f), secondary_analysis_report)
        else:
            log.info("Dropping final record without confirmed vulnerability", file=str(py_f), vuln_type=task.vuln_type)
        return None
    else:
        return Task(
            type=TaskType.SECONDARY,
            file_path=str(py_f),
            vuln_type=task.vuln_type,
            iteration_count=next_i,
            previous_analysis=initial_text,
            stored_code_definitions=stored_defs,
            same_context=(task.same_context or current_no_progress),
            previous_context_amount=len(stored_defs),
        )

class RepoOps:
    def __init__(self, repo_path: Path | str ) -> None:
        self.repo_path = Path(repo_path)
        self.to_exclude = {'/setup.py', '/test', '/example', '/docs', '/site-packages', '.venv', 'virtualenv', '/dist'}
        self.file_names_to_exclude = ['test_', 'conftest', '_test.py']

        patterns = [
            #Async
            r'async\sdef\s\w+\(.*?request',

            # Gradio
            r'gr.Interface\(.*?\)',
            r'gr.Interface\.launch\(.*?\)',

            # Flask
            r'@app\.route\(.*?\)',
            r'@blueprint\.route\(.*?\)',
            r'class\s+\w+\(MethodView\):',
            r'@(?:app|blueprint)\.add_url_rule\(.*?\)',

            # FastAPI
            r'@app\.(?:get|post|put|delete|patch|options|head|trace)\(.*?\)',
            r'@router\.(?:get|post|put|delete|patch|options|head|trace)\(.*?\)',

            # Django
            r'url\(.*?\)', #Too broad?
            r're_path\(.*?\)',
            r'@channel_layer\.group_add',
            r'@database_sync_to_async',

            # Pyramid
            r'@view_config\(.*?\)',

            # Bottle
            r'@(?:route|get|post|put|delete|patch)\(.*?\)',

            # Tornado
            r'class\s+\w+\((?:RequestHandler|WebSocketHandler)\):',
            r'@tornado\.gen\.coroutine',
            r'@tornado\.web\.asynchronous',

            #WebSockets
            r'websockets\.serve\(.*?\)',
            r'@websocket\.(?:route|get|post|put|delete|patch|head|options)\(.*?\)',

            # aiohttp
            r'app\.router\.add_(?:get|post|put|delete|patch|head|options)\(.*?\)',
            r'@routes\.(?:get|post|put|delete|patch|head|options)\(.*?\)',

            # Sanic
            r'@app\.(?:route|get|post|put|delete|patch|head|options)\(.*?\)',
            r'@blueprint\.(?:route|get|post|put|delete|patch|head|options)\(.*?\)',

            # Falcon
            r'app\.add_route\(.*?\)',

            # CherryPy
            r'@cherrypy\.expose',

            # web2py
            r'def\s+\w+\(\):\s*return\s+dict\(',

            # Quart (ASGI version of Flask)
            r'@app\.route\(.*?\)',
            r'@blueprint\.route\(.*?\)',

            # Starlette (which FastAPI is based on)
            r'@app\.route\(.*?\)',
            r'Route\(.*?\)',

            # Responder
            r'@api\.route\(.*?\)',

            # Hug
            r'@hug\.(?:get|post|put|delete|patch|options|head)\(.*?\)',

            # Dash (for analytical web applications)
            r'@app\.callback\(.*?\)',

            # GraphQL entry points
            r'class\s+\w+\(graphene\.ObjectType\):',
            r'@strawberry\.type',

            # Generic decorators that might indicate custom routing
            r'@route\(.*?\)',
            r'@endpoint\(.*?\)',
            r'@api\.\w+\(.*?\)',

            # AWS Lambda handlers (which could be used with API Gateway)
            r'def\s+lambda_handler\(event,\s*context\):',
            r'def\s+handler\(event,\s*context\):',

            # Azure Functions
            r'def\s+\w+\(req:\s*func\.HttpRequest\)\s*->',

            # Google Cloud Functions
            r'def\s+\w+\(request\):'

            # Server startup code
            r'app\.run\(.*?\)',
            r'serve\(app,.*?\)',
            r'uvicorn\.run\(.*?\)',
            r'application\.listen\(.*?\)',
            r'run_server\(.*?\)',
            r'server\.start\(.*?\)',
            r'app\.listen\(.*?\)',
            r'httpd\.serve_forever\(.*?\)',
            r'tornado\.ioloop\.IOLoop\.current\(\)\.start\(\)',
            r'asyncio\.run\(.*?\.serve\(.*?\)\)',
            r'web\.run_app\(.*?\)',
            r'WSGIServer\(.*?\)\.serve_forever\(\)',
            r'make_server\(.*?\)\.serve_forever\(\)',
            r'cherrypy\.quickstart\(.*?\)',
            r'execute_from_command_line\(.*?\)',  # Django's manage.py
            r'gunicorn\.app\.wsgiapp\.run\(\)',
            r'waitress\.serve\(.*?\)',
            r'hypercorn\.run\(.*?\)',
            r'daphne\.run\(.*?\)',
            r'werkzeug\.serving\.run_simple\(.*?\)',
            r'gevent\.pywsgi\.WSGIServer\(.*?\)\.serve_forever\(\)',
            r'grpc\.server\(.*?\)\.start\(\)',
            r'app\.start_server\(.*?\)',  # Sanic
            r'Server\(.*?\)\.run\(\)',    # Bottle
        ]

        # Compile the patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in patterns]

    def get_readme_content(self) -> str:
        # Use glob to find README.md or README.rst in a case-insensitive manner in the root directory
        prioritized_patterns = ["[Rr][Ee][Aa][Dd][Mm][Ee].[Mm][Dd]", "[Rr][Ee][Aa][Dd][Mm][Ee].[Rr][Ss][Tt]"]
        
        # First, look for README.md or README.rst in the root directory with case insensitivity
        for pattern in prioritized_patterns:
            for readme in self.repo_path.glob(pattern):
                with readme.open(encoding='utf-8') as f:
                    return f.read()
                
        # If no README.md or README.rst is found, look for any README file with supported extensions
        for readme in self.repo_path.glob("[Rr][Ee][Aa][Dd][Mm][Ee]*.[Mm][DdRrSsTt]"):
            with readme.open(encoding='utf-8') as f:
                return f.read()
        
        return

    def get_relevant_py_files(self) -> Generator[Path, None, None]:
        """Gets all Python files in a repo minus the ones in the exclude list (test, example, doc, docs)"""
        files = []
        for f in self.repo_path.rglob("*.py"):
            # Convert the path to a string with forward slashes
            f_str = str(f).replace('\\', '/')
            
            # Lowercase the string for case-insensitive matching
            f_str = f_str.lower()

            # Check if any exclusion pattern matches a substring of the full path
            if any(exclude in f_str for exclude in self.to_exclude):
                continue

            # Check if the file name should be excluded
            if any(fn in f.name for fn in self.file_names_to_exclude):
                continue
            
            files.append(f)

        return files

    def get_network_related_files(self, files: List) -> Generator[Path, None, None]:
        for py_f in files:
            with py_f.open(encoding='utf-8') as f:
                content = f.read()
            if any(re.search(pattern, content) for pattern in self.compiled_patterns):
                yield py_f

    def get_files_to_analyze(self, analyze_path: Path | None = None) -> List[Path]:
        path_to_analyze = analyze_path or self.repo_path
        if path_to_analyze.is_file():
            return [ path_to_analyze ]
        elif path_to_analyze.is_dir():
            return path_to_analyze.rglob('*.py')
        else:
            raise FileNotFoundError(f"Specified analyze path does not exist: {path_to_analyze}")

def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    """
    https://github.com/anthropics/anthropic-cookbook/blob/main/misc/how_to_enable_json_mode.ipynb
    """
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list

def initialize_llm(llm_arg: str, system_prompt: str = "", model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> Claude | ChatGPT | Ollama | OpenRouter:
    llm_arg = llm_arg.lower()
    if llm_arg == 'claude':
        use_model = model or "claude-3-5-sonnet-latest"
        use_base = base_url or "https://api.anthropic.com"
        llm = Claude(use_model, use_base, system_prompt, api_key=api_key)
    elif llm_arg == 'gpt':
        use_model = model or "chatgpt-4o-latest"
        use_base = base_url or "https://api.openai.com/v1"
        llm = ChatGPT(use_model, use_base, system_prompt, api_key=api_key)
    elif llm_arg == 'ollama':
        use_model = model or "llama3"
        use_base = base_url or "http://127.0.0.1:11434/api/generate"
        llm = Ollama(use_model, use_base, system_prompt, api_key=api_key)
    elif llm_arg == 'openrouter':
        # OpenRouter uses the OpenAI-compatible client with a different base_url and API key
        use_model = model or "openai/gpt-5-nano"
        use_base = "https://openrouter.ai/api/v1"
        llm = OpenRouter(use_model, use_base, system_prompt, api_key=api_key)
    else:
        raise ValueError(f"Invalid LLM argument: {llm_arg}\nValid options are: claude, gpt, ollama, openrouter")
    return llm

def print_readable(filename: str, report: Response | ResponseInitial) -> None:
    print("=" * 80)
    print(f"File: {filename} | RESPONSE TYPE: {type(report).__name__}")
    print("=" * 80)
    
    for attr, value in vars(report).items():
        print(f"{attr}:")
        if isinstance(value, str):
            # For multiline strings, add indentation
            lines = value.split('\n')
            for line in lines:
                print(f"  {line}")
        elif isinstance(value, list):
            # For lists, print each item on a new line
            for item in value:
                print(f"  - {item}")
        else:
            # For other types, just print the value
            print(f"  {value}")
        print('-' * 40)
        print()  # Add an empty line between attributes

def run():
    parser = argparse.ArgumentParser(description='Analyze a GitHub project for vulnerabilities. Export your ANTHROPIC_API_KEY/OPENAI_API_KEY or OPENROUTER_API_KEY before running.')
    parser.add_argument('-r', '--root', type=str, required=True, help='Path to the root directory of the project')
    parser.add_argument('-a', '--analyze', type=str, help='Specific path or file within the project to analyze')
    parser.add_argument('-l', '--llm', type=str, choices=['claude', 'gpt', 'ollama', 'openrouter'], default='claude', help='LLM client to use (default: claude)')
    parser.add_argument('--model', type=str, help='Model name for the selected LLM provider')
    parser.add_argument('--base-url', type=str, help='Base URL for the selected LLM provider')
    parser.add_argument('--api-key', type=str, help='API key for the selected LLM provider')
    parser.add_argument('--restart', action='store_true', help='Start fresh and ignore cached analyses')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='Increase output verbosity (-v for INFO, -vv for DEBUG)')
    parser.add_argument('--parallelism', type=int, default=96, help='Max number of parallel tasks to run (default: 32)')
    args = parser.parse_args()

    repo = RepoOps(args.root)
    code_extractor = SymbolExtractor(args.root)
    # Get repo files that don't include stuff like tests and documentation
    files = repo.get_relevant_py_files()

    # Initialize persistence under results/<target_dir>-<timestamp>/
    target_dir_name = Path(args.root).resolve().name
    timestamp = str(int(time.time()))
    results_root = Path('results')
    run_dir = results_root / f"{target_dir_name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")
    primary_cache_path = run_dir / 'vulnhuntr_primary.jsonl'
    secondary_cache_path = run_dir / 'vulnhuntr_secondary.jsonl'
    primary_map: dict[str, dict] = {}
    secondary_map: dict[str, dict[str, list]] = {}
    if args.restart:
        if primary_cache_path.exists():
            primary_cache_path.unlink()
        if secondary_cache_path.exists():
            secondary_cache_path.unlink()
    else:
        if primary_cache_path.exists():
            with primary_cache_path.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        primary_map[rec['file_path']] = rec
                    except Exception:
                        continue
        if secondary_cache_path.exists():
            with secondary_cache_path.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        fp = rec.get('file_path')
                        vt = rec.get('vuln_type')
                        if not fp or not vt:
                            continue
                        secondary_map.setdefault(fp, {}).setdefault(vt, []).append(rec)
                    except Exception:
                        continue

    # User specified --analyze flag
    if args.analyze:
        # Determine the path to analyze
        analyze_path = Path(args.analyze)

        # If the path is absolute, use it as is, otherwise join it with the root path so user can specify relative paths
        if analyze_path.is_absolute():
            path_to_analyze = analyze_path
        else:
            path_to_analyze = Path(args.root) / analyze_path

        # Always build a deque of files
        files_to_analyze = deque(str(p) for p in repo.get_files_to_analyze(path_to_analyze))

    # Analyze the entire project: iterate every relevant file, keep filters for tests/generated
    else:
        files_to_analyze = deque(str(f) for f in files)
    
    llm = initialize_llm(args.llm, model=args.model, base_url=args.base_url, api_key=args.api_key)

    readme_content = repo.get_readme_content()
    if readme_content:
        log.info("Summarizing project README")
        readme_prompt = (
            ReadmeContent(content=readme_content).to_xml() + b'\n' +
            Instructions(instructions=README_SUMMARY_PROMPT_TEMPLATE).to_xml()
            ).decode()
        summary = llm.chat(readme_prompt)
        summary = extract_between_tags("summary", summary)[0]
        log.info("README summary complete", summary=summary)
    else:
        log.warning("No README summary found")
        summary = ''
    
    # Initialize the system prompt with the README summary
    system_prompt = (Instructions(instructions=SYS_PROMPT_TEMPLATE).to_xml() + b'\n' +
                ReadmeSummary(readme_summary=summary).to_xml()
                ).decode()
    
    llm = initialize_llm(args.llm, system_prompt, model=args.model, base_url=args.base_url, api_key=args.api_key)

    # Let the LLM filter out test and generated files from the initial queue
    # Filter out test and generated files using string patterns
    exclude_patterns = [
        'tests/', 'test_', 'test', '_test.py', 'conftest.py', '__pycache__/', 
        'build/', 'dist/', 'generated/', 'site-packages/', 'migrations/', 
        '_pb2.py', '_pb2_grpc.py', "venv", "virtualenv", ".git/", 
    ]
    
    filtered_files = []
    for file_path in files_to_analyze:
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern in file_path:
                should_exclude = True
                break
        if not should_exclude:
            filtered_files.append(file_path)
    
    files_to_analyze = deque(filtered_files)

    # Build phase-specific task queues
    initial_queue: deque[Task] = deque(Task(type=TaskType.INITIAL, file_path=p) for p in files_to_analyze)
    secondary_queue: deque[Task] = deque()
    final_findings_path = run_dir / 'vulnhuntr_final_findings.jsonl'

    # Initialize async file writers
    primary_writer = FileWriter(primary_cache_path)
    secondary_writer = FileWriter(secondary_cache_path)
    final_writer = FileWriter(final_findings_path)
    primary_writer.start()
    secondary_writer.start()
    final_writer.start()

    print(f"Initial files to analyze: {files_to_analyze}")

    # Phase 1 parallel execution
    with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as pool:
        future_to_task = {
            pool.submit(
                process_initial_task,
                t,
                primary_writer,
                llm,
                primary_map,
            ): t for t in list(initial_queue)
        }
        for fut in as_completed(list(future_to_task.keys())):
            task_obj = future_to_task[fut]
            try:
                res = fut.result()
                if res:
                    secondary_queue.extend(res)
            except Exception as e:
                log.error("Initial task failed", exc_info=e, file=task_obj.file_path, task_type=str(task_obj.type))

    # Phase 2 parallel execution with waves
    while secondary_queue:
        wave = list(secondary_queue)
        secondary_queue.clear()
        with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as pool:
            future_to_task = {
                pool.submit(
                    process_secondary_task,
                    t,
                    secondary_writer,
                    final_writer,
                    llm,
                    primary_map,
                    secondary_map,
                    code_extractor,
                    files,
                ): t for t in wave
            }
            for fut in as_completed(list(future_to_task.keys())):
                t = future_to_task[fut]
                try:
                    nxt = fut.result()
                    if nxt:
                        secondary_queue.append(nxt)
                except Exception as e:
                    log.error("Secondary task failed", exc_info=e, file=t.file_path, vuln_type=t.vuln_type, iteration=t.iteration_count)

    # Flush and stop writers
    try:
        primary_writer.flush(); secondary_writer.flush(); final_writer.flush()
    finally:
        primary_writer.stop(); secondary_writer.stop(); final_writer.stop()

if __name__ == '__main__':
    run()
