import logging
import json
import time
from typing import List, Union, Dict, Any
from pydantic import BaseModel, ValidationError
import anthropic
import os
import openai
import dotenv
import requests

dotenv.load_dotenv()

log = logging.getLogger(__name__)

class LLMError(Exception):
    """Base class for all LLM-related exceptions."""
    pass

class RateLimitError(LLMError):
    pass

class APIConnectionError(LLMError):
    pass

class APIStatusError(LLMError):
    def __init__(self, status_code: int, response: Dict[str, Any]):
        self.status_code = status_code
        self.response = response
        super().__init__(f"Received non-200 status code: {status_code}")

# Base LLM class to handle common functionality
class LLM:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.prev_prompt: Union[str, None] = None
        self.prev_response: Union[str, None] = None
        self.prefill = None

    def _extract_json_object(self, text: str) -> str | None:
        if not text:
            return None
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
        return None

    def _close_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        def _set_closed(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    # Require closed objects
                    node["additionalProperties"] = False
                    # Ensure 'required' includes every key in properties
                    props = node.get("properties")
                    if isinstance(props, dict):
                        node["required"] = list(props.keys())
                    # Recurse into properties
                    if isinstance(props, dict):
                        for v in props.values():
                            _set_closed(v)
                # Recurse common schema containers
                if "items" in node:
                    _set_closed(node["items"])
                for key in ("anyOf", "oneOf", "allOf"):
                    if key in node and isinstance(node[key], list):
                        for v in node[key]:
                            _set_closed(v)
                for defs_key in ("definitions", "$defs"):
                    if defs_key in node and isinstance(node[defs_key], dict):
                        for v in node[defs_key].values():
                            _set_closed(v)
        try:
            _set_closed(schema)
        except Exception:
            pass
        return schema

    def _strip_code_fences(self, text: str) -> str:
        if not text:
            return text
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove the first line (``` or ```json) and the trailing ```
            lines = stripped.splitlines()
            if len(lines) >= 2:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                return "\n".join(lines).strip()
        return text

    def _fallback_response(self, response_model: BaseModel, raw_text: str) -> BaseModel:
        # Provide a minimal valid structure so the pipeline can continue
        try:
            return response_model.model_validate({
                "analysis": (raw_text or ""),
                "poc": "",
                "confidence_score": 0,
                "vulnerability_types": [],
                "context_code": []
            })
        except Exception as e:
            raise LLMError("Validation failed and fallback construction failed") from e

    def _validate_response(self, response_text: str, response_model: BaseModel) -> BaseModel:
        # Multiple strategies to coerce into the schema (balanced extraction, direct, loads, naive slice)
        if self.prefill:
            response_text = self.prefill + response_text
        text = self._strip_code_fences(response_text)
        if not text or not text.strip():
            raise LLMError("Empty response from model")

        # 1) Prefer complete JSON object outside strings (balanced braces)
        candidate = self._extract_json_object(text)
        if candidate:
            try:
                return response_model.model_validate_json(candidate)
            except ValidationError:
                pass

        # 2) Direct parse of the whole text
        try:
            return response_model.model_validate_json(text)
        except ValidationError:
            pass

        # 3) JSON.loads the whole text (may be dict or JSON-encoded string)
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return response_model.model_validate(loaded)
            if isinstance(loaded, str):
                return response_model.model_validate_json(loaded)
        except Exception:
            pass

        # 4) Naive slice between first '{' and last '}' as a last resort
        try:
            first = text.find('{')
            last = text.rfind('}')
            if first != -1 and last != -1 and last > first:
                naive = text[first:last+1]
                return response_model.model_validate_json(naive)
        except Exception:
            pass

        # Could not validate
        raise LLMError("Validation failed")
            # try:
            #     response_clean_attempt = response_text.split('{', 1)[1]
            #     return response_model.model_validate_json(response_clean_attempt)
            # except ValidationError as e:
            #     log.warning("Response validation failed", exc_info=e)
            #    raise LLMError("Validation failed") from e

    def _add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _handle_error(self, e: Exception, attempt: int) -> None:
        log.error(f"An error occurred on attempt {attempt}: {str(e)}", exc_info=e)
        raise e

    def _log_response(self, response: Dict[str, Any]) -> None:
        try:
            usage_info = getattr(response, "usage", None)
            if usage_info is not None:
                usage_info = getattr(usage_info, "__dict__", usage_info)
            log.debug("Received chat response", extra={"usage": usage_info})
        except Exception:
            log.debug("Received chat response")

    def chat(self, user_prompt: str, response_model: BaseModel = None, max_tokens: int = 16000) -> Union[BaseModel, str]:
        self._add_to_history("user", user_prompt)
        messages = self.create_messages(user_prompt)
        response = self.send_message(messages, max_tokens, response_model)
        self._log_response(response)

        response_text = self.get_response(response)
        if response_model:
            try:
                validated = self._validate_response(response_text, response_model)
            except LLMError:
                # One-shot reformat attempt: ask the model to reprint as strict JSON
                try:
                    reform_prompt = (
                        "Reprint the following content as valid RFC 8259 JSON ONLY (no prose, no code fences), strictly matching the schema. "
                        "Escape all quotes (\"), newlines (\\n), and backslashes (\\\\) in string fields.\n\n"
                        f"Schema:\n{json.dumps(response_model.model_json_schema())}\n\n"
                        f"Content to reprint as JSON:\n{response_text}"
                    )
                    reform_messages = self.create_messages(reform_prompt)
                    reform_response = self.send_message(reform_messages, max_tokens, response_model)
                    reform_text = self.get_response(reform_response)
                    validated = self._validate_response(reform_text, response_model)
                    response_text = reform_text
                except Exception:
                    validated = self._fallback_response(response_model, response_text)
            # Store the raw assistant text in history, not the model object
            self._add_to_history("assistant", response_text or "")
            return validated
        else:
            self._add_to_history("assistant", response_text or "")
            return response_text

class Claude(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        # API key is retrieved from an environment variable by default
        self.client = anthropic.Anthropic(max_retries=3, base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        if "Provide a very concise summary of the README.md content" in user_prompt:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            self.prefill = "{    \"scratchpad\": \"1."
            messages = [{"role": "user", "content": user_prompt}, 
                        {"role": "assistant", "content": self.prefill}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        
        # response_model is not used here, only in ChatGPT
        return self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=messages
        )        

    def get_response(self, response: Dict[str, Any]) -> str:
        return response.content[0].text.replace('\n', '')


class ChatGPT(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}, 
                    {"role": "user", "content": user_prompt}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model=None) -> Dict[str, Any]:
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Add response format configuration if a model is provided
        if response_model:
            schema = None
            try:
                schema = response_model.model_json_schema()
            except Exception:
                schema = None
            if schema:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "VulnhuntrResponse",
                        "schema": schema,
                        "strict": True,
                    },
                }
            else:
                params["response_format"] = {"type": "json_object"}
            # Lower temperature to reduce format drift for structured outputs
            params["temperature"] = 0.2

            return self.client.chat.completions.create(**params)


    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.choices[0].message.content
        return response


class OpenRouter(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        default_headers = {}
        # Optional but recommended headers for OpenRouter analytics/rate limits
        if os.getenv("OPENROUTER_REFERRER"):
            default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERRER")
        if os.getenv("OPENROUTER_TITLE"):
            default_headers["X-Title"] = os.getenv("OPENROUTER_TITLE")

        self.client = openai.OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url,
            default_headers=default_headers or None,
        )
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model=None) -> Dict[str, Any]:
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if response_model:
            schema = None
            try:
                schema = response_model.model_json_schema()
            except Exception:
                schema = None
            if schema:
                closed = self._close_json_schema(schema)
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "VulnhuntrResponse",
                        "schema": closed,
                        "strict": True,
                    },
                }
            else:
                params["response_format"] = {"type": "json_object"}
            params["temperature"] = 0.2
        return self.client.chat.completions.create(**params)

    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.choices[0].message.content
        return response

class Ollama(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.api_url = base_url
        self.model = model

    def create_messages(self, user_prompt: str) -> str:
        return user_prompt

    def send_message(self, user_prompt: str, max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "options": {
            "temperature": 1,
            "system": self.system_prompt,
            }
            ,"stream":False,
        }

        try:
            response = requests.post(self.api_url, json=payload)
            return response
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 429:
                raise RateLimitError("Request was rate-limited") from e
            elif e.response.status_code >= 500:
                raise APIConnectionError("Server could not be reached") from e
            else:
                raise APIStatusError(e.response.status_code, e.response.json()) from e

    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.json()['response']
        return response

    def _log_response(self, response: Dict[str, Any]) -> None:
        log.debug("Received chat response", extra={"usage": "Ollama"})

