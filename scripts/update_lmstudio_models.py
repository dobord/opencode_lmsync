#!/usr/bin/env python3
"""
update_lmstudio_models.py

Обновляет для подключений типа `lmstudio` в конфиге
~/.config/opencode/opencode.json информацию о моделях и их размерах
контекста.

Поведение:
- Для каждого подключения типа `lmstudio` ищет API-ключ в
  ~/.local/share/opencode/auth.json. Если там ключ не найден, в качестве
  fallback используется переменная окружения $LM_API_TOKEN.
- Делает запрос GET /api/v1/models к указанному в подключении серверу и
  собирает два набора данных:
  * loaded_instances -> кэшируются значения loaded_instances[i].config.context_length
    по ключу loaded_instances[i].id (в кеше они привязаны к подключению)
  * models -> у каждой модели берётся max_context_length как запасной вариант
 - Обновляет в конфиге opencode.json список моделей у каждого подключения
  типа lmstudio: для каждой модели записывается поле `limit.context` (значение
  сначала ищется в кеше/loaded_instances, иначе используется max_context_length).
  Также инициализируется `limit.output = 0`, если отсутствует.

Пример использования:
  ./scripts/update_lmstudio_models.py

Опции:
  --config-path    Путь к opencode.json (по умолчанию ~/.config/opencode/opencode.json)
  --auth-path      Путь к auth.json (по умолчанию ~/.local/share/opencode/auth.json)
  --cache-path     Путь к файлу кэша (по умолчанию ~/.local/share/opencode/lmstudio_context_cache.json)
   --dry-run        Не записывать изменения, только показать что будет изменено
   --verbose        Подробный вывод
  Новые опции:
   --add-connection HOST[:PORT|/path]  Добавить подключение lmstudio по указанному адресу (например example.com:8080 или http://example.com:8080)
   --api-key KEY                       API ключ для нового подключения (если указан, будет записан в подключение)
   --connection-name NAME              Человекочитаемое имя для создаваемого подключения
   --no-autoload                        Если указан при добавлении подключения, не загружать модели автоматически

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/opencode/opencode.json")
DEFAULT_AUTH_PATH = os.path.expanduser("~/.local/share/opencode/auth.json")
DEFAULT_CACHE_PATH = os.path.expanduser(
    "~/.local/share/opencode/lmstudio_context_cache.json"
)
DEFAULT_LMSTUDIO_PORT = 1234
INTERNAL_FIELDS = ("_added_by_cli", "_provider_key")


def build_lmstudio_config_base_url(raw: str, default_port: int = DEFAULT_LMSTUDIO_PORT) -> str:
    raw = raw.strip()
    candidate = raw if "://" in raw else f"http://{raw}"
    parsed = urllib.parse.urlparse(candidate)
    host = parsed.hostname or parsed.netloc or parsed.path
    if not host:
        return raw.rstrip("/")
    port = parsed.port or default_port
    path = parsed.path if parsed.path and parsed.path != "/" else "/v1"
    if not path.startswith("/"):
        path = "/" + path
    return f"{parsed.scheme or 'http'}://{host}:{port}{path.rstrip('/')}"


def normalize_lmstudio_request_base_url(raw: str, default_port: int = DEFAULT_LMSTUDIO_PORT) -> str:
    raw = raw.strip()
    candidate = raw if "://" in raw else f"http://{raw}"
    parsed = urllib.parse.urlparse(candidate)
    host = parsed.hostname or parsed.netloc or parsed.path
    if not host:
        return raw.rstrip("/")
    port = parsed.port or default_port
    return f"{parsed.scheme or 'http'}://{host}:{port}"


def load_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logging.error("Не удалось прочитать JSON %s: %s", path, e)
        return None


def safe_write_json(path: str, data: Any) -> None:
    path = os.path.expanduser(path)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(p)


def strip_internal_fields(obj: Any, fields: Iterable[str] = INTERNAL_FIELDS) -> None:
    if isinstance(obj, dict):
        for field in fields:
            obj.pop(field, None)
        for value in obj.values():
            strip_internal_fields(value, fields)
    elif isinstance(obj, list):
        for item in obj:
            strip_internal_fields(item, fields)


def annotate_provider_keys(cfg: Any) -> None:
    if not isinstance(cfg, dict):
        return
    providers = cfg.get("provider")
    if not isinstance(providers, dict):
        return
    for provider_key, provider in providers.items():
        if isinstance(provider_key, str) and isinstance(provider, dict):
            provider["_provider_key"] = provider_key


def set_auth_api_key(auth: Any, auth_key: str, api_key: str) -> Tuple[Dict[str, Any], bool]:
    if not isinstance(auth, dict):
        auth = {}
    existing = auth.get(auth_key)
    if isinstance(existing, dict) and existing.get("type") == "api" and existing.get("key") == api_key:
        return auth, False
    auth[auth_key] = {"type": "api", "key": api_key}
    return auth, True


def find_connection_nodes(obj: Any) -> List[Dict[str, Any]]:
    """Рекурсивно находит в структуре все объекты, которые выглядят как подключения
    (dict с ключом 'type'). Возвращает список ссылок на эти dict'ы (изменения
    будут применены в исходной структуре).
    """
    found: List[Dict[str, Any]] = []

    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            if "type" in o:
                found.append(o)
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for item in o:
                _walk(item)

    _walk(obj)
    return found


def get_base_url_from_conn(conn: Dict[str, Any]) -> Optional[str]:
    """Попытки извлечь базовый URL/хост из описания подключения.
    Не конструирует URL, если нет хотя бы host/url-like значения.
    """
    candidates = [
        "base_url",
        "baseURL",
        "baseUrl",
        "url",
        "endpoint",
        "address",
        "host",
        "server",
        "server_url",
        "api_base_url",
        "api_url",
        "uri",
    ]
    for k in candidates:
        v = conn.get(k)
        if not v:
            continue
        if isinstance(v, str):
            s = v.strip()
            # If there's no scheme, assume http for parsing
            if "://" not in s:
                s_for_parse = "http://" + s
            else:
                s_for_parse = s
            try:
                p = urllib.parse.urlparse(s_for_parse)
                if p.netloc:
                    # return only scheme://netloc (strip any path)
                    return normalize_lmstudio_request_base_url(f"{p.scheme}://{p.netloc}")
            except Exception:
                # fallback to simple behavior
                if "://" not in s:
                    s = "http://" + s
                return normalize_lmstudio_request_base_url(s.rstrip("/"))

    # Попробуем собрать из host + port
    host = conn.get("host") or conn.get("hostname")
    port = conn.get("port")
    if host:
        # If host looks like a URL, parse and return scheme://netloc
        if isinstance(host, str) and ("/" in host or ":" in host and "//" in host):
            try:
                p = urllib.parse.urlparse(host if "://" in host else f"http://{host}")
                if p.netloc:
                    return normalize_lmstudio_request_base_url(f"{p.scheme}://{p.netloc}")
            except Exception:
                pass
        # Otherwise build from host(+port)
        if isinstance(host, str) and "://" not in host:
            host_str = host
        else:
            host_str = str(host)
        if port:
            return normalize_lmstudio_request_base_url(f"http://{host_str.rstrip('/')}:{port}")
        return normalize_lmstudio_request_base_url(f"http://{host_str.rstrip('/')}")

    # Некоторые провайдеры хранят url в nested options, например {"options": {"baseURL": "http://.../v1"}}
    options = conn.get("options") or conn.get("config") or conn.get("settings")
    if isinstance(options, dict):
        for k in candidates:
            v = options.get(k)
            if not v:
                continue
            if isinstance(v, str):
                s = v.strip()
                if "://" not in s:
                    s_for_parse = "http://" + s
                else:
                    s_for_parse = s
                try:
                    p = urllib.parse.urlparse(s_for_parse)
                    if p.netloc:
                        return normalize_lmstudio_request_base_url(f"{p.scheme}://{p.netloc}")
                except Exception:
                    if "://" not in s:
                        s = "http://" + s
                    return normalize_lmstudio_request_base_url(s.rstrip("/"))

    return None


def request_models(base_url: str, token: Optional[str]) -> Optional[Dict[str, Any]]:
    url = base_url.rstrip("/") + "/api/v1/models"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            try:
                return json.loads(data.decode("utf-8"))
            except Exception:
                logging.error("Не удалось распарсить JSON от %s", url)
                return None
    except urllib.error.HTTPError as e:
        logging.error("HTTP %s при обращении к %s: %s", e.code, url, e.reason)
    except urllib.error.URLError as e:
        logging.error("Ошибка запроса к %s: %s", url, e)
    except Exception as e:
        logging.error("Неожиданная ошибка при запросе %s: %s", url, e)
    return None


def extract_token_from_auth(auth: Any, conn: Dict[str, Any], base_url: Optional[str]) -> Optional[str]:
    """Хитрые эвристики: ищем подходящий токен в структуре auth.json.
    Если ничего не найдено, возвращаем None.
    """
    if auth is None:
        return None

    # helper to try to extract token string from a value
    def try_extract(v: Any) -> Optional[str]:
        if v is None:
            return None
        # If this value is a plain string, treat it as a token (common simple case)
        if isinstance(v, str):
            return v

        # Search dictionaries for known token-like keys and common nested sections.
        if isinstance(v, dict):
            # direct token keys at this level
            for key in ("token", "api_key", "api_token", "access_token", "value", "key"):
                if key in v and isinstance(v[key], str):
                    return v[key]

            # headers.Authorization -> 'Bearer ...' or similar
            headers = v.get("headers") or v.get("auth")
            if isinstance(headers, dict):
                authv = headers.get("Authorization") or headers.get("authorization")
                if isinstance(authv, str):
                    if authv.lower().startswith("bearer "):
                        return authv.split(None, 1)[1]
                    return authv

            # Common nested sections that may contain keys, e.g. {"api": {"key": "..."}}
            for nested in ("api", "credentials", "auth", "headers", "authorization"):
                if nested in v:
                    tok = try_extract(v[nested])
                    if tok:
                        return tok

            # As a last resort, recurse into any dict/list children (but do not return simple strings
            # found under arbitrary keys to avoid false positives).
            for subv in v.values():
                if isinstance(subv, (dict, list)):
                    tok = try_extract(subv)
                    if tok:
                        return tok

        # If this is a list, try items (items may be strings or dicts)
        if isinstance(v, list):
            for item in v:
                tok = try_extract(item)
                if tok:
                    return tok

        return None

    # 1) Если auth — dict и содержит ключи, совпадающие с id/name/url подключения
    if isinstance(auth, dict):
        # Try direct identifiers
        for key in (conn.get("_provider_key"), conn.get("id"), conn.get("name"), base_url):
            if not key:
                continue
            if key in auth:
                tok = try_extract(auth[key])
                if tok:
                    return tok

        # Try to match by host/URL in top-level keys. Use normalized hostname
        # comparison (preferred) and fall back to substring match for
        # backward-compatibility with earlier auth.json shapes.
        def _get_host(s: Optional[str]) -> Optional[str]:
            """Return normalized hostname for a string that may be a bare host,
            host:port, or a full URL. Returns lower-cased hostname or None.
            """
            if not s or not isinstance(s, str):
                return None
            s = s.strip()
            if not s:
                return None
            # Ensure parseable URL
            s_for_parse = s if ("://" in s) else ("http://" + s)
            try:
                p = urllib.parse.urlparse(s_for_parse)
                if p.hostname:
                    return p.hostname.lower()
            except Exception:
                pass
            # fallback: split off port if present
            if ":" in s:
                return s.split(":", 1)[0].lower()
            return s.lower()

        base_host = _get_host(base_url) if base_url else None
        for k, v in auth.items():
            if not isinstance(k, str):
                continue
            # Prefer normalized hostname equality
            if base_host:
                key_host = _get_host(k)
                if key_host and key_host == base_host:
                    tok = try_extract(v)
                    if tok:
                        return tok
            # Fallback to substring match for legacy keys
            if k and base_url and k in base_url:
                tok = try_extract(v)
                if tok:
                    return tok

        # Если это подключение lmstudio и в auth.json есть секция 'lmstudio',
        # используем её как fallback для подключений без host-specific ключа.
        if conn.get("type") == "lmstudio" and "lmstudio" in auth:
            tok = try_extract(auth["lmstudio"])
            if tok:
                return tok

        # If auth contains a single top-level token-like key, use it as global
        for k in ("lm_api_token", "LM_API_TOKEN", "token", "api_token", "api_key"):
            if k in auth:
                tok = try_extract(auth[k])
                if tok:
                    return tok

        # As a last resort for dict-shaped auth, try to find any token anywhere
        # inside the auth structure. This covers cases where tokens are nested
        # under arbitrary keys (e.g. {"hosts": [{...}]}) and earlier heuristics
        # did not match. Use this only as low-priority fallback to avoid false
        # positives.
        tok = try_extract(auth)
        if tok:
            return tok

    # 2) If auth is a list, try to find an entry matching base_url/host
    if isinstance(auth, list):
        for item in auth:
            if not isinstance(item, dict):
                continue
            # check host/url fields
            for hkey in ("host", "url", "base_url", "endpoint"): 
                hv = item.get(hkey)
                if hv and base_url and str(hv) in base_url:
                    tok = try_extract(item)
                    if tok:
                        return tok
            tok = try_extract(item)
            if tok:
                return tok

    return None


def extract_token_from_conn(conn: Any) -> Optional[str]:
    """Try to extract an API token from a connection dict itself.
    This mirrors the heuristics used for auth.json but looks inside the
    connection description so CLI-added connections with --api-key work.
    """
    if conn is None:
        return None

    def try_extract(v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for key in ("token", "api_key", "api_token", "access_token", "value", "key"):
                if key in v and isinstance(v[key], str):
                    return v[key]

            headers = v.get("headers") or v.get("auth")
            if isinstance(headers, dict):
                authv = headers.get("Authorization") or headers.get("authorization")
                if isinstance(authv, str):
                    if authv.lower().startswith("bearer "):
                        return authv.split(None, 1)[1]
                    return authv

            for nested in ("api", "credentials", "auth", "headers", "authorization"):
                if nested in v:
                    tok = try_extract(v[nested])
                    if tok:
                        return tok

        if isinstance(v, list):
            for item in v:
                tok = try_extract(item)
                if tok:
                    return tok

        return None

    return try_extract(conn)


def parse_models_response(resp: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Возвращает кортеж двух словарей:
    - server_models: model_id -> max_context_length (если найдено)
    - loaded_instances: instance_id -> config.context_length
    """
    server_models: Dict[str, int] = {}
    loaded_instances: Dict[str, int] = {}

    if not isinstance(resp, dict):
        return server_models, loaded_instances

    # models
    models = resp.get("models") or resp.get("available_models") or []
    if isinstance(models, dict):
        # maybe keyed by id
        for k, v in models.items():
            if isinstance(v, dict):
                max_ctx = v.get("max_context_length") or v.get("max_context") or v.get("context_window") or v.get("context_length")
                if isinstance(max_ctx, int):
                    server_models[k] = int(max_ctx)
            else:
                # unknown structure
                continue
    elif isinstance(models, list):
        for m in models:
            if not isinstance(m, dict):
                continue
            # model id may be stored under different keys depending on API
            mid = m.get("id") or m.get("model") or m.get("name") or m.get("key")
            if not mid:
                continue
            max_ctx = (
                m.get("max_context_length")
                or m.get("max_context")
                or m.get("context_window")
                or m.get("context_length")
            )
            if isinstance(max_ctx, int):
                server_models[str(mid)] = int(max_ctx)

            # If the model has loaded_instances listed per-model, collect their context lengths
            per_li = m.get("loaded_instances") or m.get("loaded_models") or []
            if isinstance(per_li, dict):
                for k, v in per_li.items():
                    if not isinstance(v, dict):
                        continue
                    iid = v.get("id") or k
                    cfg = v.get("config") or {}
                    if isinstance(cfg, dict):
                        ctx = cfg.get("context_length") or cfg.get("context")
                        if isinstance(ctx, int):
                            loaded_instances[str(iid)] = int(ctx)
            elif isinstance(per_li, list):
                for v in per_li:
                    if not isinstance(v, dict):
                        continue
                    iid = v.get("id") or v.get("instance_id") or v.get("name")
                    cfg = v.get("config") or {}
                    if isinstance(cfg, dict):
                        ctx = cfg.get("context_length") or cfg.get("context")
                        if isinstance(ctx, int) and iid:
                            loaded_instances[str(iid)] = int(ctx)

    # loaded_instances
    li = resp.get("loaded_instances") or resp.get("loaded_models") or []
    if isinstance(li, dict):
        # values might be instance dicts
        for k, v in li.items():
            if not isinstance(v, dict):
                continue
            iid = v.get("id") or k
            cfg = v.get("config") or {}
            if isinstance(cfg, dict):
                ctx = cfg.get("context_length") or cfg.get("context")
                if isinstance(ctx, int):
                    loaded_instances[str(iid)] = int(ctx)
    elif isinstance(li, list):
        for v in li:
            if not isinstance(v, dict):
                continue
            iid = v.get("id") or v.get("model")
            cfg = v.get("config") or {}
            if isinstance(cfg, dict):
                ctx = cfg.get("context_length") or cfg.get("context")
                if isinstance(ctx, int) and iid:
                    loaded_instances[str(iid)] = int(ctx)

    return server_models, loaded_instances


def get_model_id_from_entry(entry: Any) -> Optional[str]:
    if entry is None:
        return None
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("id", "model", "name", "model_id"):
            v = entry.get(key)
            if isinstance(v, str):
                return v
    return None


def get_context_for_model(
    connection_key: str,
    model_id: str,
    server_models: Dict[str, int],
    cache: Dict[str, Dict[str, int]],
    loaded_instances: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    """Return context size with priority:
    1) loaded_instances (from REST response)
    2) cache
    3) server_models (max_context_length)
    The function performs exact match first and then fuzzy matches (startswith / contains).
    """
    # 1) check loaded_instances (REST) first
    if isinstance(loaded_instances, dict):
        # exact match
        if model_id in loaded_instances:
            return loaded_instances[model_id]
        # fuzzy matches
        for inst_id, ctx in loaded_instances.items():
            if inst_id == model_id:
                return ctx
            if inst_id.startswith(model_id):
                return ctx
            if model_id in inst_id:
                return ctx

    # 2) check cache
    if connection_key in cache and isinstance(cache[connection_key], dict):
        # exact match
        if model_id in cache[connection_key]:
            return cache[connection_key][model_id]
        # fuzzy matches
        for inst_id, ctx in cache[connection_key].items():
            if inst_id == model_id:
                return ctx
            if inst_id.startswith(model_id):
                return ctx
            if model_id in inst_id:
                return ctx

    # 3) fallback to server reported max_context
    if model_id in server_models:
        return server_models[model_id]

    return None


def update_connection_models(
    conn: Dict[str, Any],
    connection_key: str,
    server_models: Dict[str, int],
    cache: Dict[str, Dict[str, int]],
    loaded_instances: Dict[str, int],
) -> Tuple[bool, List[str]]:
    """Обновляет conn['models'] (или создаёт) и возвращает (changed, messages).
    changed - True если внесены изменения.
    messages - список коротких сообщений о том что изменилось.
    """
    messages: List[str] = []
    changed = False

    # Ensure cache contains loaded_instances for this connection
    conn_cache = cache.setdefault(connection_key, {})
    # merge loaded_instances into cache (overwrite)
    for iid, ctx in loaded_instances.items():
        if conn_cache.get(iid) != ctx:
            conn_cache[iid] = ctx
            changed = True
            messages.append(f"cache[{connection_key}][{iid}]={ctx}")

    # current models in connection
    models_obj = conn.get("models")

    if isinstance(models_obj, list):
        converted_models: Dict[str, Any] = {}
        for entry in models_obj:
            mid = get_model_id_from_entry(entry)
            if not mid:
                continue
            if isinstance(entry, dict):
                converted_models[str(mid)] = entry
            else:
                converted_models[str(mid)] = {"limit": {"output": 0}}
        models_obj = converted_models
        conn["models"] = models_obj
        changed = True
        messages.append(f"models: converted list to object with {len(converted_models)} entries")

    # Ensure we don't leave or write legacy context keys into the config.
    banned_keys = {
        "context_window",
        "contextWindow",
        "context_size",
        "context-size",
        "contextLength",
        "context_length",
    }
    if isinstance(models_obj, dict):
        for mid, info in list(models_obj.items()):
            if isinstance(info, dict):
                for bk in list(banned_keys):
                    if bk in info:
                        info.pop(bk, None)
                        changed = True
                        messages.append(f"models[{mid}]: removed {bk}")

        for mid, info in list(models_obj.items()):
            if not isinstance(info, dict):
                models_obj[mid] = {"limit": {"output": 0}}
                changed = True
                messages.append(f"models[{mid}].limit.output=0")
                continue
            lim = info.setdefault("limit", {})
            if "output" not in lim:
                lim["output"] = 0
                changed = True
                messages.append(f"models[{mid}].limit.output=0")

        # Update existing entries and add missing server models.
        for mid, maxctx in server_models.items():
            ctx = get_context_for_model(connection_key, mid, server_models, cache, loaded_instances)
            if mid in models_obj:
                info = models_obj[mid]
                if ctx is not None:
                    if not isinstance(info, dict):
                        models_obj[mid] = {"limit": {"context": ctx, "output": 0}}
                        changed = True
                        messages.append(f"models[{mid}].limit.context={ctx}")
                        messages.append(f"models[{mid}].limit.output=0")
                    else:
                        lim = info.setdefault("limit", {})
                        if lim.get("context") != ctx:
                            lim["context"] = ctx
                            changed = True
                            messages.append(f"models[{mid}].limit.context={ctx}")
            else:
                entry: Dict[str, Any] = {"id": mid}
                if ctx is not None:
                    entry["limit"] = {"context": ctx, "output": 0}
                else:
                    entry["limit"] = {"output": 0}
                models_obj[mid] = entry
                changed = True
                messages.append(f"models: add {mid} limit.context={ctx}")
                messages.append(f"models: add {mid} limit.output=0")

    else:
        # no models defined, create object from server_models
        new_models: Dict[str, Any] = {}
        for mid, maxctx in server_models.items():
            ctx = get_context_for_model(connection_key, mid, server_models, cache, loaded_instances)
            entry: Dict[str, Any] = {"id": mid}
            if ctx is not None:
                entry["limit"] = {"context": ctx, "output": 0}
            else:
                entry["limit"] = {"output": 0}
            new_models[mid] = entry
        if new_models:
            conn["models"] = new_models
            changed = True
            messages.append(f"models: created {len(new_models)} entries")

    return changed, messages


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Update LM Studio model context sizes in opencode config")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--auth-path", default=DEFAULT_AUTH_PATH)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--add-connection", help="Добавить lmstudio подключение (host[:port] или URL)")
    parser.add_argument("--api-key", help="API ключ для добавляемого подключения")
    parser.add_argument("--connection-name", help="Имя для добавляемого подключения")
    parser.add_argument("--no-autoload", action="store_true", help="При добавлении подключения не загружать модели автоматически")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config_path = os.path.expanduser(args.config_path)
    auth_path = os.path.expanduser(args.auth_path)
    cache_path = os.path.expanduser(args.cache_path)

    config_exists = Path(config_path).exists()
    cfg = load_json(config_path)
    if cfg is None:
        if config_exists:
            logging.error("Не удалось загрузить конфиг %s", config_path)
            return 2
        logging.info("Конфиг %s не найден, начинаю с пустой конфигурации", config_path)
        cfg = {}

    auth = load_json(auth_path)
    # Track whether we modified auth in-memory so we can persist it later.
    auth_modified = False
    config_modified = False

    cache = load_json(cache_path) or {}
    # ensure cache shape: {connection_key: {instance_id: ctx, ...}}
    if not isinstance(cache, dict):
        cache = {}

    # If requested, add a new connection to the in-memory config before we
    # discover connections. The new connection will be processed in the same
    # run and (unless --dry-run) persisted to the config file.
    if args.add_connection:
        host = args.add_connection.strip()
        new_conn: Dict[str, Any] = {"type": "lmstudio"}
        # store normalized base_url so get_base_url_from_conn can parse it
        new_conn["base_url"] = build_lmstudio_config_base_url(host)
        # Determine a human-friendly hostname for display (without port)
        try:
            p = urllib.parse.urlparse(new_conn["base_url"])
            hostname = p.hostname or p.netloc or host
        except Exception:
            hostname = host

        # Build a base display name like "LM Studio (192.168.1.82)" and ensure
        # uniqueness among existing connections by appending a numeric suffix if needed.
        existing_conns = find_connection_nodes(cfg)
        used_names = set()
        for c in existing_conns:
            if isinstance(c, dict):
                cid = c.get("id")
                cname = c.get("name")
                if isinstance(cid, str):
                    used_names.add(cid)
                if isinstance(cname, str):
                    used_names.add(cname)

        base_display = f"LM Studio ({hostname})"
        candidate = base_display
        idx = 2
        while candidate in used_names:
            candidate = f"{base_display} #{idx}"
            idx += 1

        # Use the unique candidate for both id and name unless user provided a name
        new_conn["id"] = candidate
        new_conn["name"] = args.connection_name or candidate
        # mark new connection so we can treat it specially in the processing loop
        new_conn["_added_by_cli"] = True
        # If API key provided, write it into auth.json (preferred) rather than
        # storing it directly in the config entry. Use a structured object so
        # auth.json remains consistent with other entries (type/key).
        # Compute a network identifier for auth (netloc includes optional port).
        try:
            parsed = urllib.parse.urlparse(new_conn["base_url"])
            netloc_str = parsed.netloc or parsed.hostname or host
        except Exception:
            netloc_str = host

        pending_api_key = args.api_key or os.environ.get("LM_API_TOKEN")
        pending_api_key_source = "--api-key" if args.api_key else "LM_API_TOKEN" if pending_api_key else None

        # inject into provider mapping (preferred shape for opencode config)
        if isinstance(cfg, dict):
            providers = cfg.setdefault("provider", {})
            # Build a machine-friendly provider key (no spaces) based on netloc
            safe_host = "".join(c if c.isalnum() else "_" for c in netloc_str).lower()
            base_key = f"lmstudio_{safe_host}" if safe_host else "lmstudio"
            prov_key = base_key
            idx = 2
            while prov_key in providers:
                prov_key = f"{base_key}_{idx}"
                idx += 1

            if pending_api_key:
                auth, stored_auth = set_auth_api_key(auth, prov_key, pending_api_key)
                if stored_auth:
                    auth_modified = True
                    logging.info(
                        "Added API key from %s to auth.json under provider id '%s'",
                        pending_api_key_source,
                        prov_key,
                    )

            # build provider object similar to existing providers (if a template exists)
            base_url_for_options = new_conn["base_url"]
            template = providers.get("lmstudio") if isinstance(providers.get("lmstudio"), dict) else None
            provider_obj: Dict[str, Any] = {}
            if template and isinstance(template.get("npm"), str):
                provider_obj["npm"] = template.get("npm")
            provider_obj.update({
                "type": "lmstudio",
                "name": new_conn.get("name"),
                "id": new_conn.get("id"),
                "options": {"baseURL": base_url_for_options},
            })
            # mark so the processing loop can detect CLI-added providers
            provider_obj["_added_by_cli"] = True
            provider_obj["_provider_key"] = prov_key
            # keep the models shape stable even before autoload fills it
            provider_obj["models"] = {}

            providers[prov_key] = provider_obj
            config_modified = True
            logging.info("Added provider entry provider['%s'] name=%s base_url=%s", prov_key, provider_obj.get("name"), base_url_for_options)
        else:
            logging.error("Config file has unexpected shape; cannot add connection")

    # Find connection nodes in several common shapes.
    annotate_provider_keys(cfg)
    connections = find_connection_nodes(cfg)

    # Some configs (like opencode) may list providers under cfg['provider']['lmstudio']
    providers = cfg.get("provider") if isinstance(cfg, dict) else None
    if isinstance(providers, dict) and "lmstudio" in providers:
        prov = providers.get("lmstudio")
        if isinstance(prov, dict):
            # Normalize provider dict to look like a connection: ensure it has 'type'
            prov_conn = prov
            # set in-memory type so it will be processed as lmstudio
            prov_conn.setdefault("type", "lmstudio")
            # set an id so auth.json keys like 'lmstudio' can be matched
            prov_conn.setdefault("id", "lmstudio")
            if prov_conn not in connections:
                connections.append(prov_conn)
    if not connections:
        logging.info("Не найдены подключения в конфиге %s", config_path)
        return 0

    total_changed = False
    cache_modified = False
    for conn in connections:
        try:
            if conn.get("type") != "lmstudio":
                continue

            base_url = get_base_url_from_conn(conn)
            if not base_url:
                logging.warning("Пропускаю подключение (не найден base_url): %s", conn)
                continue

            connection_key = base_url  # используем base_url как ключ кеша

            # Найдём токен: сначала в самом описании подключения (если пользователь
            # добавил его через --api-key), затем в auth.json, затем LM_API_TOKEN
            token_from_conn = extract_token_from_conn(conn)
            token_from_auth = extract_token_from_auth(auth, conn, base_url)
            token_env = os.environ.get("LM_API_TOKEN")
            if token_from_conn:
                token = token_from_conn
                token_source = "conn"
            elif token_from_auth:
                token = token_from_auth
                token_source = "auth.json"
            elif token_env:
                token = token_env
                token_source = "LM_API_TOKEN"
            else:
                token = None
                token_source = "none"

            provider_key = conn.get("_provider_key")
            if isinstance(provider_key, str) and token and token_source == "auth.json" and isinstance(auth, dict):
                auth, stored_auth = set_auth_api_key(auth, provider_key, token)
                if stored_auth:
                    auth_modified = True
                    logging.info("Migrated API key in auth.json to provider id '%s'", provider_key)

            # Информация о подключении и текущем состоянии
            conn_id = conn.get("id")
            conn_name = conn.get("name")
            logging.info("Обрабатываю lmstudio подключение id=%s name=%s base_url=%s token_source=%s",
                         conn_id or "<none>", conn_name or "<none>", base_url, token_source)

            # If this connection was added by CLI and the user requested no autoload,
            # skip attempting to fetch models from server but ensure config will be
            # persisted.
            if conn.get("_added_by_cli") and args.no_autoload:
                logging.info("New connection added via CLI and --no-autoload: skipping model fetch for %s", conn.get("id"))
                total_changed = True
                # show current models (which will likely be None) and continue
                continue

            # Показываем текущие модели в конфиге (до изменений)
            current_models = conn.get("models")
            logging.info("  configured models (before): %s",
                         json.dumps(current_models, ensure_ascii=False, indent=2) if current_models is not None else "None")

            # Показываем кэшированные значения до запроса
            # Make a shallow copy so we can detect later whether cache for this
            # connection was modified (we persist cache file if it was).
            conn_cache_before = dict(cache.get(connection_key, {}))
            logging.info("  cache (before): %s", json.dumps(conn_cache_before, ensure_ascii=False, indent=2) if conn_cache_before else "{}")

            # Выполняем запрос к серверу
            resp = request_models(base_url, token)
            if not resp:
                logging.warning("Нет ответа от %s, пропускаю (token_source=%s)", base_url, token_source)
                # даже если нет ответа, выводим текущ состояние и переходим к следующему
                continue

            server_models, loaded_instances = parse_models_response(resp)
            logging.info("  server models: %s", json.dumps(server_models, ensure_ascii=False, indent=2) if server_models else "{}")
            logging.info("  loaded instances: %s", json.dumps(loaded_instances, ensure_ascii=False, indent=2) if loaded_instances else "{}")

            # If server_models is empty, attempt to extract fallback structure
            if not server_models:
                # Maybe the API returned a flat list
                if isinstance(resp, list):
                    for m in resp:
                        if isinstance(m, dict):
                            mid = m.get("id") or m.get("model")
                            max_ctx = m.get("max_context_length") or m.get("max_context")
                            if mid and isinstance(max_ctx, int):
                                server_models[str(mid)] = int(max_ctx)

            changed, messages = update_connection_models(conn, connection_key, server_models, cache, loaded_instances)
            if changed:
                total_changed = True
                for msg in messages:
                    logging.info("  %s", msg)

            # Показываем кэш после обновления
            cache_after = cache.get(connection_key, {})
            logging.info("  cache (after): %s", json.dumps(cache_after, ensure_ascii=False, indent=2) if cache_after else "{}")
            # If the cache for this connection changed, remember it so we can
            # persist the cache file even when no config changes were made.
            if conn_cache_before != cache_after:
                cache_modified = True

            # Показываем итоговый набор моделей (id -> context_size)
            final_models = conn.get("models")
            def _models_map(models_obj):
                out = []
                if isinstance(models_obj, list):
                    for e in models_obj:
                        mid = get_model_id_from_entry(e)
                        out_entry: Dict[str, Any] = {"id": mid}
                        if isinstance(e, dict):
                            # prefer new nested limit section if present
                            lim = e.get("limit")
                            if isinstance(lim, dict) and isinstance(lim.get("context"), int):
                                out_entry["limit"] = {
                                    "context": lim.get("context"),
                                    "output": lim.get("output"),
                                }
                            else:
                                ctx = e.get("context_size")
                                if ctx is not None:
                                    out_entry["context_size"] = ctx
                        out.append(out_entry)
                elif isinstance(models_obj, dict):
                    for mid, info in models_obj.items():
                        out_entry = {"id": mid}
                        if isinstance(info, dict):
                            lim = info.get("limit")
                            if isinstance(lim, dict) and isinstance(lim.get("context"), int):
                                out_entry["limit"] = {
                                    "context": lim.get("context"),
                                    "output": lim.get("output"),
                                }
                            else:
                                ctx = info.get("context_size")
                                if ctx is not None:
                                    out_entry["context_size"] = ctx
                        out.append(out_entry)
                return out

            logging.info("  final models: %s", json.dumps(_models_map(final_models), ensure_ascii=False, indent=2))

        except Exception as e:
            logging.exception("Ошибка при обработке подключения: %s", e)

    if (total_changed or cache_modified or auth_modified or config_modified) and not args.dry_run:
        try:
            # Persist only the files that changed to avoid unnecessary rewrites.
            if total_changed or config_modified:
                strip_internal_fields(cfg)
                safe_write_json(config_path, cfg)
            if cache_modified:
                safe_write_json(cache_path, cache)
            if auth_modified:
                safe_write_json(auth_path, auth)
            if total_changed or config_modified:
                logging.info("Конфиг успешно сохранён")
            if cache_modified:
                logging.info("Кэш успешно сохранён")
            if auth_modified:
                logging.info("auth.json успешно сохранён")
        except Exception as e:
            logging.error("Не удалось сохранить файлы: %s", e)
            return 3
    else:
        if args.dry_run:
            logging.info("Dry-run: изменений не записываю")
        else:
            logging.info("Изменений не обнаружено")

    return 0


if __name__ == "__main__":
    sys.exit(main())
