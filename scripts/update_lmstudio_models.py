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
  типа lmstudio, указывая для каждой модели поле `context_size`. В качестве
  значения сначала ищется кэшированное значение (или значение из текущего
  ответа loaded_instances), иначе подставляется max_context_length.

Пример использования:
  ./scripts/update_lmstudio_models.py

Опции:
  --config-path    Путь к opencode.json (по умолчанию ~/.config/opencode/opencode.json)
  --auth-path      Путь к auth.json (по умолчанию ~/.local/share/opencode/auth.json)
  --cache-path     Путь к файлу кэша (по умолчанию ~/.local/share/opencode/lmstudio_context_cache.json)
  --dry-run        Не записывать изменения, только показать что будет изменено
  --verbose        Подробный вывод

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
                    return f"{p.scheme}://{p.netloc}"
            except Exception:
                # fallback to simple behavior
                if "://" not in s:
                    s = "http://" + s
                return s.rstrip("/")

    # Попробуем собрать из host + port
    host = conn.get("host") or conn.get("hostname")
    port = conn.get("port")
    if host:
        # If host looks like a URL, parse and return scheme://netloc
        if isinstance(host, str) and ("/" in host or ":" in host and "//" in host):
            try:
                p = urllib.parse.urlparse(host if "://" in host else f"http://{host}")
                if p.netloc:
                    return f"{p.scheme}://{p.netloc}"
            except Exception:
                pass
        # Otherwise build from host(+port)
        if isinstance(host, str) and "://" not in host:
            host_str = host
        else:
            host_str = str(host)
        if port:
            return f"http://{host_str.rstrip('/')}:{port}"
        return f"http://{host_str.rstrip('/')}"

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
                        return f"{p.scheme}://{p.netloc}"
                except Exception:
                    if "://" not in s:
                        s = "http://" + s
                    return s.rstrip("/")

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
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            # Common keys
            for key in ("token", "api_key", "api_token", "access_token", "value", "key"):
                if key in v and isinstance(v[key], str):
                    return v[key]
            # headers.Authorization -> 'Bearer ...'
            headers = v.get("headers") or v.get("auth")
            if isinstance(headers, dict):
                authv = headers.get("Authorization") or headers.get("authorization")
                if isinstance(authv, str):
                    # strip Bearer
                    if authv.lower().startswith("bearer "):
                        return authv.split(None, 1)[1]
                    return authv
        return None

    # 1) Если auth — dict и содержит ключи, совпадающие с id/name/url подключения
    if isinstance(auth, dict):
        # Try direct identifiers
        for key in (conn.get("id"), conn.get("name"), base_url):
            if not key:
                continue
            if key in auth:
                tok = try_extract(auth[key])
                if tok:
                    return tok

        # Try to match by host substring in keys
        if base_url:
            for k, v in auth.items():
                if not isinstance(k, str):
                    continue
                if k and k in base_url:
                    tok = try_extract(v)
                    if tok:
                        return tok

        # If auth contains a single top-level token-like key, use it as global
        for k in ("lm_api_token", "LM_API_TOKEN", "token", "api_token", "api_key"):
            if k in auth and isinstance(auth[k], str):
                return auth[k]

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


def get_context_for_model(connection_key: str, model_id: str, server_models: Dict[str, int], cache: Dict[str, Dict[str, int]]) -> Optional[int]:
    # 1) exact cache match
    if connection_key in cache and model_id in cache[connection_key]:
        return cache[connection_key][model_id]

    # 2) try to match cache keys by prefix or substring
    if connection_key in cache:
        for inst_id, ctx in cache[connection_key].items():
            if inst_id == model_id:
                return ctx
            # common pattern: instance id starts with model_id
            if inst_id.startswith(model_id):
                return ctx
            # or model_id contained in inst_id
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

    # Build existing map: model_id -> (entry, index) for lists
    if isinstance(models_obj, list):
        existing_map: Dict[str, Tuple[Any, int]] = {}
        for idx, entry in enumerate(models_obj):
            mid = get_model_id_from_entry(entry)
            if mid:
                existing_map[str(mid)] = (entry, idx)

        # Update existing entries
        for mid, (entry, idx) in list(existing_map.items()):
            ctx = get_context_for_model(connection_key, mid, server_models, cache)
            if ctx is not None:
                # if entry is string, replace with dict
                if isinstance(entry, str):
                    conn["models"][idx] = {"id": mid, "context_size": ctx}
                    changed = True
                    messages.append(f"models: set {mid}.context_size={ctx}")
                elif isinstance(entry, dict):
                    if entry.get("context_size") != ctx:
                        entry["context_size"] = ctx
                        changed = True
                        messages.append(f"models: set {mid}.context_size={ctx}")

        # Add missing server models that are not present in config
        for mid, maxctx in server_models.items():
            if mid in existing_map:
                continue
            ctx = get_context_for_model(connection_key, mid, server_models, cache)
            entry = {"id": mid}
            if ctx is not None:
                entry["context_size"] = ctx
            conn.setdefault("models", []).append(entry)
            changed = True
            messages.append(f"models: add {mid} context_size={ctx}")

    elif isinstance(models_obj, dict):
        # keys are model ids
        for mid, info in models_obj.items():
            ctx = get_context_for_model(connection_key, mid, server_models, cache)
            if ctx is not None:
                if not isinstance(info, dict):
                    models_obj[mid] = {"context_size": ctx}
                    changed = True
                    messages.append(f"models[{mid}] = {{context_size: {ctx}}}")
                else:
                    if info.get("context_size") != ctx:
                        info["context_size"] = ctx
                        changed = True
                        messages.append(f"models[{mid}].context_size={ctx}")

        # add missing server models
        for mid, maxctx in server_models.items():
            if mid in models_obj:
                continue
            ctx = get_context_for_model(connection_key, mid, server_models, cache)
            models_obj[mid] = {"context_size": ctx} if ctx is not None else {}
            changed = True
            messages.append(f"models: add {mid} context_size={ctx}")

    else:
        # no models defined, create list from server_models
        new_models = []
        for mid, maxctx in server_models.items():
            ctx = get_context_for_model(connection_key, mid, server_models, cache)
            entry = {"id": mid}
            if ctx is not None:
                entry["context_size"] = ctx
            new_models.append(entry)
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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config_path = os.path.expanduser(args.config_path)
    auth_path = os.path.expanduser(args.auth_path)
    cache_path = os.path.expanduser(args.cache_path)

    cfg = load_json(config_path)
    if cfg is None:
        logging.error("Не удалось загрузить конфиг %s", config_path)
        return 2

    auth = load_json(auth_path)

    cache = load_json(cache_path) or {}
    # ensure cache shape: {connection_key: {instance_id: ctx, ...}}
    if not isinstance(cache, dict):
        cache = {}

    # Find connection nodes in several common shapes.
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
    for conn in connections:
        try:
            if conn.get("type") != "lmstudio":
                continue

            base_url = get_base_url_from_conn(conn)
            if not base_url:
                logging.warning("Пропускаю подключение (не найден base_url): %s", conn)
                continue

            connection_key = base_url  # используем base_url как ключ кеша

            # Найдём токен: сначала auth.json, затем LM_API_TOKEN (определяем источник)
            token_from_auth = extract_token_from_auth(auth, conn, base_url)
            token_env = os.environ.get("LM_API_TOKEN")
            if token_from_auth:
                token = token_from_auth
                token_source = "auth.json"
            elif token_env:
                token = token_env
                token_source = "LM_API_TOKEN"
            else:
                token = None
                token_source = "none"

            # Информация о подключении и текущем состоянии
            conn_id = conn.get("id")
            conn_name = conn.get("name")
            logging.info("Обрабатываю lmstudio подключение id=%s name=%s base_url=%s token_source=%s",
                         conn_id or "<none>", conn_name or "<none>", base_url, token_source)

            # Показываем текущие модели в конфиге (до изменений)
            current_models = conn.get("models")
            logging.info("  configured models (before): %s",
                         json.dumps(current_models, ensure_ascii=False, indent=2) if current_models is not None else "None")

            # Показываем кэшированные значения до запроса
            cache_before = cache.get(connection_key, {})
            logging.info("  cache (before): %s", json.dumps(cache_before, ensure_ascii=False, indent=2) if cache_before else "{}")

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

            # Показываем итоговый набор моделей (id -> context_size)
            final_models = conn.get("models")
            def _models_map(models_obj):
                out = []
                if isinstance(models_obj, list):
                    for e in models_obj:
                        mid = get_model_id_from_entry(e)
                        ctx = e.get("context_size") if isinstance(e, dict) else None
                        out.append({"id": mid, "context_size": ctx})
                elif isinstance(models_obj, dict):
                    for mid, info in models_obj.items():
                        ctx = info.get("context_size") if isinstance(info, dict) else None
                        out.append({"id": mid, "context_size": ctx})
                return out

            logging.info("  final models: %s", json.dumps(_models_map(final_models), ensure_ascii=False, indent=2))

        except Exception as e:
            logging.exception("Ошибка при обработке подключения: %s", e)

    if total_changed and not args.dry_run:
        try:
            safe_write_json(config_path, cfg)
            safe_write_json(cache_path, cache)
            logging.info("Конфиг и кэш успешно сохранены")
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
