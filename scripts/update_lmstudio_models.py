#!/usr/bin/env python3
"""Synchronize LM Studio metadata without converting OpenCode config formats."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request


DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/opencode/opencode.json")
DEFAULT_AUTH_PATH = os.path.expanduser("~/.local/share/opencode/auth.json")
DEFAULT_CACHE_PATH = os.path.expanduser("~/.local/share/opencode/lmstudio_context_cache.json")


class SyncError(ValueError):
    """An intentionally sanitized message safe for the user-facing log."""


def reject_constant(_):
    raise SyncError("JSON содержит нечисловую константу")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SyncError("JSON содержит повторяющиеся ключи")
        result[key] = value
    return result


def detect_opencode_version(binary):
    """Never infer a version from the config, PATH failures, or arbitrary output."""
    try:
        result = subprocess.run([binary, "--version"], shell=False, timeout=10,
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SyncError("Не удалось определить версию OpenCode CLI") from exc
    match = re.fullmatch(r"(?:opencode\s+)?v?([12])\.\d+\.\d+(?:[-+][\w.+-]+)?", result.stdout.strip(), re.I)
    if not match:
        raise SyncError("Неизвестная версия OpenCode CLI; запись запрещена")
    return int(match.group(1))


def file_snapshot(path):
    try:
        if path.is_symlink():
            raise SyncError("Символические ссылки для входных файлов не поддерживаются")
        return path.read_bytes(), stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return None, None


def load_snapshot(path, snapshot):
    raw, _ = snapshot
    if raw is None:
        return {}
    try:
        data = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as exc:
        raise SyncError("Некорректный JSON в %s (JSONC не поддерживается)" % path) from exc
    if not isinstance(data, dict):
        raise SyncError("Корень %s должен быть объектом" % path)
    return data


def guard_files(snapshots):
    for path, before in snapshots.items():
        if file_snapshot(path) != before:
            raise SyncError("Файл изменён другим процессом: %s" % path)


def write_changes(changes, snapshots):
    """Stage unique same-directory files, guard all inputs, then replace individually.

    This is deliberately NOT a multi-file transaction or a TOCTOU-proof lock.
    """
    staged = []
    try:
        guard_files(snapshots)
        for path, data in changes:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
            staged.append((path, name))
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            mode = snapshots[path][1]
            os.chmod(name, mode if mode is not None else 0o600)
        guard_files(snapshots)
        for path, name in staged:
            # Record OUR staged bytes before replace, never adopt an external
            # writer's bytes as the new baseline in the post-replace window.
            expected = file_snapshot(Path(name))
            guard_files(snapshots)
            os.replace(name, path)
            snapshots[path] = expected
        guard_files(snapshots)
    finally:
        for _, name in staged:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass


def build_lmstudio_config_base_url(raw, default_port=1234):
    explicit_url = "://" in raw
    parsed = urllib.parse.urlsplit(raw if explicit_url else "http://" + raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise SyncError("Ожидается HTTP(S) адрес без встроенных credentials")
    if parsed.query or parsed.fragment:
        raise SyncError("Query и fragment в адресе не поддерживаются")
    host = parsed.hostname
    if ":" in host:
        host = "[" + host + "]"
    port = parsed.port if explicit_url else parsed.port or default_port
    authority = host + (":" + str(port) if port is not None else "")
    if parsed.path.rstrip("/") not in ("", "/v1"):
        raise SyncError("Path-prefixed LM Studio endpoints не поддерживаются")
    return "%s://%s%s" % (parsed.scheme, authority, parsed.path.rstrip("/") or "/v1")


def get_base_url_from_conn(conn):
    for source in (conn, conn.get("settings", {}), conn.get("options", {}), conn.get("config", {})):
        if not isinstance(source, dict):
            continue
        for key in ("baseURL", "base_url", "api", "url", "endpoint", "address", "host", "server", "server_url", "api_base_url", "api_url", "uri"):
            value = source.get(key)
            if isinstance(value, str) and value:
                if key == "host" and source.get("port"):
                    value += ":" + str(source["port"])
                parsed = urllib.parse.urlsplit(build_lmstudio_config_base_url(value))
                # LM Studio's native endpoint lives at the server root.
                return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise SyncError("Перенаправления LM Studio запрещены во избежание утечки API-ключа")


def request_models(base_url, token):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(base_url + "/api/v1/models", headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        # Never print exception text: response bodies/URLs may contain credentials.
        raise SyncError("Ошибка запроса метаданных LM Studio") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), (dict, list)):
        raise SyncError("Некорректный ответ LM Studio: ожидается объект models")
    return data


def resolve_secret(value):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\{env:([^}]+)\}", value)
    if match:
        secret = os.environ.get(match.group(1))
        if not secret:
            raise SyncError("Не задана переменная окружения для API-ключа")
        return secret
    if value.startswith("{"):
        raise SyncError("Неподдерживаемая подстановка API-ключа")
    return value or None


def extract_token_from_conn(conn):
    for source in (conn.get("settings", {}), conn.get("options", {}), conn):
        if isinstance(source, dict):
            for key in ("apiKey", "api_key", "token", "api_token"):
                if source.get(key):
                    return resolve_secret(source[key])
    return None


def extract_token_from_auth(auth, conn, base_url, provider_key=None):
    # Only exact provider IDs or exact endpoint/netloc keys; never scan unrelated credentials.
    keys = [provider_key] if provider_key else []
    if base_url:
        parsed = urllib.parse.urlsplit(base_url)
        keys.append(base_url)
        if parsed.port is not None:
            keys.append(parsed.netloc)
    for key in keys:
        entry = auth.get(key)
        if isinstance(entry, dict) and entry.get("type") == "api":
            return resolve_secret(entry.get("key"))
    return None


def extract_opencode_model_metadata(model):
    caps = model.get("capabilities", {})
    if not isinstance(caps, dict):
        return {}
    result = {}
    reasoning = caps.get("reasoning")
    if isinstance(reasoning, bool):
        result["reasoning"] = reasoning
    elif isinstance(reasoning, dict):
        allowed = reasoning.get("allowed_options")
        result["reasoning"] = (any(str(x).lower() not in {"off", "none", "disabled", "false"} for x in allowed)
                               if isinstance(allowed, list) else True)
    if isinstance(caps.get("trained_for_tool_use"), bool):
        result["tool_call"] = caps["trained_for_tool_use"]
    if isinstance(caps.get("vision"), bool):
        result["attachment"] = caps["vision"]
        result["modalities"] = {"input": ["text", "image"] if caps["vision"] else ["text"], "output": ["text"]}
    return result


def parse_models_response(resp):
    server, loaded, metadata = {}, {}, {}
    models = resp.get("models", [])
    entries = models.items() if isinstance(models, dict) else ((None, m) for m in models)
    for key, model in entries:
        if not isinstance(model, dict):
            raise SyncError("Некорректная запись модели LM Studio")
        mid = key or model.get("key") or model.get("id") or model.get("model") or model.get("name")
        if not isinstance(mid, str) or not mid:
            raise SyncError("Отсутствует ID модели LM Studio")
        if model.get("type") in ("embedding", "embeddings"):
            continue
        ctx = model.get("max_context_length") or model.get("max_context") or model.get("context_window") or model.get("context_length")
        if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
            server[mid] = ctx
        metadata[mid] = extract_opencode_model_metadata(model)
        instances = model.get("loaded_instances", [])
        for instance in instances.values() if isinstance(instances, dict) else instances:
            ctx = instance.get("config", {}).get("context_length")
            if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
                loaded[mid] = ctx
    instances = resp.get("loaded_instances", [])
    entries = instances.items() if isinstance(instances, dict) else ((None, i) for i in instances)
    for key, instance in entries:
        mid = instance.get("id") or instance.get("model") or key
        ctx = instance.get("config", {}).get("context_length")
        if mid and isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
            loaded[mid] = ctx
    return server, loaded, metadata


def get_context_for_model(connection_key, model_id, server_models, cache, loaded_instances=None):
    for source in (loaded_instances or {}, cache.get(connection_key, {}), server_models):
        if model_id in source:
            return source[model_id]
    return None


def validate_config(cfg, major):
    # Native top-level settings are not inferred from a legacy provider entry.
    native_keys = {"providers", "agents", "commands", "permissions", "plugins", "snapshots", "media", "policies", "update"}
    native_nested = (isinstance(cfg.get("model"), dict)
                     or isinstance(cfg.get("mcp"), dict) and {"servers", "timeout"}.intersection(cfg["mcp"])
                     or isinstance(cfg.get("compaction"), dict) and {"keep", "buffer"}.intersection(cfg["compaction"]))
    if major == 1 and (native_keys.intersection(cfg) or isinstance(cfg.get("skills"), list) or native_nested):
        raise SyncError("V1 CLI не может безопасно обновлять V2 конфиг")
    for name in ("provider", "providers"):
        if name in cfg and not isinstance(cfg[name], dict):
            raise SyncError("Секция %s должна быть объектом" % name)
    if set(cfg.get("provider", {})).intersection(cfg.get("providers", {})):
        raise SyncError("Одинаковые provider IDs в provider и providers")
    for section in ("provider", "providers"):
        native = section == "providers"
        for provider in cfg.get(section, {}).values():
            if not isinstance(provider, dict):
                raise SyncError("Провайдер должен быть объектом")
            forbidden = ({"npm", "options", "api", "type", "id", "base_url", "whitelist", "blacklist"}
                         if native else {"package", "settings", "canonical", "compatibility"})
            if forbidden.intersection(provider):
                raise SyncError("Смешанный формат провайдера; требуется явная миграция")
            models = provider.get("models", {})
            if not isinstance(models, dict):
                raise SyncError("models должен быть объектом; списки требуют явной миграции")
            for model in models.values():
                if not isinstance(model, dict):
                    raise SyncError("Модель должна быть объектом")
                forbidden_model = ({"id", "tool_call", "modalities", "attachment", "reasoning", "options",
                                    "release_date", "temperature", "experimental", "status", "interleaved"}
                                   if native else {"modelID", "capabilities", "settings", "package"})
                if forbidden_model.intersection(model):
                    raise SyncError("Смешанный формат модели; требуется явная миграция")
                for field in ("limit", "capabilities"):
                    if field in model and not isinstance(model[field], dict):
                        raise SyncError("%s модели должен быть объектом" % field)
                if "variants" in model and not isinstance(model["variants"], list if native else dict):
                    raise SyncError("variants не соответствует формату модели")


def update_connection_models(conn, connection_key, server_models, cache, loaded_instances, model_metadata, native=False):
    before = deepcopy(conn)
    if loaded_instances:
        cache.setdefault(connection_key, {}).update(loaded_instances)
    if "models" not in conn and not server_models and not model_metadata:
        return False, []
    models = conn.setdefault("models", {})
    # Update all aliases for a remote ID without creating a duplicate raw ID entry.
    represented = set()
    for alias, info in list(models.items()):
        remote = info.get("modelID" if native else "id", alias)
        if not isinstance(remote, str):
            raise SyncError("ID модели должен быть строкой")
        represented.add(remote)
    for mid in server_models.keys() | model_metadata.keys():
        if mid not in represented:
            if mid in models:
                raise SyncError("Remote ID конфликтует с существующим alias; модель не будет перезаписана")
            models[mid] = {"modelID" if native else "id": mid}
    for alias, info in models.items():
        remote = info.get("modelID" if native else "id", alias)
        ctx = get_context_for_model(connection_key, remote, server_models, cache, loaded_instances)
        if ctx is not None:
            info.setdefault("limit", {})["context"] = ctx
        info.setdefault("limit", {}).setdefault("output", 0)
        metadata = model_metadata.get(remote, {})
        if native:
            caps = {}
            if "tool_call" in metadata:
                caps["tools"] = metadata["tool_call"]
            caps.update(metadata.get("modalities", {}))
            if caps:
                info.setdefault("capabilities", {}).update(caps)
        else:
            info.update(deepcopy(metadata))
    return conn != before, []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--auth-path", default=DEFAULT_AUTH_PATH)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--opencode-bin", default="opencode", help="Путь к OpenCode CLI для проверки --version")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--add-connection")
    parser.add_argument("--api-key")
    parser.add_argument("--connection-name")
    parser.add_argument("--no-autoload", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        # Gate precedes ALL config/auth/cache reads, even for dry-run/no-autoload.
        major = detect_opencode_version(args.opencode_bin)
        paths = [Path(os.path.abspath(os.path.expanduser(p))) for p in (args.config_path, args.auth_path, args.cache_path)]
        if len(set(os.path.normcase(str(p.resolve())) for p in paths)) != 3:
            raise SyncError("config, auth и cache должны быть разными файлами")
        for index, path in enumerate(paths):
            for other in paths[:index]:
                if path.exists() and other.exists() and os.path.samefile(path, other):
                    raise SyncError("config, auth и cache не могут быть hardlink aliases")
        snapshots = {p: file_snapshot(p) for p in paths}
        cfg, auth, cache = [load_snapshot(p, snapshots[p]) for p in paths]
        originals = deepcopy((cfg, auth, cache))
        validate_config(cfg, major)
        for entry in auth.values():
            if not isinstance(entry, dict):
                raise SyncError("Записи auth должны быть объектами OpenCode credentials")
            if entry.get("type") == "api" and (not isinstance(entry.get("key"), str) or not entry["key"]):
                raise SyncError("API credential должен содержать непустой строковый key")
        for entry in cache.values():
            if not isinstance(entry, dict) or any(not isinstance(ctx, int) or isinstance(ctx, bool) or ctx <= 0 for ctx in entry.values()):
                raise SyncError("Некорректная структура кэша контекстов")
        added = None
        if args.add_connection:
            url = build_lmstudio_config_base_url(args.add_connection.strip())
            netloc = urllib.parse.urlsplit(url).netloc
            base = "lmstudio_" + re.sub(r"[^a-z0-9]", "_", netloc.lower())
            # Orphaned credentials still belong to the user. Never reuse their
            # IDs, even when adding without a key (which would reuse the secret).
            used = set(cfg.get("provider", {})) | set(cfg.get("providers", {})) | set(auth)
            added, index = base, 2
            while added in used:
                added, index = base + "_" + str(index), index + 1
            native = major == 2
            provider = {"name": args.connection_name or "LM Studio (%s)" % netloc, "models": {}}
            provider.update({"package": "@opencode/ai/providers/openai-compatible", "settings": {"baseURL": url}}
                            if native else {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": url}})
            cfg.setdefault("providers" if native else "provider", {})[added] = provider
            secret = args.api_key or os.environ.get("LM_API_TOKEN")
            if secret:
                auth[added] = {"type": "api", "key": secret}
        for section in ("provider", "providers"):
            for key, conn in cfg.get(section, {}).items():
                if not (key == "lmstudio" or key.startswith("lmstudio_") or conn.get("type") == "lmstudio"):
                    continue
                if key == added and args.no_autoload:
                    continue
                base_url = get_base_url_from_conn(conn)
                if not base_url and key == "lmstudio":
                    base_url = "http://127.0.0.1:1234"
                if not base_url:
                    raise SyncError("Не задан адрес LM Studio провайдера")
                token = (extract_token_from_conn(conn) or extract_token_from_auth(auth, conn, base_url, key)
                         or os.environ.get("LM_API_TOKEN"))
                response = request_models(base_url, token)
                server, loaded, metadata = parse_models_response(response)
                update_connection_models(conn, base_url, server, cache, loaded, metadata, native=section == "providers")
        validate_config(cfg, major)
        changes = [(path, data) for path, data, original in zip(paths, (cfg, auth, cache), originals) if data != original]
        if args.dry_run:
            logging.info("Dry-run: изменилось файлов: %d; запись отключена", len(changes))
        elif changes:
            write_changes(changes, snapshots)
            logging.info("Сохранено файлов: %d (OpenCode V%d)", len(changes), major)
        else:
            logging.info("Изменений нет (OpenCode V%d)", major)
        return 0
    except Exception as exc:
        # No traceback/exception payloads from network, CLI, or credentials in logs.
        if isinstance(exc, SyncError):
            logging.error("%s", exc)
        else:
            logging.error("Синхронизация остановлена (%s); проверьте права и входные данные", type(exc).__name__)
        return 2


if __name__ == "__main__":
    sys.exit(main())
