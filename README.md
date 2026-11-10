# opencode_lmsync

Синхронизация моделей LM Studio с конфигурацией **OpenCode V1 и V2**. Зависимости — только стандартная библиотека Python 3.8+.

## Безопасность и совместимость

Перед чтением **любого** config/auth/cache скрипт выполняет `opencode --version` с тайм-аутом 10 секунд, без shell. Поддерживаются `1.x.y` и `2.x.y`, включая semver prerelease. Отсутствующий CLI, ошибка запуска или неизвестная версия останавливают работу без записи. Gate обязателен также для `--dry-run` и `--no-autoload`.

**Версия CLI и формат файла независимы:**

| CLI | Существующие записи | Новое подключение |
| --- | --- | --- |
| V1 | `provider` / V1 models | `provider`, `npm`, `options` |
| V2 | V1 сохраняются V1; native V2 сохраняются V2 | `providers`, `package`, `settings` |

V1 CLI отказывается изменять обнаруженные V2-only настройки. В V2 допустимы обе top-level maps с **разными** IDs; одинаковый ID в `provider` и `providers` вызывает отказ. Вложенное смешение форматов provider/model запрещено: требуется явная миграция. Скрипт не мигрирует весь конфиг и сохраняет посторонние настройки, aliases, variants, выбранную модель и существующий `limit.output`.

Источники: [миграция V1 → V2](https://opencode.ai/v2/docs/migrate-v1), [providers](https://opencode.ai/v2/docs/providers), [models](https://opencode.ai/v2/docs/models).

## Запуск

Нужны Python, OpenCode CLI и доступ к LM Studio по HTTP(S).

```bash
python scripts/update_lmstudio_models.py --help
python scripts/update_lmstudio_models.py --dry-run
python scripts/update_lmstudio_models.py
```

Для изолированной конфигурации:

```bash
python scripts/update_lmstudio_models.py \
  --opencode-bin /path/to/opencode \
  --config-path ./fixtures/opencode.json \
  --auth-path ./fixtures/auth.json \
  --cache-path ./fixtures/cache.json \
  --dry-run
```

## Параметры

| Флаг | Значение |
| --- | --- |
| `--config-path PATH` | По умолчанию `~/.config/opencode/opencode.json` |
| `--auth-path PATH` | По умолчанию `~/.local/share/opencode/auth.json` |
| `--cache-path PATH` | По умолчанию `~/.local/share/opencode/lmstudio_context_cache.json` |
| `--opencode-bin PATH` | CLI для проверки версии; по умолчанию `opencode` |
| `--dry-run` | Выполнить чтение и запросы, не записывать ничего |
| `--verbose` | Подробность логирования; ключи и содержимое моделей не выводятся |
| `--add-connection HOST[:PORT]` | Добавить провайдер; допускается полный HTTP(S) URL |
| `--connection-name NAME` | Отображаемое имя нового провайдера |
| `--api-key KEY` | Ключ нового подключения; сохраняется только в `auth.json` |
| `--no-autoload` | Не запрашивать модели **нового** подключения; остальные обрабатываются |
| `--connection PROVIDER_ID` | Обработать только выбранное LM Studio подключение; можно повторять. Не сочетается с `--add-connection` |
| `--skip-unavailable` | Явно разрешить сохранение доступных подключений при сетевой недоступности остальных |

Добавление использует свободный ID `lmstudio_<host_port>` в обеих maps и `auth.json`, не перезаписывая и не переиспользуя существующие credentials. Для адреса без схемы порт по умолчанию — `1234`; полный HTTP(S) URL сохраняет указанный authority (без порта — стандартный порт схемы). Путь по умолчанию — `/v1`.

```bash
python scripts/update_lmstudio_models.py --add-connection 127.0.0.1:1234
python scripts/update_lmstudio_models.py --add-connection gpu.example:1234 --no-autoload
```

### Если модель не появилась из-за другого сервера

По умолчанию проход строгий: недоступность **любого** выбранного сервера отменяет сохранение всего прохода, даже если нужная модель успешно обнаружена на другом сервере. В ошибке указан provider ID. Для синхронизации только одного подключения:

```bash
python scripts/update_lmstudio_models.py --connection lmstudio_192_168_1_82_1234 --dry-run
python scripts/update_lmstudio_models.py --connection lmstudio_192_168_1_82_1234
```

Для синхронизации всех доступных серверов:

```bash
python scripts/update_lmstudio_models.py --skip-unavailable --dry-run
python scripts/update_lmstudio_models.py --skip-unavailable
```

Пропускаются только ошибки соединения/тайм-ауты. HTTP-ошибки (включая 401/403), redirects, повреждённые ответы и ошибки обработки по-прежнему отменяют запись. Если все выбранные серверы недоступны, exit code `2`, записи нет. При частичном успехе — exit code `0` и предупреждение с количеством пропущенных подключений; их модели и кэш не удаляются. Логи показывают число обнаруженных/добавленных моделей и загруженных контекстов, без credentials и содержимого моделей.

Квантизации с точными ID, например `qwen3.8-27b@iq3_xs`, `qwen3.8-27b@iq3_xxs` и `qwen3.8-27b@iq4_xs`, синхронизируются независимо. Скрипт **не загружает веса в RAM/VRAM**, а регистрирует модели и метаданные в OpenCode; `--no-autoload` относится к запросу списка нового подключения.

`--api-key` может быть виден в списке процессов/истории shell. Предпочитайте `LM_API_TOKEN` или заранее настроенный credential. В конфиге поддерживается `settings.apiKey` / `options.apiKey` с подстановкой `{env:LMSTUDIO_API_KEY}`. Подстановка должна разрешаться в непустое значение; `{file:...}` не поддерживается.

## Форматы

### V1 (сохраняется таким же и под V2 CLI)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "lmstudio_work": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://127.0.0.1:1234/v1"},
      "models": {
        "coding": {
          "id": "remote-model-id",
          "tool_call": true,
          "modalities": {"input": ["text", "image"], "output": ["text"]},
          "limit": {"context": 32768, "output": 8192}
        }
      }
    }
  }
}
```

### Native V2

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "lmstudio_work/coding",
  "providers": {
    "lmstudio_work": {
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {"baseURL": "http://127.0.0.1:1234/v1"},
      "models": {
        "coding": {
          "modelID": "remote-model-id",
          "capabilities": {"tools": true, "input": ["text", "image"], "output": ["text"]},
          "limit": {"context": 32768, "output": 8192}
        }
      }
    }
  }
}
```

Alias `coding` обновляется по `modelID` (V1: `id`), без добавления дублирующей записи remote ID. Все aliases одной модели обновляются. В native V2 скрипт не внедряет V1 `id`, `tool_call`, `modalities`, `reasoning`, `attachment`, provider `type` или служебные маркеры. Native reasoning controls не выводятся из LM Studio: существующие `capabilities.reasoning` сохраняются. В legacy V1 поля reasoning/attachment остаются V1; V2 может игнорировать их согласно migration guide.

## Алгоритм

1. Проверить версию CLI; сделать snapshots исходных bytes и modes всех трёх файлов.
2. Строго прочитать JSON-объекты: отсутствие файла допустимо, повреждение — ошибка до записи. Проверить provider/model maps и структуру кэша.
3. Найти `lmstudio` / `lmstudio_*` в обеих maps даже без `type`, а также V1 entries с `type: lmstudio`. Рекурсивного поиска в произвольных настройках нет.
4. Определить endpoint; для `lmstudio` без адреса используется `http://127.0.0.1:1234`. Запросить `GET /api/v1/models` на корне сервера.
5. Токен: явно заданный в подключении → точный provider ID в auth → точный endpoint/netloc в auth → `LM_API_TOKEN`. Credential другого провайдера, совпадение подстроки или только hostname **не используются**. OAuth credential не используется как API-ключ. HTTP redirects запрещены.
6. Контекст: точное совпадение в loaded instances → кэш → максимальный серверный контекст. Fuzzy-сопоставления нет; embedding-модели исключаются. `limit.output = 0` добавляется только при отсутствии поля. Известные vision/tool capabilities обновляются; прочие поля сохраняются.
7. После успешной обработки всех выбранных подключений сохранить действительно изменённые файлы. По умолчанию ошибка запроса/разбора/обработки прерывает проход с ненулевым exit code; частичные in-memory изменения не сохраняются. Явный `--skip-unavailable` допускает сетевые сбои отдельных серверов, если хотя бы один успешно обработан.

## Ограничения

- Поддерживается **строгий JSON**, не JSONC: комментарии, trailing commas, повторяющиеся ключи, NaN/Infinity вызывают безопасный отказ. Комментарии не удаляются автоматически.
- Списки `models` и смешанные вложенные формы требуют явной миграции. Проверяется используемая структура, а не вся схема OpenCode.
- Перед записью повторно проверяются **все три** входных файла (bytes/mode), даже если меняется лишь кэш. Обнаруженное внешнее изменение прекращает запись.
- Уникальные temporary files в том же каталоге, fsync и atomic replace каждого файла. Существующие permission bits сохраняются; новые файлы получают `0600` на POSIX. На Windows это не гарантия сохранения индивидуальных ACL; их следует контролировать отдельно.
- Это **не** межфайловая транзакция и **не** устранение TOCTOU. Ошибка/внешнее изменение между отдельными replace может оставить уже заменённый файл; завершение будет неуспешным. Автоматического rollback/backup нет; перед первым применением сохраните резервную копию.
- Symlink входных файлов запрещён; одинаковые resolved paths и hardlink aliases config/auth/cache отклоняются. При содержательном изменении JSON форматируется заново; неизменённые файлы не переписываются.
- URL с credentials/query/fragment запрещены. Native API запрашивается от корня хоста; path-prefixed reverse proxy отклоняется до запроса, чтобы не отправить credential в другой path. Для удалённых серверов используйте HTTPS.
- V2 уже имеет встроенный LM Studio discovery. Этот инструмент нужен для явных overrides/кэша, а не для обязательной настройки V2.

Exit code: `0` — успех/нет изменений/dry-run (в том числе частичный проход с явным `--skip-unavailable`); `2` — отказ, ошибка обработки/записи или все выбранные серверы недоступны.

## Изолированные тесты

```bash
python -m unittest discover -s tests -v
```

Тесты создают временные config/auth/cache, подставляют version responses и запускают subprocess CLI с fake OpenCode и HTTP fixture на loopback. Реальные пользовательские конфиги и production endpoints не затрагиваются.
