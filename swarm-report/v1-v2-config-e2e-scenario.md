# E2E Scenario: безопасная синхронизация V1/V2
Платформы: Python unittest и изолированный CLI; разрешены пользователем вместо prod.
Все данные — временные fixtures; реальные конфиги и серверы не используются.

## Шаги
- [x] 1. Запустить unittest проверки CLI version gate: версия определяется до чтения файлов; неизвестный/missing/timeout CLI не меняет файлы. ✅
- [x] 2. Проверить V1, native V2, V1 entries в V2 и смешанные top-level maps; вложенные поля остаются в своём формате. ✅
- [x] 3. Проверить отказ V1 на V2, конфликты IDs, nested mixing и повреждённые config/auth/cache без записи. ✅
- [x] 4. Проверить aliases, capabilities, output limits, variants, новые подключения и отсутствие служебных полей. ✅
- [x] 5. Проверить dry-run, concurrency guard, cache-only, permissions, idempotence и выбор credentials без утечки чужих ключей. ✅
- [x] 6. Запустить изолированный subprocess CLI с fake OpenCode и локальным HTTP fixture; проверить код завершения и native V2 результат. ✅
- [x] 7. Выполнить финальные unittest, проверку diff и сохранить результат в отчёт. ✅ Финализация 2026-11-10: 35 tests, OK (0.946s); git diff --check: exit 0.
