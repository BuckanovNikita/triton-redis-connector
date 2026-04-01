# Финальный план: triton-redis-connector

## Контекст

Проект — монорепозиторий с 3 пакетами (`triton_redis_connector`, `triton_model_tools`, `triton_shared_models`), реализующий граф-ориентированный пайплайн ML-инференса (Redis Streams -> Triton -> Redis Streams). Кодовая база достигла фазы 5 единого плана с рабочим ядром. Данный план охватывает 26 задач до продакшн-качества: очистка тестов, полнота инструментов, контракты, упрощение примера, E2E-верификация.

Ключевой принцип: **используем собственные инструменты для верификации результатов — расширяем функциональность инструментов, а не делаем напрямую**.

---

## Фаза 0 — Очистка, актуализация документации, определения агентов

### 0.1 Удаление устаревших примеров (задача 23)
- `examples/camera_inference/` и `examples/yolo_detection_pipeline/` уже удалены в рабочем дереве — коммит удаления
- Проверить отсутствие ссылок в конфигах (pyrightconfig.json, pyproject.toml per-file-ignores, import-linter)

### 0.1a Реверс зависимостей пакетов
Текущее (неверно):
- `triton_shared_models` → ничего (leaf)
- `triton_redis_connector` → `triton-shared-models`
- `triton_model_tools` → `triton-shared-models` (НЕ зависит от connector)

Целевое:
- `triton_shared_models` → ничего (leaf)
- `triton_redis_connector` → `triton-shared-models`
- `triton_model_tools` → `triton-redis-connector` + `triton-shared-models`

**Изменения:**
- **`packages/triton_model_tools/pyproject.toml`**: добавить `triton-redis-connector` в dependencies
- **`packages/triton_model_tools/pyproject.toml` [tool.uv.sources]**: добавить `triton-redis-connector = { workspace = true }`
- **`pyproject.toml` (root)**: удалить import-linter контракт `model-tools-no-connector` (#4) — model_tools ДОЛЖЕН импортировать connector
- **Сохранить** контракт `connector-no-model-tools` (#3) — connector НЕ должен импортировать model_tools
- **Сохранить** контракт `shared-models-independence` (#5)

**Почему**: model_tools нужен доступ к:
- Shipped процессорам connector (EnvelopeExtractor, EventSerializer, etc.)
- Pipeline framework (PipelineRegistry, Application) для E2E тестирования
- Preprocessing (LetterBoxPreprocessor, decoders) уже переиспользуется через shared_models, но PostProcessor и InferenceWorker нужны напрямую
- Inference tool как упрощённый app instance

### 0.2 Актуализация документации (задачи 19, 25)
- **`CLAUDE.md`** (корень): указано "два пакета" — обновить до трёх (`triton_shared_models`)
- ~~`docs/design.md`~~: удалён как устаревший — новый архитектурный документ будет создан в фазе 8.4
- **`docs/unified-plan.md`**: заменить на ссылку на этот план; фазы 0-5 как завершённые
- **`README.md`**: проверить таблицу метрик vs аннотации Counter/Gauge/GroupBy в примере; обновить секцию примеров
- Вся документация на русском (кроме CLAUDE.md)

### 0.3 Обновление определений агентов
- `.claude/agents/architect.md`, `implementer.md`, `tester.md` — трёхпакетная структура, текущий список задач, подход tool-first

### 0.4 Верификация
- `uv run pre-commit run --all-files` проходит
- Нет висячих ссылок на удалённые примеры

---

## Фаза 1 — Аудит тестов и удаление антипаттернов (задача 1)

Аудит всех тестовых файлов по Testing Policy. Удалить тесты, проверяющие:
- Гарантии фреймворка (frozen pydantic, enum string coercion)
- Конструирование/дефолты (создание объекта, assert полей == аргументам конструктора)
- Protocol isinstance проверки (mypy/pyright уже верифицируют)
- Количества констант (`assert len(REGISTRY) == N`)
- Содержимое генерируемых файлов (pbtxt, config content)

### Известные кандидаты
- `packages/triton_shared_models/tests/` — проверить на default-value-only assertions
- `packages/triton_model_tools/tests/test_registry.py` — default value assertions (сохранить тесты валидации)
- `packages/triton_model_tools/tests/test_nms.py`, `test_dali.py` — проверить на content-matching

### Верификация
- `uv run pytest` проходит после очистки
- Реальная логика не осталась без тестов

---

## Фаза 2 — Корректность экспорта моделей (задачи 2, 13, rect, shapes)

### 2.1 Исправление бага rect
**Баг**: `converter.py:61-62` — `if rect: merged_kwargs["rect"] = True` никогда не устанавливает `rect=False`
- **Файл**: `packages/triton_model_tools/triton_model_tools/export/converter.py`
- **Исправление**: `merged_kwargs["rect"] = rect` (всегда передавать значение)
- **Также**: добавить `--no-rect` в `cli/export.py` (сейчас только в `cli/convert.py`)
- Приоритет: CLI `--no-rect` > YAML config `rect:` > модель по умолчанию (`True`)

### 2.2 Исправление игнорирования imgsz в DALI конфиге
**Баг**: CLI создают `DaliPipelineConfig()` с захардкоженным `(640, 640)` вместо `imgsz` модели.
- **Файлы**: `cli/convert.py`, `cli/export.py`, `cli/populate_repository.py`
- **Исправление**: `target_size=variant.resolved_imgsz` при создании `DaliPipelineConfig` для каждой модели
- **Критично для 960x608**: без фикса DALI создаёт тензоры `[1,3,640,640]` для модели ожидающей `[1,3,608,960]`

### 2.2a Удаление batch export инфраструктуры
- **Удалить файлы**:
  - `cli/export.py` — batch export CLI
  - `cli/populate_repository.py` — batch export + DALI/NMS generation CLI
  - `export/exporter.py:export_all_models()` (lines 325-388) — batch export функция
- **Обновить** `__main__.py`: удалить import и регистрацию `export` и `populate-repository` субкоманд
- **Сохранить** `validate_repository()` из `export/validate.py` — добавить как отдельную CLI субкоманду `validate` (полезна standalone)
- **Единственный путь экспорта**: `convert` CLI, одна модель за раз
- rect override через `--no-rect` применяется к единственной экспортируемой модели

### 2.3 Локальный ONNX inference инструмент (задача 2)
- **Создать**: `inference/onnx_inference.py` — `OnnxInferenceClient(onnx_path)` с `infer(tensor) -> dict[str, NDArray]`
- **Создать**: `cli/local_infer.py` — CLI субкоманда `local-infer`
- Использует те же `YoloPreprocessor` и `YoloPostprocessor` что и pipeline
- Расширить `__main__.py` для регистрации новой субкоманды

### 2.4 Верификация FP16/FP32 (задача 13)
- Флаг `local-infer`: экспорт одной модели в FP16 и FP32, запуск обоих, сравнение
- Использовать `export_model()` + `OnnxInferenceClient` — подход tool-first
- Детекции должны совпадать в пределах floating-point tolerance

### 2.5 Верификация валидации разрешений
- **Валидный нестандартный**: `512x512` — делится на 32, должен работать end-to-end
- **Невалидный**: `601x612` — НЕ делится на 32 — должен вызвать `ValueError`
- **Проверить через инструмент**: `triton-model-tools convert --imgsz 601x612` — падает с ясной ошибкой
- **Проверить через инструмент**: `triton-model-tools convert --imgsz 512` — успешно, корректные конфиги
- **Тест**: параметризованный — валидные (640, 512, 960x608, 512x512) и невалидные (601x612, 639x640) размеры
- **Проверить DALI конфиг** для 512x512 (не захардкоженный 640x640)

### 2.6 Проверка что end2end NMS предотвращает двойной NMS
- **Уже корректно**: `convert.py:175` ставит `effective_nms = None if end2end`, `export.py:182` проверяет `end2end_names`
- **Добавить тест**: при `end2end=True` + `--nms --dali-ensemble` — ensemble без NMS шага

### 2.7 Верификация всех вариантов экспорта на разных размерах
Каждый вариант экспорта должен корректно работать с каждым размером:

| Вариант | 640x640 | 960x608 | 512x512 |
|---------|---------|---------|---------|
| ONNX FP32 | convert + local-infer | convert + local-infer | convert + local-infer |
| ONNX FP16 | convert + local-infer | convert + local-infer | convert + local-infer |
| ONNX + TRT optimize | convert --trt-optimize | convert --trt-optimize | convert --trt-optimize |
| TensorRT FP16 | convert -f tensorrt | convert -f tensorrt | convert -f tensorrt |
| ONNX + DALI ensemble | convert --dali-ensemble | convert --dali-ensemble | convert --dali-ensemble |
| ONNX + DALI + NMS | convert --dali --nms | convert --dali --nms | convert --dali --nms |
| ONNX + DALI + rescale | convert --dali --rescale | convert --dali --rescale | convert --dali --rescale |

Проверить для каждой ячейки:
- Config.pbtxt содержит правильные dims
- DALI config (если есть) содержит правильный target_size
- `local-infer` (или `infer` для Triton) выдаёт детекции
- Невалидный размер (601x612) — ошибка на всех вариантах

**Модели для верификации экспорта**: yolo11s, yolo12s, yolo26l (все 3). Минимум: каждая модель × ONNX FP32 × 640 + одна нестандартная комбинация.

Дополнительно:
- `convert --no-rect` → `export_kwargs["rect"] == False`
- FP16 и FP32 детекции в пределах tolerance

---

## Фаза 3 — Улучшения коннектора (задачи 5, 15, 17)

### 3.1 Режимы Redis коллектора (задача 5)
- **Enum** `CollectorMode` в `config/schema.py`: `RECOVERY` (текущий), `FROM_LATEST`, `FROM_OLDEST`
- **Изменить** `consumer_group.py:ensure_group_exists()` — параметр start ID (`$` для FROM_LATEST, `0` для остальных)
- **Изменить** `redis_stream.py:RedisStreamSource` — пропуск pending recovery для `FROM_LATEST`
- **Конфиг**: `collector_mode` в `SourceConfig`

### 3.2 Per-model sink streams (задача 15)
- `OutputStreamRouter` уже существует в `connectors/output_router.py`
- **Проверить**: ясная ошибка при отсутствии matching stream И default_stream
- **Добавить**: startup валидация — проверка model_streams до первого события
- **Интегрировать** в пример для демонстрации per-model routing
- **Конфиг**: `model_streams` в `SinkConfig`

### 3.3 Инструмент наполнения Redis (задача 17)
- **Создать**: `cli/populate_stream.py` — `populate-stream --stream NAME --image-dir DIR --event-type TYPE`
- Читает изображения, создаёт события по схеме входного типа
- XADD с полями: `camera_no`, `image_path`, `frame_counter`, `receipt_ts`, `metal_type`
- Верификация: прочитать обратно и проверить полноту полей

### 3.4 Верификация
- Каждый режим с реальным Redis (`@pytest.mark.integration`)
- Per-model routing: модель A → stream A, модель B → stream B
- Population tool: push N событий, read back, проверка полноты

---

## Фаза 4 — Верификация preprocessing (задачи 4, 7, 8, 9)

### 4.1 DALI vs ultralytics preprocessing (задача 4)
- Сравнить `run_dali_preprocess()` (dali.py:516-610) vs `LetterBoxPreprocessor.preprocess()`
- Тест на `test_data/coco8/` с квадратными (640x640) и прямоугольными (960x608) размерами
- Сравнение tensor shapes (точное совпадение) и pixel values (max abs diff < 2.0/255.0)
- Метка `@pytest.mark.gpu`

### 4.2 Shipped decoders и публичные экспорты (задача 7)
- **Connector `__init__.py`**: добавить re-export `OpenCVDecoder`, `TurboJPEGDecoder`, `ImageDecoderProtocol`, `LetterBoxPreprocessor`, wrappers (`DetachedThreadProcessor`, etc.) — сейчас НЕ экспортируются из корня пакета
- **Model tools `__init__.py`**: добавить re-export `TritonInferenceClient`, `YoloPreprocessor`, `YoloPostprocessor` — сейчас пустой
- DALI decoder не нужен standalone — серверная сторона через ensemble
- Проверить thread safety `TurboJPEGDecoder` (использует `threading.local()`)

### 4.3 Очистка NMS (задача 8)
Только `torchvision.ops.nms` — numpy NMS удалить полностью.
- **Перенести NMS в `triton_shared_models`**: публичная функция `nms()` использующая `torchvision.ops.nms`
- **Добавить `torch` + `torchvision` в зависимости** `triton_shared_models/pyproject.toml`
- **Удалить** `_nms_numpy()` отовсюду:
  - `triton_model_tools/inference/postprocessing.py:19-68` → импорт из `triton_shared_models`
  - `examples/dual_model_pipeline/processors.py:335-374` → импорт из `triton_shared_models`
  - `examples/dual_model_pipeline/test_deterministic.py:~200+` → импорт из `triton_shared_models`
- **Серверный NMS** (`scripts/nms_postprocess.py:44`) уже использует `torchvision.ops.nms` — корректно
- **Тест**: проверить что NMS работает на CPU и GPU, результаты детерминированы

### 4.4 Верификация rescaling (задача 9)
- Серверная: `rescale_postprocess.py` — `(coord - pad) / scale`
- Клиентская: `YoloPostprocessor._postprocess_*` — та же формула
- **Проблема**: `export.py:136-137` предупреждает но не ошибается при отсутствии `--rescale`
- **Исправление**: `--strict` флаг — ошибка при отсутствии rescale для non-end2end
- **Тест**: roundtrip — preprocess, inference, rescale, проверка маппинга в оригинальные координаты
- **Через инструменты**: расширить `local-infer` для rescaling и проверки диапазонов координат

### 4.5 Верификация
- DALI тест проходит на GPU
- NMS дедупликация — единый источник для клиентского NMS
- Rescale roundtrip тест проходит

---

## Фаза 5 — Улучшения pipeline framework (задачи 3/12, 6, 10, 20, 21, 22)

### 5.1 Output = input + detection_results (задачи 3, 12)
Требование: output событие содержит ВСЕ поля input события + поле с результатами детекции. Все атрибуты входного события пробрасываются через весь пайплайн от начала до конца.

**Текущие проблемы (3 штуки):**

1. **`InspectionResult` не содержит `receipt_ts`, `image_path`** из `SteelInspectionInput`:
   - Input: `receipt_ts`, `camera_no`, `image_path`, `frame_counter`, `metal_type`
   - Output: `camera_no`, `metal_type`, `frame_counter` + detection fields — **пропущены** `receipt_ts`, `image_path`

2. **`OutputSerializer` создаёт новый Envelope** (`processors.py:307`):
   ```python
   return Envelope(payload=payload)  # ← original_fields ПОТЕРЯНЫ
   ```
   Должен использовать `derive()` или получать Envelope от framework. Сейчас `OutputSerializer.__call__` принимает `InspectionResult` (не `Envelope[InspectionResult]`), поэтому не имеет доступа к `original_fields`.

3. **Промежуточные события теряют поля**: `EvenCameraFrame` и `OddCameraFrame` имеют `image_bytes`, `camera_no`, `frame_counter`, `metal_type` — но **не** `receipt_ts`, `image_path`. Эти поля нужно пробрасывать через все промежуточные типы.

**Решение — два уровня:**

**Уровень A: Данные события** — все промежуточные и финальный тип содержат ВСЕ поля входного типа:
- `EvenCameraFrame` / `OddCameraFrame`: добавить `receipt_ts: float`, `image_path: str`
- `MetadataInferenceInput` / `MetadataInferenceOutput`: добавить `receipt_ts: float`, `image_path: str`
- `InspectionResult`: добавить `receipt_ts: float`, `image_path: str`
- Каждый процессор пробрасывает эти поля при создании нового события

**Уровень B: Envelope original_fields** — framework гарантирует проброс:
- **Проблема**: inline dispatcher передаёт PAYLOAD в processor, не Envelope (app.py:174). EventSerializer НЕ МОЖЕТ получить Envelope через обычный type-based routing
- **Решение**: расширить `EventContext` (engine/context.py:14-22) — добавить `original_fields: dict[str, str]` — framework уже устанавливает контекст через contextvars перед диспатчингом (app.py:159-165)
- Shipped `EventSerializer[T]` читает `current_context().original_fields` и создаёт `Envelope(payload=serialized_bytes, original_fields=ctx.original_fields, source_stream=ctx.source_stream, message_id=ctx.message_id)`
- Sink пишет: `original_fields` (все входные поля как Redis fields) + `data` (сериализованный output event)
- **Результат в Redis**: каждое поле входного события доступно И как Redis field, И внутри data JSON

**Верификация**:
- Push `SteelInspectionInput` с `receipt_ts=1.0, camera_no=1, image_path=/img/1.jpg, frame_counter=42, metal_type=steel`
- Прочитать output stream
- Проверить: Redis message содержит `receipt_ts`, `camera_no`, `image_path`, `frame_counter`, `metal_type` + `data` с полным `InspectionResult` включающим ВСЕ эти поля + `detections_json`

### 5.2 Типизация register() и wrappers для статической проверки (задачи 6, 10)
**Проблема найдена**: `register(name, node: object)` и wrappers принимают `inner: object` — mypy/pyright НЕ МОГУТ поймать ошибки типов в пользовательских процессорах. Базовые классы НЕ нужны (protocol-based structural subtyping), но нужны typed overloads.

- **Файл**: `engine/registry.py` — добавить `@overload` для `register()`:
  ```
  @overload
  def register(self, name: str, node: Source[TOut]) -> None: ...
  @overload
  def register(self, name: str, node: Processor[TIn, TOut]) -> None: ...
  @overload
  def register(self, name: str, node: Sink[TIn]) -> None: ...
  ```
- **Файл**: `engine/wrappers.py` — типизировать `inner` в каждом wrapper:
  - `DetachedThreadSource[TOut].__init__(inner: Source[TOut])` вместо `inner: object`
  - `DetachedThreadProcessor[TIn, TOut].__init__(inner: Processor[TIn, TOut])` вместо `inner: object`
  - и т.д. для всех 6 wrappers
- **Верификация**: создать намеренно ошибочные процессоры → mypy/pyright должны поймать:
  - Неверный тип `__call__` параметра
  - Отсутствующий метод `__call__`
  - Неверный return type
  - Неверный layer import (import-linter уже ловит)

### 5.4 Документация параллелизма (задача 20)
- Decision tree выбора вариантов в README или docs/
- Когда использовать ThreadProcessor vs ProcessPoolProcessor vs inline
- Примеры конфигурации для каждого варианта

### 5.5 Нет main-thread blocking процессоров (задача 21)
- **Найдено**: `MetadataInferenceWorker` зарегистрирован inline без wrapper (`run_pipeline.py:88-91`) — делает blocking gRPC вызовы
- **Исправить**: обернуть в `DetachedThreadProcessor(MetadataInferenceWorker(...))`
- **Проверить** остальные register() calls — все остальные уже обёрнуты (confirmed)

### 5.6 Чтение изображений из filesystem (задача 22)
- `SteelInspectionInput.image_path` — пример ДОЛЖЕН читать из filesystem (камера пишет файл → путь в Redis → connector читает файл)
- Текущее: `Path.read_bytes()` в `CameraBranchRouter` — корректно
- **Убедиться**: shipped `ClientPreprocessor` (из 6.1) принимает `image_path: str` и читает из filesystem
- **Shipped `DaliInferenceInputBuilder`** тоже должен уметь читать bytes из filesystem path
- Абстракция `ImageReader` protocol не нужна — filesystem единственный вариант

### 5.7 Верификация
- Output Redis сообщения содержат все input поля + результаты
- Нет inline процессоров с blocking I/O
- mypy/pyright ловят ошибки типов в пользовательских процессорах

---

## Фаза 6 — Упрощение примера (задачи 11, 14, 16, 24)

### 6.1 Стандартные shipped процессоры (задача 11)
Все 6 поставляются одновременно. **Важно**: connector НЕ может импортировать model_tools (контракт #3). Поэтому:
- В **connector** (5 штук): `EnvelopeExtractor[T]`, `DaliInferenceInputBuilder`, `ClientPreprocessor`, `EventSerializer[T]`, `LogErrorSink`
- В **model_tools** (1 штука): `YoloPostProcessorBridge` — bridge `YoloPostprocessor` в connector pipeline (InferenceOutput → DetectionResult). Живёт в model_tools т.к. зависит от `YoloPostprocessor` из model_tools, а model_tools зависит от connector

Подробно:
- **`EnvelopeExtractor[T]`** — generic: `Envelope[bytes]` -> typed event T (connector)
- **`DaliInferenceInputBuilder`** — bytes → `InferenceInput` для DALI ensemble (connector)
- **`ClientPreprocessor`** — decode + letterbox + wrap → `InferenceInput` (connector)
- **`EventSerializer[T]`** — принимает `T` (payload), сериализует в bytes, создаёт `Envelope[bytes]` с `original_fields` из `current_context()` → **сохраняет original_fields** (connector)
- **`LogErrorSink`** — логирование ошибок (connector)
- **`YoloPostProcessorBridge`** — InferenceOutput → DetectionResult с NMS + rescale (model_tools)

### 6.2 Проверка shared models (задача 14)
С новой схемой зависимостей (tools → connector → shared):
- `LetterBoxPreprocessor`: используется connector (preprocessing/letterbox.py) и tools (inference/preprocessing.py) — ДОЛЖЕН быть в shared
- `TritonDataType`, `TritonTensorSpec`, `TritonModelConfig`: используются tools для export — ДОЛЖНЫ быть в shared (connector не использует, но tools через connector не может тянуть export types)
- Типы используемые ТОЛЬКО connector: проверить что не дублируются в shared
- Типы используемые ТОЛЬКО tools: оставить в tools, НЕ тянуть в shared

### 6.3 Верификация контракта метрик (задача 16)
- Сравнить таблицу метрик README vs аннотации в `events.py`
- Обновить README при расхождениях

### 6.4 Финальный пример (задача 24)
- Пользователь определяет ТОЛЬКО: `events.py`, `CameraBranchRouter`, `config.yaml`, `run_pipeline.py`
- Всё остальное — imported из framework

### 6.5 Верификация
- Пример работает с минимальным пользовательским кодом
- Метрики соответствуют документации

---

## Фаза 7 — Бенчмарки (вне контейнера, лимит 20GB RAM + VRAM)

**Важно**: бенчмарки запускаются на ХОСТЕ, не в devcontainer. Triton и Redis на хосте через docker-compose.debug.yml (порты exposed).

### 7.1 TRT-оптимизация для ONNX моделей
Это НЕ отдельный формат экспорта — это Triton config optimization для ONNX моделей. ONNX файл остаётся тем же, но Triton использует TensorRT execution provider вместо дефолтного onnxruntime.
- **Расширить `OptimizationConfig`** в `triton_shared_models/triton_config.py:50-54`: добавить поле `trt_acceleration: bool = False`
- **Расширить Jinja шаблон** `config.pbtxt.j2`: при `trt_acceleration=True` рендерить:
  ```
  optimization {
    execution_accelerators {
      gpu_execution_accelerator: [{ name: "tensorrt" }]
    }
  }
  ```
- **Добавить в convert CLI**: `--trt-optimize` флаг — при экспорте ONNX добавляет TRT optimization в config.pbtxt
- **НЕ новый ExportFormat** — ONNX файл и platform остаются `onnxruntime_onnx`, меняется только optimization section
- **Бенчмарк**: `benchmark --model yolo11s --trt-optimize` — сравнить latency чистого ONNX vs ONNX+TRT vs TensorRT engine
- **Проверить**: при первом inference Triton строит TRT engine из ONNX (медленный warmup), затем inference быстрый

### 7.2 Мониторинг памяти в бенчмарках
Текущее: бенчмарки измеряют только latency/throughput через `time.perf_counter()`. Нет мониторинга RAM/VRAM.
- **Добавить**: `nvidia-ml-py` (pynvml) для GPU VRAM мониторинга
- **Добавить**: `psutil` для системного RAM мониторинга
- **Записывать** перед и после каждого бенчмарка: peak VRAM, текущий VRAM, системный RAM
- **Добавить поля** в `BenchmarkResult`: `peak_vram_mb`, `peak_ram_mb`
- **Лимит**: если VRAM + RAM > 20GB — остановить бенчмарк с предупреждением, не загружать следующую модель
- **CLI флаг**: `--max-memory-gb 20` (дефолт: 20)

### 7.3 Бенчмарк по одной модели
Текущее: `full-benchmark` создаёт матрицу N моделей × 4 конфига = переполнение GPU.
- **Удалить** `full-benchmark` CLI (batch не нужен, см. фазу 2.2a)
- **Единственный путь**: `benchmark` CLI — одна модель, один backend, один прогон
- **Поток**: convert модель → загрузить в Triton (explicit mode) → benchmark → выгрузить → следующая
- **Тестовые размеры**: 640x640, 960x608, 512x512

### 7.4 perf_analyzer как optional dependency
- `perf_analyzer` — бинарник из Triton SDK, доступен только внутри `triton-sdk` контейнера или при установке на хост
- **Сделать опциональным**: `run_perf_analyzer()` должен проверять наличие бинарника, при отсутствии — `raise ImportError` с ясным сообщением
- **CLI**: `triton-model-tools perf-analyze` — работает только при наличии бинарника, при отсутствии — ясная ошибка
- **Тесты**: не должны падать если `perf_analyzer` отсутствует — пропускать через `pytest.mark.skipif`
- **Проверить**: что импорт `benchmarks.perf_analyzer` модуля не падает при отсутствии бинарника (сейчас модуль импортируется нормально, subprocess вызывается только при run)

### 7.5 Матрица бенчмарков и тестов
Тесты ДОЛЖНЫ покрывать три модели: **yolo11s**, **yolo12s**, **yolo26l** — разные поколения YOLO, разные размеры. Плюс yolo26s для end2end. Запускать последовательно, по одной (лимит 20GB).

**Матрица бенчмарков** (на хосте):

| Модель | Формат | imgsz | Метрики |
|--------|--------|-------|---------|
| yolo11s | ONNX FP32 | 640 | latency, throughput, VRAM, RAM |
| yolo11s | ONNX FP16 | 640 | latency, throughput, VRAM, RAM |
| yolo11s | ONNX + TRT optimize | 640 | latency, throughput, VRAM, RAM |
| yolo11s | TensorRT FP16 | 640 | latency, throughput, VRAM, RAM |
| yolo11s | ONNX FP32 | 960x608 | latency, throughput, VRAM, RAM |
| yolo11s | ONNX + DALI | 640 | latency, throughput, VRAM, RAM |
| yolo12s | ONNX FP32 | 640 | latency, throughput, VRAM, RAM |
| yolo12s | ONNX + TRT optimize | 640 | latency, throughput, VRAM, RAM |
| yolo12s | TensorRT FP16 | 640 | latency, throughput, VRAM, RAM |
| yolo26l | ONNX FP32 | 640 | latency, throughput, VRAM, RAM |
| yolo26l | ONNX FP16 | 640 | latency, throughput, VRAM, RAM |
| yolo26l | TensorRT FP16 | 640 | latency, throughput, VRAM, RAM |
| yolo26s (end2end) | ONNX FP32 | 640 | latency, throughput, VRAM, RAM |

**Матрица drift-тестов** (local ONNX vs Triton, через инструменты):

| Модель | Проверка |
|--------|----------|
| yolo11s 640 | local-infer vs infer → детекции совпадают |
| yolo11s 960x608 | local-infer vs infer → детекции совпадают на прямоугольном размере |
| yolo12s 640 | local-infer vs infer → детекции совпадают (другая архитектура) |
| yolo26l 640 | local-infer vs infer → детекции совпадают (large модель) |
| yolo26s 640 (end2end) | local-infer vs infer → NMS встроен, нет двойного NMS |

**yolo26l специфика**: large модель, может занимать 2-4GB VRAM. Обязательно выгрузить перед следующей моделью. При TRT optimize — первый warmup inference строит TRT engine (может занять 30-60 секунд и пиковый VRAM).

Между моделями: выгрузить через Triton API, подождать освобождения VRAM, проверить `peak_vram_mb`.

### 7.6 Верификация
- Каждый бенчмарк завершается без OOM
- Peak VRAM + RAM < 20GB на каждом прогоне
- Результаты сохраняются в CSV/JSON
- `perf_analyzer` при отсутствии не ломает ничего (тесты, import, другие CLI)
- ONNX-TRT вариант даёт промежуточные результаты между ONNX и TRT

---

## Фаза 8 — Очистка линтеров, E2E тестирование, финальная верификация (задачи 18, 26)

### 8.1 Минимизация linter ignores (задача 18)
Текущее: 75 suppressions в 36 файлах. Три стратегии сокращения:

**Стратегия A: Настройка pyright на актуальные библиотеки (~20 suppressions)**
Pyright может анализировать реальный код библиотек если знает где они лежат. Настроить `pyrightconfig.json`:
- `extraPaths`: указать пути к установленным пакетам в `.venv/lib/python3.12/site-packages/`
- Или `venvPath` + `venv`: указать `.venv` — pyright сам найдёт пакеты

Библиотеки для подключения:
| Библиотека | Suppressions | Тип проблемы |
|-----------|-------------|--------------|
| `tritonclient.grpc` | 3 | reportMissingTypeStubs |
| `turbojpeg` | 5 | reportMissingTypeStubs + unknown types |
| `ultralytics` | 3 | attr-defined |
| `cv2` | 4 | assignment (return type) |
| `omegaconf` | 2 | assignment |
| `redis` (xack) | 3 | no-untyped-call (types-redis stub уже в dev deps) |

Действие: добавить в `pyrightconfig.json`:
```json
{
  "venvPath": ".",
  "venv": ".venv"
}
```
Проверить: какие `type: ignore` / `pyright: ignore` можно убрать после того как pyright увидит реальный код библиотек.

**Стратегия B: Typed overloads для wrappers/app (~15 suppressions)**
Покрывается фазой 5.2 — `register(node: object)` → typed overloads, `inner: object` → typed parameters. Это уберёт `type: ignore[operator]` и `type: ignore[return-value]` из wrappers.py и app.py.

**Стратегия C: Тестовые suppressions (keep ~7)**
Тестовые файлы с mock-патчами (`type: ignore[attr-defined]`, `type: ignore[call-arg]`) — оставить, это intentional.

**Сохранить**: `PLC0415` (lazy imports в CLI), `TC002` (pydantic runtime), `nvidia.dali` (optional, не установлен)
- Цель: с 75 до ~20 (оставляем тесты + optional deps + intentional casts)

### 8.2 FiftyOne интеграция и визуальная верификация drift
- **Баг**: `fiftyone_export.py:44-45` нормализует bbox используя `max(det.x2, 1.0)` как ширину вместо реальных размеров
- **Новая архитектура**: FiftyOne принимает ДВА набора YOLO txt директорий (local-infer + Triton infer) и создаёт ОДИН FiftyOne dataset с двумя полями детекций:
  - `local_detections` — результаты local ONNX inference
  - `triton_detections` — результаты Triton inference
  - Это позволяет визуально сравнить drift между двумя backend'ами
- **Вход**: две директории с YOLO-format txt файлами + директория изображений
- **CLI**: `triton-model-tools compare-visual --local-dir results/local --triton-dir results/triton --image-dir test_data/coco8 --dataset-name drift-check`
- **Исправить нормализацию bbox**: читать размеры изображений из filepath при экспорте (изолированное изменение, не трогаем `ImageDetectionResult`)
- Mock тесты должны проверять корректность нормализации bbox

### 8.3 E2E тестовые сценарии (задача 26)
Полный pipeline тест через CLI инструменты:

**Ограничения**:
- Загрузка многих моделей переполняет GPU — тестировать ПО ОДНОЙ
- Использовать `triton-model-tools` CLI для всех шагов
- Метка `@pytest.mark.integration`
- **Triton в explicit mode**: создать `.devcontainer/docker-compose.test.yml` с `--model-control-mode=explicit` (dev compose остаётся с `poll`). E2E тесты запускают Triton через test compose

**Поток на модель**:
1. `triton-model-tools convert -w yolo.pt -r /models --imgsz 640 --dali-ensemble --nms --rescale`
2. Загрузить модель через Triton Model Control API (`POST /v2/repository/models/{name}/load`)
3. Ожидание готовности модели (poll `/v2/models/{name}/ready`)
4. `triton-model-tools populate-stream --stream test_input --image-dir test_data/coco8/`
5. Запуск pipeline или `triton-model-tools infer`
6. Сравнение результатов с `triton-model-tools local-infer` reference
7. Выгрузка модели (`POST /v2/repository/models/{name}/unload`) перед следующей

**Матрица E2E тестов** (все 3 модели + end2end):

| Модель | Тест | Верификация |
|--------|------|-------------|
| yolo11s | ONNX FP32 640 | convert → load → infer → local-infer → compare detections |
| yolo11s | ONNX FP32 960x608 + DALI | convert --dali → load → infer → verify DALI dims correct |
| yolo11s | ONNX FP32 640 + rescale | convert --rescale → verify coords in original image space |
| yolo12s | ONNX FP32 640 | convert → load → infer → local-infer → compare detections |
| yolo12s | ONNX + TRT optimize 640 | convert --trt-optimize → load → warmup → infer |
| yolo26l | ONNX FP32 640 | convert → load → infer → verify VRAM < 20GB |
| yolo26l | ONNX FP16 640 | convert --half → load → compare FP16 vs FP32 detections |
| yolo26s | ONNX FP32 640 (end2end) | convert --end2end --nms --dali → verify NO double NMS |
| pipeline | yolo11s full pipeline | populate-stream → run pipeline → verify output = input + detections |

### 8.4 Финальная актуализация документации
После завершения ВСЕХ задач — обновить документацию под итоговое состояние кода:
- **`docs/architecture.md`**: написать НОВЫЙ архитектурный документ с нуля, отражающий реальное состояние кода (shipped процессоры, typed overloads, collector modes, per-model sinks, TRT optimize, 3 пакета, актуальные контракты import-linter). Старый `docs/design.md` удалён как устаревший.
- **`README.md`**: обновить quick start, примеры CLI, таблицу метрик, секцию инфраструктуры, пример pipeline
- **`CLAUDE.md`** (оба): отразить три пакета, новые CLI команды, удалённые команды, dependency chain
- **`docs/final-plan.md`**: отметить все задачи как завершённые

### 8.5 Финальная верификация
- `uv run pre-commit run --all-files` — всё зелёное
- `uv run pytest` — все unit тесты проходят
- `uv run pytest -m integration` — все integration тесты (с Docker)
- Пример работает end-to-end
- Документация точна и соответствует коду
- FiftyOne показывает корректные bounding boxes

---

## Найденные баги

| # | Серьёзность | Местоположение | Описание | Решение |
|---|-------------|----------------|----------|---------|
| 1 | **Высокая** | `converter.py:61-62` | `rect=False` не передаётся в ultralytics | Фаза 2.1: `merged_kwargs["rect"] = rect` |
| 2 | **Высокая** | `convert.py:173` | DALI конфиг (640,640) игнорирует imgsz — ломает 960x608 | Фаза 2.2: `target_size=variant.resolved_imgsz` |
| 3 | **Средняя** | `fiftyone_export.py:44-45` | BBox нормализация — координаты детекций вместо размеров изображения | Фаза 7.2: читать размеры из filepath |
| 4 | **Средняя** | `processors.py:335-374`, `test_deterministic.py` | NMS дублирован в 2 местах | Фаза 4.3: импорт из model_tools |
| 5 | **Средняя** | `run_pipeline.py:88-91` | `MetadataInferenceWorker` inline без wrapper — blocking gRPC | Фаза 5.5: обернуть в `DetachedThreadProcessor` |
| 6 | **Средняя** | `register()`, wrappers | Принимают `object` — mypy не ловит ошибки типов | Фаза 5.2: typed overloads |
| 7 | **Низкая** | connector/model_tools `__init__.py` | Публичные классы не re-exported из корня пакетов | Фаза 4.2: добавить экспорты |
| 8 | **Средняя** | `engine/context.py:14-22` | `EventContext` не содержит `original_fields` — serializer не может восстановить входные поля | Фаза 5.1: добавить `original_fields` в EventContext |

*Баги #2,#3 из export.py/populate_repository.py — файлы будут удалены (фаза 2.2a), баг остаётся только в convert.py*

## Архитектурная заметка: Inference инструмент как экземпляр App

Текущий `triton-model-tools infer` (`cli/infer.py` + `folder_inference.py`) — standalone цикл: preprocess -> Triton gRPC -> postprocess на изображение. НЕ является экземпляром pipeline app.

**Требуется верификация**: инструмент архитектурно эквивалентен упрощённому app instance:
- `FileSystemSource` как Source
- `YoloPreprocessor` как Processor
- `TritonInferenceClient.infer_raw()` как inference Processor
- `YoloPostprocessor` как postprocessing Processor
- `DetectionResultSink` как Sink

Инструмент использует те же preprocessor/postprocessor что и pipeline (`YoloPreprocessor`, `YoloPostprocessor`). Необходимо поддержать extension processors — пользовательский postprocessor или замену preprocessing.

**Действие**: проверить что `infer` и `local-infer` используют ТОТ ЖЕ preprocessing/postprocessing код что и pipeline. Инструмент — headless pipeline без Redis.

## Принятые решения

| Вопрос | Решение | Обоснование |
|--------|---------|-------------|
| Базовые классы процессоров | НЕ нужны, typed overloads на register()/wrappers | Линтеры ловят ошибки через protocol conformance в overloads |
| FiftyOne вход | Два YOLO txt каталога + изображения | Работает с внешними инструментами, visual drift comparison |
| Shipped процессоры | Все 6 сразу (включая YoloPostProcessorBridge) | Пример полностью упрощается за одну фазу |
| Redis collector modes | Enum с 3 режимами | Явный, самодокументирующийся, предотвращает невалидные комбинации |
| Batch export | Удалить export/populate-repository CLI, только convert | Контроль GPU памяти, предсказуемость |
| Обратная совместимость | Не нужна | Проект в стадии разработки, можно ломать API |
| Зависимости | tools → connector → shared | tools использует shipped процессоры и pipeline framework |
| E2E Triton mode | `--model-control-mode=explicit` | Управление загрузкой/выгрузкой через API |
| Filesystem чтение | Единственный вариант, без абстракции | Камера → файл → путь в Redis → connector читает файл |
| Бенчмарки | На хосте, не в контейнере, лимит 20GB | Корректные измерения GPU, контроль ресурсов |
| perf_analyzer | Optional, не ломает ничего при отсутствии | Доступен только в Triton SDK контейнере или при ручной установке |
| ONNX TRT optimize | Config-level optimization, не отдельный формат | Triton строит TRT engine из ONNX автоматически |

## Сквозные принципы

- **Tool-first верификация**: всегда использовать `triton-model-tools` CLI. Если шаг верификации отсутствует — расширить инструмент.
- **GPU память**: никогда не загружать несколько моделей одновременно в E2E. Тестировать по одной, выгружать.
- **Консистентность shapes**: каждое изменение export/DALI/preprocessing тестировать с 640x640 И 960x608.
- **Ревью после каждой фазы**: проверка возможностей упрощения и тестовых антипаттернов.

## Порядок выполнения и параллелизм

Большинство задач НЕ зависят друг от друга. Агенты запускаются параллельно максимально.

### Волна 0 — стартовая (всё параллельно)
Запускать все одновременно, нет зависимостей между собой:
- **0.1** Удаление старых примеров
- **0.1a** Реверс зависимостей пакетов
- **0.2** Актуализация документации
- **0.3** Обновление агентов
- **1** Аудит тестов
- **2.1** Исправление rect
- **2.2 + 2.2a** Исправление DALI imgsz в convert.py ЗАТЕМ удаление batch export (последовательно — 2.2 правит файлы, которые 2.2a удаляет)
- **2.5** Верификация валидации разрешений
- **2.6** Проверка end2end NMS
- **3.1** Режимы Redis коллектора
- **3.2** Per-model sink streams
- **4.2** Shipped decoders и публичные экспорты
- **4.3** Очистка NMS
- **4.4** Верификация rescaling
- **5.1** Output = input + detection_results
- **5.2** Типизация register()/wrappers
- **5.4** Документация параллелизма
- **5.5** Нет inline blocking процессоров
- **5.6** Filesystem чтение
- **6.3** Верификация метрик
- **7.1** TRT-оптимизация для ONNX
- **7.2** Мониторинг памяти
- **7.4** perf_analyzer optional
- **8.2** FiftyOne интеграция и визуальная верификация drift

### Волна 1 — зависят от волны 0
- **2.3** ONNX local-infer tool (нет блокирующих зависимостей, но логически после 2.2a удаления batch export)
- **3.3** Инструмент наполнения Redis
- **4.1** DALI vs ultralytics preprocessing (после 2.2 fix DALI imgsz)
- **6.1** Ship стандартных процессоров (после 0.1a реверс зависимостей)
- **6.2** Проверка shared models (после 0.1a)
- **8.1** Минимизация linter ignores (после 5.2 typed overloads)

### Волна 2 — зависят от волны 1
- **2.4** FP16/FP32 верификация (после 2.3 local-infer)
- **2.7** Верификация всех вариантов экспорта на разных размерах (после 2.1, 2.2, 2.3, 7.1)
- **6.4** Финальный пример (после 6.1 shipped процессоры, 5.1 output=input+results)
- **7.3** Бенчмарк по одной модели (после 7.1 TRT optimize, 7.2 memory)
- **7.5** Матрица бенчмарков (после 2.3, 7.1, 7.2, 7.3)

### Волна 3 — финальная
- **8.3** E2E тестовые сценарии (после 2.3, 3.3, 6.4)
- **8.4** Финальная актуализация документации и создание нового architecture.md (после всех задач — отражает итоговое состояние кода)
- **8.5** Финальная верификация (после 8.4)

### Граф зависимостей (только реальные блокеры)
```
[Волна 0: 25 задач параллельно]
  |
  ├─ 0.1a ──→ 6.1, 6.2
  ├─ 2.2  ──→ 4.1
  ├─ 5.2  ──→ 8.1
  ├─ 2.2a ──→ 2.3 ──→ 2.4, 2.7, 7.5, 8.3
  ├─ 7.1  ──→ 2.7, 7.3, 7.5
  ├─ 7.2  ──→ 7.3, 7.5
  ├─ 5.1  ──→ 6.4
  └─ 6.1  ──→ 6.4 ──→ 8.3
                         |
                     8.4 (docs)
                         |
                        8.5
```
