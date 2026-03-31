# triton-shared-models

Общие типы и модели данных для NVIDIA Triton Inference Server, используемые пакетами `triton-redis-connector` и `triton-model-tools`.

## Основные типы

| Тип | Описание |
|-----|----------|
| `TritonDataType` | Перечисление типов данных Triton (`TYPE_FP32`, `TYPE_FP16`, `TYPE_INT8`, ...) |
| `TritonInstanceKind` | Тип instance group (`KIND_GPU`, `KIND_CPU`) |
| `ExportFormat` | Формат экспорта модели (`onnx`, `tensorrt`) |
| `TritonTensorSpec` | Спецификация одного тензора (имя, тип, размерности) |
| `TritonModelConfig` | Полная конфигурация модели Triton |

## Генерация config.pbtxt

`TritonModelConfig` рендерит конфигурацию в формат protobuf text через Jinja2-шаблон:

```python
from triton_shared_models import TritonModelConfig, TritonTensorSpec, TritonDataType

config = TritonModelConfig(
    name="yolo11s",
    platform="onnxruntime_onnx",
    max_batch_size=8,
    inputs=[
        TritonTensorSpec(name="images", data_type=TritonDataType.TYPE_FP32, dims=[3, 640, 640]),
    ],
    outputs=[
        TritonTensorSpec(name="output0", data_type=TritonDataType.TYPE_FP32, dims=[84, 8400]),
    ],
)

pbtxt = config.to_pbtxt()
```

## Дополнительные конфигурации

| Модель | Описание |
|--------|----------|
| `DynamicBatchingConfig` | Настройки динамического батчинга (`max_queue_delay`, `preferred_batch_size`) |
| `OptimizationConfig` | Оптимизации (graph level, CUDA graphs) |
| `ModelWarmupConfig` | Прогрев модели при загрузке |
| `EnsembleStep` | Шаг ансамблевого расписания (input/output mapping) |

## Структура

```
triton_shared_models/
  triton_types.py    -- TritonDataType, TritonTensorSpec, ExportFormat
  triton_config.py   -- TritonModelConfig, to_pbtxt(), вспомогательные модели
  templates/         -- Jinja2-шаблон config.pbtxt.j2
```
