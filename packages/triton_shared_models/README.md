# triton-shared-models

Общие типы и модели данных для NVIDIA Triton Inference Server, используемые пакетами `triton-redis-connector` и `triton-model-tools`.

## Основные типы

| Тип | Описание |
|-----|----------|
| `TritonDataType` | Перечисление типов данных Triton (`TYPE_FP32`, `TYPE_FP16`, `TYPE_INT8`, ...) |
| `TritonInstanceKind` | Тип instance group (`KIND_GPU`, `KIND_CPU`) |
| `ExportFormat` | Формат экспорта модели (`onnx`, `tensorrt`) |
| `TritonTensorSpec` | Спецификация одного тензора (имя, тип, размерности) |
## Структура

```
triton_shared_models/
  triton_types.py      -- TritonDataType, TritonTensorSpec, ExportFormat
  detection/
    preprocessing.py   -- LetterBoxPreprocessor
    __init__.py        -- Detection
  classification/      -- ClassificationPrediction, CenterCropPreprocessor
  nms.py               -- NMS
```

> Конфигурация моделей Triton (`TritonModelConfig`, `to_pbtxt()`) перенесена в пакет `triton-model-tools`.
