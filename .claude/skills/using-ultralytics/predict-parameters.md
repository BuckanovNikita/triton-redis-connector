# Predict Parameters — Full Reference

## Inference Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | various | required | Image, directory, URL, numpy, video, stream |
| `conf` | float | 0.25 | Minimum confidence threshold |
| `iou` | float | 0.7 | NMS IoU threshold |
| `imgsz` | int/tuple | 640 | Inference image size |
| `half` | bool | False | FP16 inference (GPU only) |
| `device` | str/int | auto | `"cpu"`, `0`, `"0,1"`, `"mps"` |
| `max_det` | int | 300 | Max detections per image |
| `classes` | list[int] | None | Filter specific class indices |
| `agnostic_nms` | bool | False | Class-agnostic NMS |
| `augment` | bool | False | Test-time augmentation |

## Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | bool | True | Print results to console |
| `stream` | bool | False | Generator mode (memory-efficient for video) |
| `save` | bool | False | Save annotated images to `runs/detect/predict/` |
| `save_txt` | bool | False | Save results as `.txt` files |
| `save_conf` | bool | False | Include confidence in `.txt` output |
| `save_crop` | bool | False | Save cropped detections |
| `show` | bool | False | Display results in window |
| `show_labels` | bool | True | Show class labels on annotated images |
| `show_conf` | bool | True | Show confidence on annotated images |
| `show_boxes` | bool | True | Show bounding boxes on annotated images |
| `line_width` | int | None | Bounding box line width |

## Advanced

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vid_stride` | int | 1 | Video frame stride (skip frames) |
| `retina_masks` | bool | False | High-res segmentation masks |
| `embed` | list[int] | None | Return feature embeddings from layers |
| `project` | str | None | Save directory project name |
| `name` | str | None | Save directory run name |
| `exist_ok` | bool | False | Overwrite existing save directory |
