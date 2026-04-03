# Acquiring COCO val2017 Images for Benchmarking

Two approaches to get the COCO val2017 dataset (5000 images) for model benchmarking.

## Approach 1: Using the built-in CLI command (recommended)

```bash
uv run triton-model-tools download-coco --output-dir benchmarks/data/coco_val2017
```

## Approach 2: Using FiftyOne (for development)

```bash
pip install fiftyone
```

```python
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("coco-2017", split="validation")
print(dataset)
```

Images are downloaded to `~/fiftyone/coco-2017/validation/data/` by default.
Copy them to your benchmark directory:

```bash
cp -r ~/fiftyone/coco-2017/validation/data/* benchmarks/data/coco_val2017/
```

## Approach 3: Direct download

```bash
mkdir -p benchmarks/data
wget http://images.cocodataset.org/zips/val2017.zip -O benchmarks/data/val2017.zip
unzip benchmarks/data/val2017.zip -d benchmarks/data/coco_val2017
rm benchmarks/data/val2017.zip
```

## Notes

- COCO val2017 contains 5000 images, which is the recommended minimum for
  reliable benchmark measurements (`--min-images 5000`).
- For non-standard model input sizes, use the `scripts/replicate_images.py`
  script to tile images to the target resolution before benchmarking.
