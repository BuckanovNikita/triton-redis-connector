# Step 11: Create Bash Scripts

**Agent:** implementer
**Depends on:** step 06

## Goal

Create bash scripts to replace dropped Python CLI tools: perf_analyze, validate_models, benchmark_all.

## Tasks

### 11.1 Create `scripts/perf_analyze.sh`

Replaces the deleted Python perf_analyzer wrapper. Runs inside triton-sdk container.

**Usage:** `./scripts/perf_analyze.sh -m yolo11s -u localhost:8001 -c 1,4,8`

**Features:**
- Accept: model name, protocol (grpc/http), url, concurrency levels (comma-separated), output dir
- For each concurrency level, run `perf_analyzer` with correct flags
- Save CSV output per model+concurrency
- Print summary table to stdout
- Handle errors gracefully (model not loaded, Triton unreachable)

**Note:** Use flags compatible with Triton 26.02 perf_analyzer (not `--measurement-count`, not `--csv` — use `-f csv` or check `perf_analyzer --help` for correct syntax).

### 11.2 Create `scripts/validate_models.sh`

Replaces the deleted validate CLI tool. Tests that models actually load in Triton.

**Usage:** `./scripts/validate_models.sh [model_repository_path] [triton_url]`

**Features:**
- For each model directory in repository:
  - Call Triton API to load model: `curl -X POST localhost:8000/v2/repository/models/{name}/load`
  - Check health: `curl localhost:8000/v2/health/ready`
  - Check model status: `curl localhost:8000/v2/models/{name}`
  - Report PASS/FAIL per model
- Print summary table
- Exit code 1 if any model fails

### 11.3 Create `scripts/benchmark_all.sh`

Replaces the deleted benchmark-all CLI tool. Orchestrates single-model benchmarks.

**Usage:** `./scripts/benchmark_all.sh [--repository-root model_repository] [--url localhost:8001] [--output-dir e2e_results]`

**Features:**
- Discover all model directories in repository (skip auxiliary: _dali_, _ensemble, _nms, _preprocess)
- **Include TRT-accelerated variants** (`*_ort_trt`) — do NOT skip them
- For each model, run: `uv run triton-model-tools benchmark -m {model} --url {url} -o {output_dir}/{model}.json`
- Collect results into summary table
- Save combined results to `{output_dir}/benchmark_all.json`

**Important:** The skip list for auxiliary models must NOT include `*_ort_trt` patterns. Only skip: `*_dali_preprocess`, `*_dali_ensemble`, `*_dali_rescale_ensemble`, `*_dali_nms_rescale_ensemble`, `rescale_postprocess`, `nms_postprocess`.

### 11.4 Make scripts executable

```bash
chmod +x scripts/perf_analyze.sh scripts/validate_models.sh scripts/benchmark_all.sh
```

### 11.5 Update docs

Add script descriptions to `CLAUDE.md` CLI commands section.

## Verification

```bash
# Syntax check
bash -n scripts/perf_analyze.sh
bash -n scripts/validate_models.sh
bash -n scripts/benchmark_all.sh

# Dry run (if Triton is available)
./scripts/validate_models.sh model_repository localhost:8000
```
