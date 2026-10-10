# HarnessFix: reproduction and OpenHands integration guide

HarnessFix is a trace-guided pipeline for diagnosing failed LLM-agent trajectories and repairing the harness that produced them. This repository contains the four original benchmark integrations from the HarnessFix paper and an adapter that applies the same analysis/repair stages to an OpenHands Software Agent SDK harness executed by [Better Harnesses, Smaller Models](https://github.com/malusamayo/slm-harness-adaptation).

This README is an operational guide for setting up a new machine, selecting Gemini or Qwen, and running the original and OpenHands closed-loop pipelines.

The [complete Vietnamese setup and run guide](docs/setup_vi.md) covers host tools,
both reproduce checkouts, SDK patches, model endpoints/credentials, all four
OpenHands tasks, SWE-Bench, GAIA, Terminal-Bench and AppWorld, plus dry runs,
resume and held-out test. Use the [Qwen](configs/qwen.yaml) and
[Gemini](configs/gemini.yaml) profiles with `scripts/configure_models.py`.

| Setup/run topic | Commands |
|---|---|
| Host and Python environments | [Guide sections 1–3](docs/setup_vi.md#1-công-cụ-host-và-hai-checkout) |
| Task and analysis model configuration | [Guide section 4](docs/setup_vi.md#4-cấu-hình-model-và-kiểm-tra-kết-nối) |
| Stock Alert, Machine Operating, RefactorBench, WebArena | [Guide section 5](docs/setup_vi.md#5-setup-và-chạy-bốn-benchmark-openhands) |
| SWE-Bench, GAIA, Terminal-Bench, AppWorld | [Guide section 6](docs/setup_vi.md#6-setup-và-chạy-các-benchmark-gốc) |
| Resume and test | [Guide section 7](docs/setup_vi.md#7-resume-test-cuối-và-đọc-kết-quả) |

## 1. What is in this repository

| Track | Initial harness | Tasks | Entry point |
|---|---|---|---|
| SWE-Bench | `task_agent/mini-swe-agent` | SWE-Bench Verified | `run_pipeline_swe.py` |
| GAIA | `task_agent/open_deep_research` | GAIA | `run_pipeline_gaia.py` |
| AppWorld | `task_agent/appworld_agent` | AppWorld | `run_pipeline_appworld.py` |
| Terminal-Bench | `task_agent/terminal_bench_agent` | Terminal-Bench 2.0 | `run_pipeline_terminal_bench.py` |
| OpenHands | `task_agent/openhands_agent/original` | Stock Alert, Machine Operating, RefactorBench, WebArena | `task_agent.openhands_agent.bridge` and `run_pipeline_openhands.py` |

All five integrations have a closed-loop driver. Better Harness remains responsible for OpenHands environment setup, execution, and official evaluation. HarnessFix materializes candidates, normalizes artifacts, diagnoses failures, creates scoped repairs, audits them, compares train/validation rollouts, and selects a harness using validation. OpenHands also exposes individual stage commands.

The common repair flow is:

```text
H0 on train
  -> official evaluator
  -> failed trajectories
  -> HTIR and per-rollout diagnosis
  -> aggregate plan + binding edit specification
  -> H1 candidate
  -> plan/diff/syntax audit
  -> train comparison
  -> paired validation gate
  -> promote or reject
  -> held-out test only after the repair loop ends
```

Important directories:

```text
failure_analysis/              HTIR, diagnosis, aggregation, audit, gate, memory
enhancement_implementation/    prompts/configuration used by the modifying agent
task_agent/                    initial harnesses and benchmark runners
task_agent/final/              final paper snapshots; not the default H0 inputs
task_agent/openhands_agent/     Better Harness/OpenHands bridge and candidate template
configs/                       Qwen and Gemini model configuration
scripts/                       shared model setup
data/                          sampling/download scripts; raw datasets are not committed
eval/                          GAIA, AppWorld, and Terminal-Bench evaluators
traces/, logs/, results/       generated runtime artifacts
artifacts/                     versioned OpenHands candidates and normalized runs
```

## 2. Host requirements

Use Linux or WSL2. The original drivers invoke Bash, use `.venv/bin/python3`, and depend heavily on Linux Docker containers. Running them directly from Windows PowerShell is not supported end to end; on Windows, install WSL2 and enable Docker Desktop's WSL integration.

Recommended host software:

- Git and Git LFS.
- Docker Engine with the Compose v2 plugin.
- Python 3.12 for this HarnessFix repository. Terminal-Bench's vendored Harbor requires Python 3.12 or newer.
- `uv` for the Better Harness repository. Its current `pyproject.toml` declares its own Python version, so keep its environment separate from `.venv` here.
- Enough disk for benchmark data and Docker images. SWE-Bench alone can require many large per-instance images.

Check the host before installation:

```bash
git --version
docker version
docker compose version
python3.12 --version
uv --version
```

### 2.1 Ubuntu 24.04 setup

Ubuntu 24.04 LTS is the simplest native setup because its repositories include Python 3.12. Start from a clean machine and install the host tools:

```bash
sudo apt update
sudo apt install -y \
  ca-certificates curl git git-lfs jq ripgrep build-essential \
  python3.12 python3.12-venv python3-pip

git lfs install
```

Install Docker Engine and the Compose v2 plugin from [Docker's official Apt repository](https://docs.docker.com/engine/install/ubuntu/):

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

Allow the current user to run Docker without `sudo`, then log out and back in so the new group is applied. Membership in the `docker` group grants root-equivalent access to the machine.

```bash
sudo usermod -aG docker "$USER"
```

After logging in again, verify Docker:

```bash
docker run --rm hello-world
docker compose version
```

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Better Harness:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Now continue with section 3. The HarnessFix virtual environment and Better Harness `uv` environment must remain separate. If Qwen/vLLM runs on another machine, Ubuntu only needs network access to that server; it does not need a local GPU or a local vLLM installation.

## 3. Install HarnessFix

```bash
git clone https://github.com/trunghieu1109/HarnessFix-reproduce.git HarnessFix-reproduce
cd HarnessFix-reproduce
export HARNESSFIX_ROOT="$PWD"
export BETTER_ROOT="$HOME/slm-harness-adaptation-reproduce"
export BETTER_HARNESS_ROOT="$BETTER_ROOT"

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Required by the repair/analysis agents and SWE runner.
python -m pip install -e task_agent/mini-swe-agent

# Required for GAIA.
python -m pip install -e task_agent/open_deep_research

# Required for Terminal-Bench.
python -m pip install -e task_agent/terminal_bench_agent/harbor

if [ ! -f .env ]; then cp .env.example .env; fi
chmod 600 .env
```

The repository deliberately excludes credentials, raw benchmark data, traces, and generated evaluations. Never commit `.env`.

Run local, non-network integration tests:

```bash
python -m unittest discover -s tests -v
python -m json.tool task_agent/model_registry.json >/dev/null
```

## 4. Configure Gemini or Qwen

Configure model IDs, endpoints and credentials before starting a new experiment.
`configure_models.py` requires a Better Harness checkout even when preparing an
original benchmark. Complete section 7 before using it, or follow the Vietnamese
guide's checkout → environment → model order in sections 1–4.
The [model setup commands](docs/setup_vi.md#4-cấu-hình-model-và-kiểm-tra-kết-nối)
create local profiles from `configs/qwen.yaml` and `configs/gemini.yaml`, probe
model discovery/chat/tool calls, and keep credentials in `.env`.

| Route | Model ID | Connection |
|---|---|---|
| Qwen on an OpenAI-compatible server | `openai/<served-model-id>` | HTTP(S) endpoint ending in `/v1`, `QWEN_API_KEY` |
| Gemini on an OpenAI-compatible gateway | `openai/<gateway-model-id>` | Gateway endpoint ending in `/v1`, `GEMINI_API_KEY` |
| Native Gemini | `gemini/<native-model-id>` | `api_base: null`, Google `GEMINI_API_KEY` |

The `openai/` prefix selects the API protocol. Replace the profile endpoint
placeholders with actual URLs; `configure_models.py` does not resolve a bare
`QWEN_BASE_URL`/`GEMINI_BASE_URL` string in a profile. The task endpoint must be
reachable inside Docker, and OpenHands requires native function/tool calls.
Set token budgets to fit the server's context, reserving room for output.

After creating the local profiles and exporting the appropriate credentials:

```bash
cd "$HARNESSFIX_ROOT"
.venv/bin/python -B scripts/configure_models.py \
  --config configs/qwen.local.yaml \
  --analysis-config configs/gemini.local.yaml \
  --better-root "$BETTER_ROOT"

set -a
source .env
set +a
```

This sets `HARNESSFIX_TASK_MODEL` and `HARNESSFIX_ANALYSIS_MODEL` independently,
adds `qwen-vllm`/`gemini-api` aliases to Better's model registry, and updates the
SWE registry and analysis/repair model kwargs. For one model, omit
`--analysis-config`. To update only analysis, use `--analysis-only` with its
profile. For OpenHands, experiment YAML model aliases must match the registry;
changing the selected profile does not rewrite experiment YAMLs.

GAIA, AppWorld and Terminal-Bench still read analysis/modifier connection kwargs
from their own YAML files. Follow [guide section 4.4](docs/setup_vi.md#44-model-cho-gaia-appworld-và-terminal-bench)
before running them. The documented commands use the task model for their
analysis/repair; two models with separate endpoints need per-stage connection
configuration. SWE and OpenHands already resolve connections independently.

## 5. Prepare the original HarnessFix benchmarks

Only prepare the benchmark you intend to run.

### 5.1 SWE-Bench Verified

Install the official evaluator in the same `.venv` so that `swebench.harness.run_evaluation` is importable. Pin version 4.1.0, which supports this pipeline's `--report_dir` argument, JSON predictions, and evaluation log paths:

```bash
python -m pip install 'swebench==4.1.0'
```

Create deterministic, non-overlapping train/validation/test subsets:

```bash
python data/sample_swebench.py
```

This creates:

```text
data/verified_train_100/
data/verified_val_50/
data/verified_test_100/
```

Optionally pre-pull the images for each subset:

```bash
bash data/pull_swebench_images.sh data/verified_train_100 test 4
bash data/pull_swebench_images.sh data/verified_val_50 test 4
bash data/pull_swebench_images.sh data/verified_test_100 test 4
```

### 5.2 GAIA

Request access to the gated `gaia-benchmark/GAIA` dataset, then set these values in `.env`:

```dotenv
HF_TOKEN=hf_...
SERPAPI_API_KEY=...
# SERPER_API_KEY may be used as the search fallback.
```

Create the deterministic splits:

```bash
python data/sample_gaia.py
```

This creates `gaia_train_60`, `gaia_val_30`, and `gaia_test_60` under `data/`.

### 5.3 Terminal-Bench 2.0

For a fresh checkout without a private split manifest, download the official
GitHub source and create deterministic local splits:

```bash
python data/download_terminal_bench.py --source github --skip-sampling
python data/sample_terminal_bench.py \
  --source data/terminal_bench_2_verified --random-split \
  --train 34 --val 17 --test 34 --seed 20260522
git -C data/terminal_bench_2_verified rev-parse HEAD
```

The driver retains the `terminal_bench_2_verified` directory name for this
source. To reproduce a previous ZAI verified experiment, provide its manifest
with pinned `source_commit` and task IDs; see [Terminal-Bench setup](docs/setup_vi.md#63-terminal-bench-20).
The default downloader expects `data/terminal_bench_splits.json`, which is not
included in this checkout.

The output is:

```text
data/terminal_bench_2_verified/
data/terminal_bench_train/
data/terminal_bench_val/
data/terminal_bench_test/
```

The sampler uses symlinks by default. On filesystems where symlinks are unavailable, use:

```bash
python data/sample_terminal_bench.py --random-split --copy
```

### 5.4 AppWorld

Build the pinned execution image:

```bash
docker build -t appworld-agent-pypi:latest task_agent/appworld_agent
```

Follow the [complete AppWorld setup commands](docs/setup_vi.md#64-appworld) to
download data using the pinned image, set absolute paths in `.env`, and build
`APPWORLD_TASK_CACHE` from the official task metadata inside Docker. The cache
contains `train`, `dev`, `test_normal` and `test_challenge` lists; it supplies the
instruction/supervisor fields required by the local runner without exporting
answers. The host vendored AppWorld shim is for the agent adapter, so use the
real AppWorld package inside the image for task loading.

```dotenv
APPWORLD_ROOT=/absolute/path/to/appworld_root
APPWORLD_AGENT_IMAGE=appworld-agent-pypi:latest
APPWORLD_TASK_CACHE=/absolute/path/to/appworld_task_cache.json
```

Then sample:

```bash
python data/sample_appworld.py
```

This creates `appworld_train_90`, `appworld_val_45`, and `appworld_test_90` under `data/`. The inline cache-building command is in the setup guide. Preserve the cache and its hash to reproduce the same split on another machine.

## 6. Run the original closed-loop pipelines

Load the model settings from section 4 and complete the benchmark-specific setup:

```bash
set -a
source .env
set +a
export MODEL="$HARNESSFIX_TASK_MODEL"
export ANALYSIS_MODEL="$HARNESSFIX_ANALYSIS_MODEL"
export LEGACY_ANALYSIS_MODEL="${LEGACY_ANALYSIS_MODEL:-$HARNESSFIX_TASK_MODEL}"
```

Add `--dry-run` to inspect commands after preparing data. The examples below
run one repair iteration; increase `--max-iterations` to 3 for the full loop.
For GAIA/AppWorld/Terminal-Bench, first synchronize the connection YAMLs in guide
section 4.4 with `LEGACY_ANALYSIS_MODEL`.

SWE-Bench:

```bash
python run_pipeline_swe.py \
  --model "$MODEL" \
  --analysis-model "$ANALYSIS_MODEL" \
  --workers 2 \
  --max-iterations 1
```

GAIA:

```bash
python run_pipeline_gaia.py \
  --model "$MODEL" \
  --analysis-model "$LEGACY_ANALYSIS_MODEL" \
  --workers 2 \
  --concurrency 2 \
  --max-iterations 1
```

Terminal-Bench:

```bash
python run_pipeline_terminal_bench.py \
  --model "$MODEL" \
  --analysis-model "$LEGACY_ANALYSIS_MODEL" \
  --workers 2 \
  --max-iterations 1
```

AppWorld:

```bash
python run_pipeline_appworld.py \
  --model "$MODEL" \
  --analysis-model "$LEGACY_ANALYSIS_MODEL" \
  --workers 2 \
  --concurrency 2 \
  --max-iterations 1
```

Each driver performs baseline validation, H0 train execution/evaluation, failure analysis, aggregation, modification, audit, enhanced train comparison, validation execution/evaluation, and a promotion decision. Existing complete artifacts are resumed by default; `--force` regenerates stages and should be used deliberately.

Typical outputs are:

```text
traces/<run>/                              raw task-agent trajectories and predictions
eval/<result-dir>/                         official or benchmark-specific evaluation
failure_analysis/results/*analysis.jsonl   one diagnosis per failed instance
improvement_plans/*.md                     human-readable repair plan
improvement_plans/*.json                   binding edit scope and acceptance targets
task_agent/enhanced_*_vN/                  generated candidate harness
failure_analysis/results/audit_*.json      plan-to-diff/syntax audit
failure_analysis/results/gate_*.json       paired validation comparison
failure_analysis/memory/                   accepted/rejected repair memory
```

At completion, each pipeline prints the held-out test command for the best promoted version. Run that command once. Do not feed held-out test failures into analysis, aggregation, retry, or promotion.

## 7. Install Better Harness for OpenHands tasks

Keep Better Harness in a sibling directory with its own `uv` environment:

```bash
cd ..
git clone --recurse-submodules https://github.com/trunghieu1109/slm-harness-adaptation-reproduce.git
cd slm-harness-adaptation-reproduce
git submodule update --init --recursive

export BETTER_ROOT="$PWD"
export HARNESSFIX_ROOT="$(cd ../HarnessFix-reproduce && pwd)"
```

This reproduction keeps the SDK Docker startup fixes in
[`patches/software-agent-sdk-docker-startup.patch`](patches/software-agent-sdk-docker-startup.patch),
based on SDK commit `89c21968922afc746bca8a653712038d5a38d6aa`. The patch cleans up
containers after failed startup, uses a monotonic health-check timer, preserves
health-check error details, and prevents the log thread from joining itself. It
also includes the corresponding workspace tests. Apply it locally after
initializing the submodule; the SDK submodule revision stays unchanged and no
push to the shared SDK repository is required.

On a checkout where the patch has not been applied:

```bash
git -C "$BETTER_ROOT/software-agent-sdk" apply --check \
  "$HARNESSFIX_ROOT/patches/software-agent-sdk-docker-startup.patch"
git -C "$BETTER_ROOT/software-agent-sdk" apply \
  "$HARNESSFIX_ROOT/patches/software-agent-sdk-docker-startup.patch"
```

If the checkout already contains these fixes, skip the two commands above.
Verify that the patch is present with:

```bash
git -C "$BETTER_ROOT/software-agent-sdk" apply --reverse --check \
  "$HARNESSFIX_ROOT/patches/software-agent-sdk-docker-startup.patch"
```

The patched submodule will appear as modified (`-dirty` in the parent diff).
The patch file is the versioned record of those changes. Better Harness installs
`openhands-workspace` from this checkout as an editable dependency. Install its
dependencies after applying the patch:

```bash
cd "$BETTER_ROOT"
uv sync --python 3.14

# Needed as a mounted file even when using a non-Vertex model.
test -e .vertex-ai.json || printf '%s\n' '{}' > .vertex-ai.json
```

For strict reproduction, record and reuse both repository revisions, the SDK
revision, and the patch checksum. Apply the same patch before all H0, H1, and
held-out test runs:

```bash
git -C "$BETTER_ROOT" rev-parse HEAD
git -C "$HARNESSFIX_ROOT" rev-parse HEAD
git -C "$BETTER_ROOT/software-agent-sdk" rev-parse HEAD
sha256sum "$HARNESSFIX_ROOT/patches/software-agent-sdk-docker-startup.patch"
```

Better Harness and the OpenHands analysis, aggregation, and modification stages now read the same model entries from `$BETTER_ROOT/configs/models.yaml`. API keys and base URLs can be literal values in that file or `${VAR}` references resolved from `$BETTER_ROOT/.env` or the environment. Use LiteLLM provider-qualified IDs for OpenAI-compatible endpoints:

```bash
uv run python -c 'from src.utils import LM_DICT; print(sorted(LM_DICT))'
```

Use the alias in both Better run YAML and HarnessFix's OpenHands `--model` option:

| Location | Gemini | Self-hosted Qwen/vLLM |
|---|---|---|
| Model alias | `gemini-api` | `qwen-vllm` |
| LiteLLM model ID in Better config | `openai/ag/gemini-3.1-pro-low` | `openai/Qwen/Qwen3.5-9B` |

Edit the selected Better task YAML so `model_name` is one of those aliases. The `openai/` prefix selects the OpenAI-compatible protocol; it is stripped before the model ID is sent to the configured endpoint. Keep `model_name`, `prompt_name`, `n_responses`, runtime limits, task IDs, and data fixed between H0 and H1.

Generation settings in `configs/qwen.yaml` and `configs/gemini.yaml` are synchronized by `scripts/configure_models.py` into Better's `configs/models.yaml` and the SWE model registry. Qwen disables thinking with `extra_body.chat_template_kwargs.enable_thinking: false`, following the [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B). The packed OpenHands launcher preserves temperature in this mode. Gemini uses temperature `0.2` and `reasoning_effort: low`; [Gemini 3.1 Pro cannot disable thinking entirely](https://ai.google.dev/gemini-api/docs/generate-content/thinking). A null reasoning effort leaves provider defaults in effect.

Build the task images you need:

```bash
cd "$BETTER_ROOT"
benchmark_uid="$(id -u)"
if [ "$benchmark_uid" -eq 0 ]; then benchmark_uid=1000; fi
BENCHMARK_UID="$benchmark_uid" docker compose build \
  woocommerce_stock_alert_s2l \
  machine_operating_s2l \
  refactorbench \
  webarena
```

Compose in this reproduce checkout uses `BENCHMARK_UID` and creates a non-root
`appuser`; the command selects UID 1000 on a root host. The SLM task setup makes
rollout workspaces writable. Build only the task images needed; OpenHands uses
the regular images, while `_codex` images belong to a separate runtime.

Additional task setup remains owned by Better Harness:

- Stock Alert and Machine Operating use its LOCA-bench submodule and task services.
- RefactorBench requires the official repository snapshots, `REFACTORBENCH_REPOS_DIR` in the launching shell, and host Python 3.11 for its evaluator; see [setup commands](docs/setup_vi.md#52-source-repository-cho-refactorbench).
- WebArena requires the shopping-admin site started before collection; see [site startup and GET health check](docs/setup_vi.md#53-website-cho-webarena). The experiment supplies `eval_model: gemini-api` for fuzzy matching.
- Better's `data/*.json` files are the raw task datasets. For a fair HarnessFix study, create disjoint train, validation, and held-out test JSON files and corresponding run YAMLs before any repair. The bridge does not invent a split.

A practical convention is:

```text
tasks/<task>/run_harnessfix_train.yaml
tasks/<task>/run_harnessfix_val.yaml
tasks/<task>/run_harnessfix_test.yaml
```

These three files should differ only where a split requires it: `data_path`, `max_examples`, and the selected `n_responses` for that split. Use one rollout per sample for train/validation and two for the final test. Validation and test data must remain disjoint from train, and test must never enter the repair loop.

The OpenHands bridge enables SDK completion logging for new rollouts in
`<workspace>_logs/llm_completions/`. Normalized manifests reference those files
through `llm_completion_paths`; analysis merges recorded requests/responses into
`model_calls` and HTIR. Older event-only traces remain supported, with missing
request payloads explicitly marked `not_recorded`. Use a new rollout version to
capture completion logs; resuming an old rollout cannot recover unrecorded requests.

## 8. Run the OpenHands closed loop

Prepared closed-loop configs cover all four tasks:

| Task | Config | Train / val / test samples | Test responses | Agent batch |
|---|---|---|---:|---:|
| Stock Alert | `configs/stock_alert.yaml` | 10 / 10 / 30 | 2 | 6 |
| Machine Operating | `configs/machine_operating_batch4.yaml` | 10 / 10 / 30 | 2 | 4 |
| RefactorBench | `configs/refactorbench_batch4.yaml` | 10 / 10 / 30 | 2 | 4 |
| WebArena Shopping Admin | `configs/webarena.yaml` | 10 / 10 / 30 | 2 | 4 |

Each full config has 30 model calls per diagnosis, at most 3 repair iterations,
evaluation batch size 6, and test after selection. See [the setup/run guide](docs/setup_vi.md#5-setup-và-chạy-bốn-benchmark-openhands)
for prerequisite images, repository snapshots, websites and direct commands.
`configs/stock_alert_smoke.yaml` uses train/val 10/10, test 5 × 1, analysis budget
10 and one repair iteration.

For example, Stock Alert:

```bash
cd "$HARNESSFIX_ROOT"
.venv/bin/python -B run_pipeline_openhands.py run --config configs/stock_alert.yaml --dry-run
.venv/bin/python -B run_pipeline_openhands.py run --config configs/stock_alert.yaml
```

`configs/stock_alert.yaml` selects test records 1–30, train records 31–40, and
validation records 41–50, in dataset order. It uses `qwen-vllm` for task execution,
`gemini-api` for analysis/aggregation/modification, one rollout per train/validation
record, two rollouts per final test record, and six concurrent task/evaluation
workers. `execution.n_responses` explicitly defines the `train`, `val`, and `test`
counts; generated Better run YAMLs each receive a scalar count for their split.
Each candidate uses the same model parameters, data, seeds, and rollout count
within a split. This gives 10 train and 10 validation rollouts per candidate,
followed by 60 test rollouts for the selected final candidate. Other supported tasks
can use the same config schema with their own dataset, prompt and base run YAML.

The runner follows the current SWE driver: baseline validation, current-base
train execution/evaluation, complete failure analysis, aggregation, modification,
audit, candidate train/validation execution, comparison, promotion, repair memory,
and validation regression analysis for the next iteration. Accepted candidates
become the next base; rejected candidates are recorded as failed attempts.
An audit failure prevents candidate execution. Train comparison and cost are
reported only. Both SWE and OpenHands call `failure_analysis/loop_policy.py` for
comparison and promotion: audit must pass, validation resolved count and target
metrics must improve, and error/invalid-submission rate increases must stay within
the configured limits. Metric availability remains benchmark-specific.

The default limit is three repair iterations, with early stopping after two
consecutive rejected candidates or when all current-base train rollouts pass.
The final candidate is chosen only from H0 and promoted versions using validation
resolved count. With `run_test: true`, only that candidate is evaluated on held-out
test after selection. With `run_test: false`, the runner prints the separate test
command, matching the original drivers. Test feedback never enters repair planning.
Success rates count individual rollouts; they are not pass@2.

Each run prints its directory under `artifacts/openhands/`. Resume an interrupted
run by specifying that exact directory:

```bash
.venv/bin/python -B run_pipeline_openhands.py run \
  --config configs/stock_alert.yaml \
  --run-dir /absolute/path/to/the/printed/run/directory
```

Completed stages are reused. Interrupted candidate edits are archived before
retrying. Config, dataset, model settings and pipeline hashes must match to resume;
changed settings require a new run directory. After final selection, resume only
returns the selected result or finishes its test; it does not start more repair.
Source dataset IDs distinguish samples across splits, independently of local
`example0` workspace names. Analysis trajectories are separated by run and stage.
The automatic runner uses the existing repair-memory format and retrieval logic
in `<run>/memory/`. It starts a new memory for each experiment so previous runs
using different train/test splits cannot enter its planner context. Individual
stage commands and original drivers retain their existing default memory path.

```text
<run>/experiment.json                 configs, split IDs/hashes, model parameters
<run>/provenance.json                 checkout commits and configured task images
<run>/configs/run_{train,val,test}.yaml
<run>/candidates/h0, h1, ...
<run>/runs/{train,val,test}_hN/        results, trace manifests and trace checksums
<run>/analysis/*.jsonl                train failures and val regressions
<run>/memory/                         accepted/rejected repairs from this experiment
<run>/iterations/vN/                  plan/spec, modifier trace, audit, comparisons,
                                     promotion decision and iteration report
<run>/selection.json                  fixed final selection before held-out test
<run>/summary.json                    selection plus test result when requested
```

The commands below expose the same stages individually for inspection or a
single manual repair iteration.

The following example uses RefactorBench. Supported Better task IDs are:

- `woocommerce_stock_alert_s2l`
- `machine_operating_s2l`
- `refactorbench`
- `webarena`

Use `prompt_name: shopping_admin` for WebArena and `default` for the other three tasks.

Set paths and models:

```bash
cd "$HARNESSFIX_ROOT"
export TASK=refactorbench
export PROMPT=default
export ANALYSIS_MODEL=gemini-api  # or qwen-vllm after configuring that alias
```

### 8.1 Materialize H0

```bash
python -B -m task_agent.openhands_agent.bridge materialize \
  --better-root "$BETTER_ROOT" \
  --task-id "$TASK" \
  --prompt-name "$PROMPT" \
  --output-dir "artifacts/$TASK/h0"
```

The output is a versioned, multi-file candidate bundle. The command refuses to overwrite an existing directory.

### 8.2 Run and normalize H0 on train

```bash
python -B -m task_agent.openhands_agent.bridge run \
  --better-root "$BETTER_ROOT" \
  --base-config "tasks/$TASK/run_harnessfix_train.yaml" \
  --candidate-dir "artifacts/$TASK/h0" \
  --rollout-version harnessfix_h0_train \
  --normalized-output "artifacts/$TASK/train_h0"
```

The bridge keeps the Better YAML configuration and overrides only `agent_file` and `rollout_version`. It calls Better's original `src.collect` and `src.evaluate`, then writes:

```text
artifacts/<task>/train_h0/results.json
artifacts/<task>/train_h0/manifest_index.json
artifacts/<task>/train_h0/traces/<stable-rollout-id>/manifest.json
```

Better Docker mode mounts one `agent_file`. To preserve a multi-file candidate, the bridge creates one self-extracting Python launcher containing `agent.py`, `prompts/`, `skills/`, `tools/`, `hooks/`, `verification/`, and the other candidate-owned components. The launcher extracts to a temporary directory inside the OpenHands process and imports the real `agent.py`; it does not patch Better Harness or the evaluator.

### 8.3 Diagnose all failed train rollouts

```bash
python -B run_pipeline_openhands.py analyze \
  --better-root "$BETTER_ROOT" \
  --traces-dir "artifacts/$TASK/train_h0/traces" \
  --eval-results "artifacts/$TASK/train_h0/results.json" \
  --agent-source-dir "artifacts/$TASK/h0" \
  --output-file "artifacts/$TASK/train_h0_analysis.jsonl" \
  --model "$ANALYSIS_MODEL" \
  --workers 2
```

Every failed rollout is analyzed. HTIR pairs OpenHands `ActionEvent` and `ObservationEvent` records by `tool_call_id` and anchors the outcome to Better's official evaluator score/feedback. A successful tool observation is not treated as proof that the task state changed correctly.

### 8.4 Aggregate, modify, and audit H1

```bash
python -B run_pipeline_openhands.py aggregate \
  --better-root "$BETTER_ROOT" \
  --results-file "artifacts/$TASK/train_h0_analysis.jsonl" \
  --output "artifacts/$TASK/plan_h1.md" \
  --spec-output "artifacts/$TASK/plan_h1.json" \
  --model "$ANALYSIS_MODEL"

python -B run_pipeline_openhands.py modify \
  --better-root "$BETTER_ROOT" \
  --base-dir "artifacts/$TASK/h0" \
  --target-dir "artifacts/$TASK/h1" \
  --plan "artifacts/$TASK/plan_h1.md" \
  --spec "artifacts/$TASK/plan_h1.json" \
  --model "$ANALYSIS_MODEL"

python -B run_pipeline_openhands.py audit \
  --base-dir "artifacts/$TASK/h0" \
  --candidate-dir "artifacts/$TASK/h1" \
  --spec "artifacts/$TASK/plan_h1.json" \
  --output "artifacts/$TASK/h1_audit.json"
```

Do not execute H1 if `h1_audit.json` reports `passed: false`. The JSON plan is the binding edit boundary; Better task setup, task evaluator, dataset, ground truth, OpenHands SDK, and HarnessFix itself are protected.

### 8.5 Re-run H1 on train

```bash
python -B -m task_agent.openhands_agent.bridge run \
  --better-root "$BETTER_ROOT" \
  --base-config "tasks/$TASK/run_harnessfix_train.yaml" \
  --candidate-dir "artifacts/$TASK/h1" \
  --rollout-version harnessfix_h1_train \
  --normalized-output "artifacts/$TASK/train_h1"
```

This is train-side repair evidence, not a held-out estimate. If a redo is needed, use a new target directory and pass the previous comparison JSON through `--redo-feedback`; the modifier intentionally refuses to overwrite an existing candidate.

Create a paired train comparison with the same comparison utility used by the gate:

```bash
python - "artifacts/$TASK/train_h0/results.json" "artifacts/$TASK/train_ids.txt" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
data = json.loads(source.read_text())
destination.write_text("\n".join(data["all_ids"]) + "\n")
PY

python -B run_pipeline_openhands.py gate \
  --baseline-traces "artifacts/$TASK/train_h0/traces" \
  --baseline-eval "artifacts/$TASK/train_h0/results.json" \
  --current-traces "artifacts/$TASK/train_h1/traces" \
  --current-eval "artifacts/$TASK/train_h1/results.json" \
  --ids-file "artifacts/$TASK/train_ids.txt" \
  --plan-spec "artifacts/$TASK/plan_h1.json" \
  --output "artifacts/$TASK/train_compare_h1.json"
```

Use the comparison counts and `net_change` as train-side retry evidence. Its `passed` field is not a validation promotion decision because this command is operating on train data.

### 8.6 Run paired validation and gate

First run both candidates with exactly the same validation YAML:

```bash
python -B -m task_agent.openhands_agent.bridge run \
  --better-root "$BETTER_ROOT" \
  --base-config "tasks/$TASK/run_harnessfix_val.yaml" \
  --candidate-dir "artifacts/$TASK/h0" \
  --rollout-version harnessfix_h0_val \
  --normalized-output "artifacts/$TASK/val_h0"

python -B -m task_agent.openhands_agent.bridge run \
  --better-root "$BETTER_ROOT" \
  --base-config "tasks/$TASK/run_harnessfix_val.yaml" \
  --candidate-dir "artifacts/$TASK/h1" \
  --rollout-version harnessfix_h1_val \
  --normalized-output "artifacts/$TASK/val_h1"
```

Create the gate ID list from the H0 normalized result:

```bash
python - "artifacts/$TASK/val_h0/results.json" "artifacts/$TASK/val_ids.txt" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
data = json.loads(source.read_text())
destination.write_text("\n".join(data["all_ids"]) + "\n")
print(f"wrote {len(data['all_ids'])} ids to {destination}")
PY
```

Run the existing HarnessFix paired gate:

```bash
python -B run_pipeline_openhands.py gate \
  --baseline-traces "artifacts/$TASK/val_h0/traces" \
  --baseline-eval "artifacts/$TASK/val_h0/results.json" \
  --current-traces "artifacts/$TASK/val_h1/traces" \
  --current-eval "artifacts/$TASK/val_h1/results.json" \
  --ids-file "artifacts/$TASK/val_ids.txt" \
  --plan-spec "artifacts/$TASK/plan_h1.json" \
  --output "artifacts/$TASK/val_gate_h1.json"
```

Promote H1 only if both its audit and validation gate pass. H0 and H1 must use the same task data, model alias, model parameters, `n_responses`, and rollout seed convention. Success comes from the official evaluator score, never from `FinishAction` alone.

### 8.7 Run the held-out test once

After selecting the final promoted candidate, run only that candidate on the test YAML:

```bash
python -B -m task_agent.openhands_agent.bridge run \
  --better-root "$BETTER_ROOT" \
  --base-config "tasks/$TASK/run_harnessfix_test.yaml" \
  --candidate-dir "artifacts/$TASK/h1" \
  --rollout-version harnessfix_h1_test \
  --normalized-output "artifacts/$TASK/test_h1"
```

Do not run `analyze`, `aggregate`, `modify`, or a promotion gate on `test_h1`. If test feedback changes the harness, that split has become development data.

## 9. OpenHands fairness checklist

Before comparing H0 and H1, verify:

- The Better Harness commit, OpenHands SDK submodule commit, SDK startup patch checksum, task images, task services, and evaluator commit are recorded. The same SDK patch is applied for every candidate and split.
- The train/validation/test task IDs are disjoint and their input files are hashed.
- H0 and H1 use identical Better run YAML values except the bridge-owned `agent_file` and `rollout_version`.
- All failed train rollouts are analyzed and grouped by stable task instance ID.
- Official evaluator output is the outcome anchor; missing scores fail closed as unsupported/error.
- The repair changes only candidate-owned paths authorized by the JSON plan.
- Validation failures may influence a later repair iteration, but held-out test failures never do.
- No task-specific state observer is added unless it is explicitly part of the candidate-owned instrumentation operator and the experiment reports that intervention.

## 10. Dry runs and troubleshooting

Inspect a Better invocation without executing Docker or evaluation:

```bash
python -B -m task_agent.openhands_agent.bridge run \
  --better-root "$BETTER_ROOT" \
  --base-config "tasks/$TASK/run_harnessfix_train.yaml" \
  --candidate-dir "artifacts/$TASK/h0" \
  --rollout-version dry_run \
  --normalized-output "artifacts/$TASK/dry_run" \
  --dry-run
```

Common failures:

- `AuthenticationError`: confirm the provider-qualified model ID and the corresponding key in the `.env` of the repository whose process is running.
- A Gemini request is sent to the vLLM/OpenAI endpoint: clear `OPENAI_API_BASE` and `LITELLM_API_BASE` for the native Gemini route.
- A self-hosted Qwen request is sent to `api.openai.com`: ensure `OPENAI_API_BASE` is the reachable vLLM URL ending in `/v1`, and that the model ID starts with `openai/`.
- Better reports an unknown model: `model_name` in the run YAML must match a `name` entry in Better's `configs/models.yaml`.
- `task_agent/.../harbor/src not found`: initialize repository contents/submodules and reinstall the vendored Harbor package.
- `.venv/bin/python3` is missing: run inside Linux/WSL2 and create `.venv` there, not in Windows PowerShell.
- Docker cannot access WebArena services: use the network and service startup specified by Better's WebArena run YAML.
- A bridge run resumes an old rollout: use a new `rollout_version` or intentionally clean the corresponding Better result directory after preserving it.

## 11. Artifact policy

The following are runtime artifacts and are ignored by Git: `.env`, raw benchmark data, `traces/`, `logs/`, generated `results/`, failure-analysis outputs, repair memory, improvement plans, and most OpenHands experiment artifacts. Archive the exact configs, commit hashes, split hashes, model IDs, and evaluator outputs separately for a reproducible experiment.

## Citation

If you use HarnessFix, cite:

```bibtex
@article{chen2026failed,
  title={From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws},
  author={Chen, Mengzhuo and Wang, Junjie and Liu, Zhe and Wang, Yawen and Wang, Qing},
  journal={arXiv preprint arXiv:2606.06324},
  year={2026}
}
```
