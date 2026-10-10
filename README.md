# HarnessFix: reproduction and OpenHands integration guide

HarnessFix is a trace-guided pipeline for diagnosing failed LLM-agent trajectories and repairing the harness that produced them. This repository contains the four original benchmark integrations from the HarnessFix paper and an adapter that applies the same analysis/repair stages to an OpenHands Software Agent SDK harness executed by [Better Harnesses, Smaller Models](https://github.com/malusamayo/slm-harness-adaptation).

This README is an operational guide for setting up a new machine, selecting Gemini or Qwen, and running the original and OpenHands closed-loop pipelines.

For setup with Qwen and native Gemini, see the [Vietnamese setup commands](docs/setup_vi.md), [Qwen configuration](configs/qwen.yaml), and [Gemini configuration](configs/gemini.yaml). Use `scripts/configure_models.py` to configure task execution and analysis/repair models, then run the existing pipeline entry points.

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
  ca-certificates curl git git-lfs jq build-essential \
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
git clone <HARNESSFIX_REPOSITORY_URL> HarnessFix-reproduce
cd HarnessFix-reproduce

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

cp .env.example .env
```

The repository deliberately excludes credentials, raw benchmark data, traces, and generated evaluations. Never commit `.env`.

Run local, non-network integration tests:

```bash
python -m unittest discover -s tests -v
python -m json.tool task_agent/model_registry.json >/dev/null
```

## 4. Configure Gemini or Qwen

All HarnessFix model calls use LiteLLM provider-qualified IDs. There are three possible routes:

| Route | HarnessFix CLI model ID | Endpoint/credential |
|---|---|---|
| Self-hosted Qwen with vLLM | `openai/<served-model-name>` | `OPENAI_API_BASE`, `OPENAI_API_KEY` |
| Google AI Studio | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |

### 4.1 Qwen served by vLLM

This is the appropriate route when you already have a Qwen server exposing [vLLM's OpenAI-compatible API](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).

First query the server for the exact model ID. Do not guess it from the Hugging Face repository name because vLLM may have been started with a different `--served-model-name`:

```bash
export OPENAI_API_BASE=http://MODEL_SERVER_HOST:8000/v1
export OPENAI_API_KEY=EMPTY  # Replace if the server enforces a key.

curl -fsS "$OPENAI_API_BASE/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq
```

Suppose `data[0].id` is `Qwen/Qwen3-Coder-30B-A3B-Instruct`. Put this in `.env`:

```dotenv
VLLM_MODEL=openai/Qwen/Qwen3-Coder-30B-A3B-Instruct
OPENAI_API_BASE=http://MODEL_SERVER_HOST:8000/v1
OPENAI_API_KEY=EMPTY
```

The `openai/` prefix tells LiteLLM which protocol to use. It is not part of the model name sent to vLLM. `VLLM_MODEL` is a repository convenience variable; commands still receive it through `--model`.

Load the values and test both model discovery and chat completion:

```bash
set -a
source .env
set +a

export MODEL="$VLLM_MODEL"
python - <<'PY'
import os
from litellm import completion

response = completion(
    model=os.environ["MODEL"],
    api_base=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
    messages=[{"role": "user", "content": "Reply with exactly OK."}],
    max_tokens=8,
)
print(response.choices[0].message.content)
PY
```

OpenAI-compatible chat completion alone is insufficient for the OpenHands track: the vLLM deployment and Qwen chat template must also support OpenAI-format function/tool calls. Verify a request containing `tools` before a full benchmark run. The required vLLM tool parser is model- and vLLM-version-specific.

```bash
python - <<'PY'
import os
from litellm import completion

response = completion(
    model=os.environ["VLLM_MODEL"],
    api_base=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
    messages=[{"role": "user", "content": "Use the health_check tool."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "health_check",
            "description": "Return server health.",
            "parameters": {"type": "object", "properties": {}},
        },
    }],
    tool_choice="required",
    max_tokens=128,
)
tool_calls = response.choices[0].message.tool_calls
assert tool_calls, response
print(tool_calls)
PY
```

The OpenHands agent runs inside Docker. Therefore `OPENAI_API_BASE=http://127.0.0.1:8000/v1` normally points back to the container, not the Ubuntu host. Prefer a DNS name or LAN address reachable from both the host and task containers. If vLLM is on the same host, bind it to an appropriate non-loopback interface and secure access at the network/reverse-proxy layer.

### 4.2 Gemini

For Google AI Studio, leave `OPENAI_API_BASE` and `LITELLM_API_BASE` empty and set:

```dotenv
GEMINI_API_KEY=your-key
HARNESSFIX_VISION_MODEL=gemini/gemini-2.5-flash
```

Use `gemini/gemini-2.5-flash` as `--model` for the original HarnessFix pipelines. The OpenHands bridge in section 8 takes the Better Harness alias instead and reads its model settings from `configs/models.yaml`.

`HARNESSFIX_VISION_MODEL` is used only by GAIA's image tool. A text-only Qwen vLLM deployment cannot handle that tool; point it to Gemini or to a separately served multimodal model. It is not required for the other benchmark tracks.

Use the same execution model name, endpoint, decoding parameters, and server configuration for H0 and every candidate compared against it. The analysis/repair model may differ, but record it with the experiment.

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

Download the pinned verified source and create the manifest-defined splits:

```bash
python data/download_terminal_bench.py
```

The output is:

```text
data/terminal_bench_2_verified/
data/terminal_bench_train/
data/terminal_bench_val/
data/terminal_bench_test/
```

The sampler uses symlinks by default. On filesystems where symlinks are unavailable, use:

```bash
python data/download_terminal_bench.py --copy-splits
```

### 5.4 AppWorld

Build the pinned execution image:

```bash
docker build -t appworld-agent-pypi:latest task_agent/appworld_agent
```

Install/download AppWorld according to its official instructions and set `APPWORLD_ROOT` to a host directory containing its `data/` directory. Build or restore a task cache at `APPWORLD_TASK_CACHE`. The cache must be a JSON object with `train`, `dev`, `test_normal`, and `test_challenge` task lists.

```dotenv
APPWORLD_ROOT=/absolute/path/to/appworld_root
APPWORLD_AGENT_IMAGE=appworld-agent-pypi:latest
APPWORLD_TASK_CACHE=/absolute/path/to/appworld_task_cache.json
```

Then sample:

```bash
python data/sample_appworld.py
```

This creates `appworld_train_90`, `appworld_val_45`, and `appworld_test_90` under `data/`. The cache builder itself is not included, so preserve the cache and its hash if you need an exactly reproducible split on another machine.

## 6. Run the original closed-loop pipelines

Load the provider configuration and choose one provider-qualified model. For the vLLM setup from section 4.1:

```bash
set -a
source .env
set +a
export MODEL="$VLLM_MODEL"
```

Run one repair iteration first. Increase `--max-iterations` only after the smoke run succeeds.

SWE-Bench:

```bash
python run_pipeline_swe.py \
  --model "$MODEL" \
  --analysis-model "$MODEL" \
  --workers 2 \
  --max-iterations 1
```

GAIA:

```bash
python run_pipeline_gaia.py \
  --model "$MODEL" \
  --analysis-model "$MODEL" \
  --workers 2 \
  --concurrency 2 \
  --max-iterations 1
```

Terminal-Bench:

```bash
python run_pipeline_terminal_bench.py \
  --model "$MODEL" \
  --analysis-model "$MODEL" \
  --workers 2 \
  --max-iterations 1
```

AppWorld:

```bash
python run_pipeline_appworld.py \
  --model "$MODEL" \
  --analysis-model "$MODEL" \
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
git clone --recurse-submodules https://github.com/malusamayo/slm-harness-adaptation.git
cd slm-harness-adaptation
git submodule update --init --recursive
uv sync

export BETTER_ROOT="$PWD"
export HARNESSFIX_ROOT="$(cd ../HarnessFix-reproduce && pwd)"
```

For strict reproduction, record and reuse both repository revisions:

```bash
git -C "$BETTER_ROOT" rev-parse HEAD
git -C "$HARNESSFIX_ROOT" rev-parse HEAD
```

Better Harness and the OpenHands analysis, aggregation, and modification stages now read the same model entries from `$BETTER_ROOT/configs/models.yaml`. API keys and base URLs can be literal values in that file or `${VAR}` references resolved from `$BETTER_ROOT/.env` or the environment. Use LiteLLM provider-qualified IDs for OpenAI-compatible endpoints:

```bash
cat "$BETTER_ROOT/configs/models.yaml"
```

Use the alias in both Better run YAML and HarnessFix's OpenHands `--model` option:

| Location | Gemini | Self-hosted Qwen/vLLM |
|---|---|---|
| Model alias | `gemini-3.1-pro-low` | `qwen-vllm` |
| LiteLLM model ID in Better config | `openai/ag/gemini-3.1-pro-low` | `openai/Qwen/Qwen3.5-9B` |

Edit the selected Better task YAML so `model_name` is one of those aliases. The `openai/` prefix selects the OpenAI-compatible protocol; it is stripped before the model ID is sent to the configured endpoint. Keep `model_name`, `prompt_name`, `n_responses`, runtime limits, task IDs, and data fixed between H0 and H1.

Generation settings in `configs/qwen.yaml` and `configs/gemini.yaml` are synchronized by `scripts/configure_models.py` into Better's `configs/models.yaml` and the SWE model registry. Qwen disables thinking with `extra_body.chat_template_kwargs.enable_thinking: false`, following the [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B). The packed OpenHands launcher preserves temperature in this mode. Gemini uses temperature `0.2` and `reasoning_effort: low`; [Gemini 3.1 Pro cannot disable thinking entirely](https://ai.google.dev/gemini-api/docs/generate-content/thinking). A null reasoning effort leaves provider defaults in effect.

Build the task images you need:

```bash
cd "$BETTER_ROOT"
env UID="$(id -u)" docker compose build \
  woocommerce_stock_alert_s2l \
  machine_operating_s2l \
  refactorbench \
  webarena
```

The image creates a non-root `appuser` with this UID. If `id -u` prints `0`, run the build and benchmark as a regular host user so the UID is nonzero and mounted workspaces remain writable. For a build-only check while running as root, use `env UID=1000 docker compose build machine_operating_s2l`; the later rollout may still need workspace ownership adjusted for that UID.

Additional task setup remains owned by Better Harness:

- Stock Alert and Machine Operating use its LOCA-bench submodule and task services.
- RefactorBench requires a local clone of `microsoft/RefactorBench` and the correct repository path in the task data.
- WebArena requires `webarena-verified`; start the shopping-admin environment and network as required by `tasks/webarena/run.yaml`.
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

Use one experiment config for Stock Alert:

```bash
cd /root/HarnessFix-reproduce
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
export ANALYSIS_MODEL=qwen-vllm  # or gemini-3.1-pro-low
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

- The Better Harness commit, OpenHands SDK submodule commit, task images, task services, and evaluator commit are recorded.
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
