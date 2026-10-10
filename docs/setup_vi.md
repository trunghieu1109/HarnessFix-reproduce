# Cài đặt và chạy HarnessFix

Hướng dẫn cho Bash trên Linux/Ubuntu hoặc WSL2. Dùng checkout reproduce của cả
HarnessFix và SLM/Better Harness. Runtime và evaluator OpenHands chạy ở phía
SLM; HarnessFix quản lý candidate, HTIR, analysis, aggregate, repair, audit,
train/validation, promotion và lựa chọn candidate trước test.

## Nội dung

1. [Công cụ host và hai checkout](#1-công-cụ-host-và-hai-checkout)
2. [Môi trường HarnessFix](#2-môi-trường-harnessfix)
3. [Môi trường SLM/OpenHands và SDK patch](#3-môi-trường-slmopenhands-và-sdk-patch)
4. [Cấu hình model và kiểm tra kết nối](#4-cấu-hình-model-và-kiểm-tra-kết-nối)
5. [Setup và chạy bốn benchmark OpenHands](#5-setup-và-chạy-bốn-benchmark-openhands)
6. [Setup và chạy các benchmark gốc](#6-setup-và-chạy-các-benchmark-gốc)
7. [Resume, test cuối và đọc kết quả](#7-resume-test-cuối-và-đọc-kết-quả)
8. [Các lỗi setup thường gặp](#8-các-lỗi-setup-thường-gặp)

## 1. Công cụ host và hai checkout

Cần Git, Git LFS, Docker Engine + Compose v2, curl, jq, ripgrep, build tools và uv.
Các lệnh cài host Ubuntu 24.04 nằm ở [README, mục 2.1](../README.md#21-ubuntu-2404-setup).
Trên WSL2, bật Docker Desktop WSL integration. Host cần truy cập được model
server; GPU/vLLM nằm ở máy khác thì máy chạy HarnessFix không cần cài vLLM.

Kiểm tra:

```bash
git --version
git lfs install
uv --version
docker info
docker compose version
curl --version
jq --version
```

Chọn đường dẫn. Trên máy đang dùng, hai checkout nằm dưới `/root`; các biến
bên dưới cũng dùng được khi home của bạn nằm ở vị trí khác:

```bash
export HARNESSFIX_ROOT="$HOME/HarnessFix-reproduce"
export BETTER_ROOT="$HOME/slm-harness-adaptation-reproduce"
export BETTER_HARNESS_ROOT="$BETTER_ROOT"
```

Nếu chưa có checkout, clone bản reproduce tương ứng; checkout đã có thì bỏ qua
lệnh clone của repo đó:

```bash
git clone https://github.com/trunghieu1109/HarnessFix-reproduce.git "$HARNESSFIX_ROOT"
git clone --recurse-submodules \
  https://github.com/trunghieu1109/slm-harness-adaptation-reproduce.git "$BETTER_ROOT"
```

Ghi lại revision khi tái lập một thí nghiệm:

```bash
git -C "$HARNESSFIX_ROOT" rev-parse HEAD
git -C "$BETTER_ROOT" rev-parse HEAD
```

Chỉ chạy benchmark đã setup. Các lệnh chuẩn bị dữ liệu ở mục 6 cần kết nối
mạng và có thể tái tạo thư mục split; thực hiện trước khi bắt đầu run mới.

## 2. Môi trường HarnessFix

HarnessFix dùng Python 3.12; Harbor vendored cũng yêu cầu Python >=3.12.
SLM có môi trường Python riêng ở mục 3. Tạo `.venv` ở repo HarnessFix nếu chưa có:

```bash
cd "$HARNESSFIX_ROOT"
uv python install 3.12
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt
if [ ! -f .env ]; then cp .env.example .env; fi
chmod 600 .env
```

`requirements.txt` đã cài editable mini-swe-agent, dùng cho analysis/repair và
SWE. Cài thêm dependency của benchmark bạn cần:

```bash
# SWE-Bench: evaluator đúng phiên bản của driver.
uv pip install --python .venv/bin/python 'swebench==4.1.0'

# GAIA: agent và công cụ search/file/vision.
uv pip install --python .venv/bin/python -e task_agent/open_deep_research

# Terminal-Bench: Harbor từ source vendored trong reproduce.
uv pip install --python .venv/bin/python -e task_agent/terminal_bench_agent/harbor
```

AppWorld chạy package `appworld==0.1.3.post1` trong Docker Python 3.11; dùng
Dockerfile của repo ở mục 6.4. OpenHands dùng SDK trong SLM, không cài SDK
Python 3.14 vào môi trường HarnessFix 3.12.

Kiểm tra các entry point; `--help` không gọi model hay chạy benchmark:

```bash
.venv/bin/python -B scripts/configure_models.py --help
.venv/bin/python -B run_pipeline_openhands.py run --help
.venv/bin/python -B run_pipeline_swe.py --help
.venv/bin/python -B run_pipeline_gaia.py --help
.venv/bin/python -B run_pipeline_terminal_bench.py --help
.venv/bin/python -B run_pipeline_appworld.py --help
.venv/bin/python -m swebench.harness.run_evaluation --help
```

## 3. Môi trường SLM/OpenHands và SDK patch

Các submodule phải có trước khi cài dependency hoặc build image:


```bash
cd "$BETTER_ROOT"
git submodule sync --recursive
git submodule update --init --recursive software-agent-sdk LOCA-bench
```

Bản reproduce lưu các sửa đổi Docker startup của SDK tại
[`patches/software-agent-sdk-docker-startup.patch`](../patches/software-agent-sdk-docker-startup.patch),
dựa trên commit SDK `89c21968922afc746bca8a653712038d5a38d6aa`. Patch cleanup
container khi startup thất bại, dùng đồng hồ monotonic cho health check, giữ
thông tin lỗi kết nối và tránh thread đọc log tự gọi `join()`. Patch cũng chứa
các test tương ứng. Giữ nguyên commit submodule và áp dụng patch cục bộ;
không cần push vào repo SDK dùng chung.

Trên checkout chưa có bản sửa, kiểm tra rồi áp dụng:

```bash
git -C "$BETTER_ROOT/software-agent-sdk" apply --check \
  "$HARNESSFIX_ROOT/patches/software-agent-sdk-docker-startup.patch"
git -C "$BETTER_ROOT/software-agent-sdk" apply \
  "$HARNESSFIX_ROOT/patches/software-agent-sdk-docker-startup.patch"
```

Nếu checkout đã có bản sửa, bỏ qua hai lệnh trên. Kiểm tra
patch đã được áp dụng bằng lệnh sau; thành công sẽ không in lỗi:

```bash
git -C "$BETTER_ROOT/software-agent-sdk" apply --reverse --check \
  "$HARNESSFIX_ROOT/patches/software-agent-sdk-docker-startup.patch"
```

Submodule hiện `-dirty` sau khi áp dụng là bình thường; file patch trong
HarnessFix lưu các sửa đổi để tái tạo trên máy khác. Dùng cùng patch cho H0,
mọi candidate và cả train/validation/test. Ghi lại commit SDK và checksum patch
cùng commit của hai repo khi lưu thông tin thí nghiệm:

```bash
git -C "$BETTER_ROOT/software-agent-sdk" rev-parse HEAD
sha256sum "$HARNESSFIX_ROOT/patches/software-agent-sdk-docker-startup.patch"
```

Cài dependency sau khi áp dụng patch. Better Harness dùng
`openhands-workspace` từ checkout SDK theo dạng editable:

```bash
cd "$BETTER_ROOT"
uv sync --python 3.14

test -e .vertex-ai.json || printf '%s\n' '{}' > .vertex-ai.json

```

HarnessFix và Better Harness có hai virtual environment riêng.
Kiểm tra model aliases sau khi cấu hình ở mục 4.

## 4. Cấu hình model và kiểm tra kết nối

### 4.1. Profile, endpoint và credential

Hai profile mẫu là [Qwen](../configs/qwen.yaml) và [Gemini](../configs/gemini.yaml).
Các experiment OpenHands dùng alias `qwen-vllm` cho task và `gemini-api` cho
analysis/aggregate/modify. CLI benchmark gốc dùng model ID có provider prefix.

| Kiểu kết nối | Model ID | Endpoint | Key |
|---|---|---|---|
| Qwen API tương thích OpenAI | `openai/<served-model-id>` | URL kết thúc `/v1` | `QWEN_API_KEY` |
| Gemini qua gateway tương thích OpenAI | `openai/<gateway-model-id>` | URL gateway kết thúc `/v1` | `GEMINI_API_KEY` |
| Gemini native | `gemini/<native-model-id>` | `api_base: null` | `GEMINI_API_KEY` |

Prefix `openai/` chọn giao thức HTTP. Model ID phía sau phải khớp server;
kiểm tra Qwen bằng `/v1/models`. Endpoint phải truy cập được từ host và Docker.
Với Docker, `127.0.0.1` thường trỏ vào chính container; chọn DNS/IP truy cập
được từ container.

Điền endpoint của bạn và key. Ví dụ dưới đây giữ Qwen/Gemini model ID trong
profile mẫu; sửa trường `model` nếu server cung cấp ID khác:

```bash
cd "$HARNESSFIX_ROOT"
export QWEN_BASE_URL='http://MODEL_SERVER_HOST:8000/v1'
export GEMINI_BASE_URL='https://GEMINI_GATEWAY_HOST/v1'
export QWEN_API_KEY='EMPTY'  # Thay nếu server yêu cầu key.
read -r -s -p 'Gemini/gateway API key: ' GEMINI_API_KEY
printf '\n'
export GEMINI_API_KEY

curl -fsS "$QWEN_BASE_URL/models" \
  -H "Authorization: Bearer $QWEN_API_KEY" | jq '.data[].id'

.venv/bin/python - <<'PY'
import os
from pathlib import Path
import yaml

for name, variable in (('qwen', 'QWEN_BASE_URL'), ('gemini', 'GEMINI_BASE_URL')):
    profile = yaml.safe_load(Path(f'configs/{name}.yaml').read_text())
    profile['api_base'] = os.environ[variable].rstrip('/')
    Path(f'configs/{name}.local.yaml').write_text(yaml.safe_dump(profile, sort_keys=False))
PY
```

Thay `MODEL_SERVER_HOST` và `GEMINI_GATEWAY_HOST` bằng địa chỉ thật trước khi
chạy. `api_base: QWEN_BASE_URL` trong template là placeholder; script
`configure_models.py` cần URL cụ thể, không tự expand tên biến trong profile.
Profile `.local.yaml` giữ key ở ngoài YAML; credential được lưu trong `.env`.

Nếu dùng Gemini native, sửa `configs/gemini.local.yaml`: đặt model thành
`gemini/<native-model-id>`, `api_base: null`, và chọn `reasoning_effort` phù hợp
với model đó. Key vẫn là `GEMINI_API_KEY`.

### 4.2. Qwen chạy task, Gemini chạy analysis/repair

```bash
cd "$HARNESSFIX_ROOT"
.venv/bin/python -B scripts/configure_models.py \
  --config configs/qwen.local.yaml \
  --analysis-config configs/gemini.local.yaml \
  --better-root "$BETTER_ROOT"

set -a
source .env
set +a
export BETTER_ROOT="$BETTER_HARNESS_ROOT"
```

Script cập nhật `.env` ở hai repo, SWE model registry, model kwargs của
analysis/repair SWE, và `configs/models.yaml` bên SLM. Nó giữ prompt, các alias
khác và task model khi cập nhật analysis. OpenHands lấy connection/settings
cho từng alias từ registry SLM; SWE dùng model registry HarnessFix.

Để dùng một model cho cả task và repair, bỏ `--analysis-config`, ví dụ:

```bash
.venv/bin/python -B scripts/configure_models.py \
  --config configs/qwen.local.yaml --better-root "$BETTER_ROOT"
```

Nếu dùng một model với OpenHands, đổi cả `models.task` và `models.analysis`
trong experiment YAML sang alias vừa cấu hình; WebArena còn có `eval_model`.
Alias mặc định trong YAML không tự đổi theo lệnh setup.

Cập nhật riêng analysis, giữ task model hiện có:

```bash
.venv/bin/python -B scripts/configure_models.py \
  --config configs/gemini.local.yaml --analysis-only --better-root "$BETTER_ROOT"
```

Load lại `.env` sau mỗi lần cấu hình. Các OpenAI-compatible model được script
đăng ký pricing 0; USD trong báo cáo không đo chi phí server thực tế.
Không commit `.env` hoặc API key.

### 4.3. Kiểm tra alias, chat và tool calls

Kiểm tra alias trong môi trường SLM, không in credential:

```bash
cd "$BETTER_ROOT"
uv run python - <<'PY'
from src.utils import LM_DICT
assert {'qwen-vllm', 'gemini-api'} <= LM_DICT.keys()
print('Task/analysis aliases OK')
PY
```

Lệnh sau có gọi API thật: test chat của cả hai model và native function calls
của task model trước khi chạy OpenHands. Task server phải hỗ trợ tool calls;
mini-swe analysis/repair dùng command được viết trong response text.

```bash
cd "$HARNESSFIX_ROOT"
.venv/bin/python - <<'PY'
from task_agent.openhands_agent.model_config_bridge import load_model, selected_model_kwargs
import litellm
import os

for alias in ('qwen-vllm', 'gemini-api'):
    os.environ['HARNESSFIX_MODEL_ALIAS'] = alias
    model = load_model(alias)
    kwargs = selected_model_kwargs()
    kwargs['max_tokens'] = 1024
    response = litellm.completion(model=model['model'],
        messages=[{'role': 'user', 'content': 'Reply with exactly OK.'}], **kwargs)
    print(alias, response.choices[0].message.content)

os.environ['HARNESSFIX_MODEL_ALIAS'] = 'qwen-vllm'
model = load_model('qwen-vllm')
kwargs = selected_model_kwargs()
kwargs['max_tokens'] = 128
response = litellm.completion(model=model['model'],
    messages=[{'role': 'user', 'content': 'Use the health_check tool.'}],
    tools=[{'type': 'function', 'function': {'name': 'health_check',
        'description': 'Return server health.',
        'parameters': {'type': 'object', 'properties': {}}}}],
    tool_choice='required', **kwargs)
assert response.choices[0].message.tool_calls
print('Task tool calls OK')
PY
```

`max_input_tokens`/`max_output_tokens` phải phù hợp context của deployment.
Chừa budget cho output và phần overhead; tăng số input trong profile không
làm tăng context thực tế của server. Cấu hình model trước run mới: resume
OpenHands kiểm tra hash của model settings.

### 4.4. Model cho GAIA, AppWorld và Terminal-Bench

Các driver này nhận `--model` và `--analysis-model`, nhưng connection kwargs
của analysis/modifier còn đọc từ YAML riêng. Script setup chung chưa tự cập
nhật sáu YAML đó. Mẫu bên dưới dùng cùng task model cho analysis/repair của
ba benchmark này; SWE/OpenHands vẫn dùng hai model như mục 4.2.

```bash
cd "$HARNESSFIX_ROOT"
set -a
source .env
set +a
export LEGACY_ANALYSIS_MODEL="$HARNESSFIX_TASK_MODEL"

.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
import yaml
from dotenv import set_key

registry = json.loads(Path('task_agent/model_registry.json').read_text())
kwargs = dict(registry[os.environ['LEGACY_ANALYSIS_MODEL']]['model_kwargs_override'])
kwargs.pop('api_key', None)  # Key lấy từ environment, không ghi vào YAML.
set_key('.env', 'LEGACY_ANALYSIS_MODEL', os.environ['LEGACY_ANALYSIS_MODEL'])
for mode in ('gaia', 'appworld', 'terminal_bench'):
    for relative in (f'failure_analysis/analysis_config_{mode}.yaml',
                     f'enhancement_implementation/config_{mode}.yaml'):
        path = Path(relative)
        text = path.read_text()
        document = yaml.compose(text)
        model = next(value for key, value in document.value if key.value == 'model')
        key, value = next((key, value) for key, value in model.value if key.value == 'model_kwargs')
        start = key.start_mark.index - key.start_mark.column
        block = yaml.safe_dump({'model_kwargs': kwargs}, sort_keys=False)
        block = ''.join('  ' + line + '\n' for line in block.splitlines())
        path.write_text(text[:start] + block + text[value.end_mark.index:])
        print('Configured', relative)
PY
```

Mẫu này chỉ thay block `model_kwargs`, giữ nguyên prompt. Task runner dùng
`OPENAI_API_BASE`/`OPENAI_API_KEY` từ `.env`; Gemini native dùng
`GEMINI_API_KEY`. Nếu chọn native Gemini cho task, để `OPENAI_API_BASE`,
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LITELLM_API_BASE` và `LITELLM_API_KEY`
rỗng trong `.env` trước khi load lại, để runner dùng key Google.

Để dùng hai model qua một gateway chung, đặt cả hai profile vào cùng endpoint
và cùng credential; thay dòng export trong block trên bằng
`export LEGACY_ANALYSIS_MODEL="$HARNESSFIX_ANALYSIS_MODEL"` rồi cấu hình lại.
Model được chọn cũng được lưu vào `.env` để các lệnh chạy dùng đúng ID.
Hai endpoint/key riêng cho GAIA/AppWorld/Terminal
cần thêm cấu hình connection theo stage; chỉ đổi `--analysis-model` chưa đủ.
Không áp dụng giới hạn này cho SWE hoặc OpenHands, vốn đã resolve connection
riêng theo model.

## 5. Setup và chạy bốn benchmark OpenHands

### 5.1. Build image và task services

Chọn các image bạn cần. Compose của reproduce dùng `BENCHMARK_UID`, không
phải biến Bash readonly `UID`. Với host root, dùng UID 1000 cho appuser:

```bash
cd "$BETTER_ROOT"
benchmark_uid="$(id -u)"
if [ "$benchmark_uid" -eq 0 ]; then benchmark_uid=1000; fi
BENCHMARK_UID="$benchmark_uid" docker compose build \
  woocommerce_stock_alert_s2l machine_operating_s2l refactorbench webarena
```

Có thể build riêng một service, ví dụ `docker compose build refactorbench`.
Luồng OpenHands hiện tại dùng image thường; image `_codex` dành cho runtime
Codex khác. Stock Alert và Machine Operating cần submodule LOCA-bench đã init;
setup của task tạo mock services/MCP theo từng rollout.

### 5.2. Source repository cho RefactorBench

Clone snapshot benchmark chính thức. Dataset có `repo_path` từ máy cũ;
`REFACTORBENCH_REPOS_DIR` cho runtime tìm repository local theo `repo_name`.

```bash
cd "$BETTER_ROOT"
mkdir -p external
if [ ! -d external/RefactorBench/.git ]; then
  git clone --depth 1 https://github.com/microsoft/RefactorBench.git external/RefactorBench
fi
export REFACTORBENCH_REPOS_DIR="$BETTER_ROOT/external/RefactorBench/repositories"
uv python install 3.11

uv run python - <<'PY'
import json
import os
from pathlib import Path

data = json.loads(Path('data/refactorbench.json').read_text())
root = Path(os.environ['REFACTORBENCH_REPOS_DIR'])
missing = sorted({row['repo_name'] for row in data if not (root / row['repo_name']).is_dir()})
assert not missing, f'Missing repositories: {missing}'
print('RefactorBench repositories OK')
PY
```

Python 3.11 được evaluator SLM gọi qua `uv run --no-project --python 3.11`.
Giữ biến `REFACTORBENCH_REPOS_DIR` trong shell chạy HarnessFix; subprocess SLM
kế thừa nó. Mở terminal mới thì export lại.

### 5.3. Website cho WebArena

Config dùng subset Shopping Admin, `prompt_name: shopping_admin`, và
`start_servers: false`. Khởi động site trước khi chạy pipeline:

```bash
cd "$BETTER_ROOT"
uvx webarena-verified env start --site shopping_admin

docker ps --format '{{.Names}} {{.Status}} {{.Ports}}' \
  | rg webarena_verified_shopping_admin
curl -sS -o /dev/null -w 'status=%{http_code}\n' http://localhost:7780/admin
```

Container cần ở trạng thái Up và HTTP GET trả 200. SLM nối site và agent vào
`webarena-net`, thay URL placeholder trong task. Nếu CLI start còn giữ
foreground, chạy lệnh start ở terminal khác rồi chạy HarnessFix khi site sẵn
sàng. `eval_model: gemini-api` phải có trong registry vì evaluator dùng nó cho
các task fuzzy_match.

### 5.4. Config và lệnh chạy

Các config đầy đủ chọn test 30 sample đầu, train 10 sample tiếp theo và val 10
sample tiếp theo, không shuffle. Train/val dùng 1 response/sample; test dùng
2 response/sample, tổng 60 rollout của candidate cuối.

| Benchmark | Experiment config | Agent batch | Eval batch |
|---|---|---:|---:|
| Stock Alert | [stock_alert.yaml](../configs/stock_alert.yaml) | 6 | 6 |
| Machine Operating | [machine_operating_batch4.yaml](../configs/machine_operating_batch4.yaml) | 4 | 6 |
| RefactorBench | [refactorbench_batch4.yaml](../configs/refactorbench_batch4.yaml) | 4 | 6 |
| WebArena Shopping Admin | [webarena.yaml](../configs/webarena.yaml) | 4 | 6 |

Các config này có analysis tối đa 30 model calls mỗi diagnosis, tối đa 3 vòng
repair, dừng sau 2 candidate liên tiếp không promote, và `run_test: true`.
`analysis_step_limit` là budget cho từng diagnosis, không phải số task cần
phân tích. Có thể dừng sớm khi current-base train đã giải hết.

Chọn đường dẫn config từ bảng trên; ví dụ Stock Alert:

```bash
cd "$HARNESSFIX_ROOT"
set -a
source .env
set +a
export BETTER_ROOT="$BETTER_HARNESS_ROOT"

export CONFIG=configs/stock_alert.yaml
```

Sau khi chọn config, kiểm tra rồi chạy:

```bash
.venv/bin/python -B run_pipeline_openhands.py run --config "$CONFIG" --dry-run
.venv/bin/python -B run_pipeline_openhands.py run --config "$CONFIG"
```

Lệnh chạy trực tiếp cho từng benchmark:

```bash
.venv/bin/python -B run_pipeline_openhands.py run --config configs/stock_alert.yaml
.venv/bin/python -B run_pipeline_openhands.py run --config configs/machine_operating_batch4.yaml
.venv/bin/python -B run_pipeline_openhands.py run --config configs/refactorbench_batch4.yaml
.venv/bin/python -B run_pipeline_openhands.py run --config configs/webarena.yaml
```

Chạy từng lệnh sau khi task tương ứng đã setup. Config
[stock_alert_smoke.yaml](../configs/stock_alert_smoke.yaml) dùng train/val 10/10,
test 5 x 1 response, analysis 10 calls và 1 vòng repair:

```bash
.venv/bin/python -B run_pipeline_openhands.py run --config configs/stock_alert_smoke.yaml
```

Pipeline tự materialize H0, chạy val/train, compile HTIR và analysis các failure,
aggregate kế hoạch, tạo candidate mới, modify/audit/check, chạy train/val,
compare/promote, ghi memory, rồi chọn candidate để test. Runtime/evaluator vẫn
ở SLM. Test feedback được giữ ngoài repair loop.

## 6. Setup và chạy các benchmark gốc

### 6.1. SWE-Bench Verified

```bash
cd "$HARNESSFIX_ROOT"
.venv/bin/python -m swebench.harness.run_evaluation --help
.venv/bin/python -B data/sample_swebench.py
```

Sampler mặc định tạo train 100, val 50, test 100 trong
`data/verified_{train_100,val_50,test_100}`, với IDs disjoint. Có thể chọn
`--train-n`, `--val-n`, `--test-n` và các option output để tạo split nhỏ hơn.
Driver SWE hỗ trợ `--train-dir` và `--val-dir` cho split riêng.

Pre-pull Docker images là tùy chọn:

```bash
bash data/pull_swebench_images.sh data/verified_train_100 test 4
bash data/pull_swebench_images.sh data/verified_val_50 test 4
bash data/pull_swebench_images.sh data/verified_test_100 test 4
```

```bash
set -a
source .env
set +a
.venv/bin/python -B run_pipeline_swe.py \
  --model "$HARNESSFIX_TASK_MODEL" --analysis-model "$HARNESSFIX_ANALYSIS_MODEL" \
  --workers 4 --max-iterations 3 --dry-run
.venv/bin/python -B run_pipeline_swe.py \
  --model "$HARNESSFIX_TASK_MODEL" --analysis-model "$HARNESSFIX_ANALYSIS_MODEL" \
  --workers 4 --max-iterations 3
```

Evaluator pin 4.1.0 hỗ trợ tham số và layout report của driver. Smoke có thể
đặt `--workers 2 --max-iterations 1 --run-label smoke`; dùng các thư mục split
nhỏ riêng nếu muốn giảm số instance.

### 6.2. GAIA

Cài package GAIA ở mục 2 và cấu hình connection YAML theo mục 4.4. Cần quyền
truy cập [GAIA trên Hugging Face](https://huggingface.co/datasets/gaia-benchmark/GAIA),
`HF_TOKEN`, cùng `SERPAPI_API_KEY` cho web search (`SERPER_API_KEY` là fallback).
Điền các key trong `.env`. Đặt `HARNESSFIX_VISION_MODEL` thành model có khả năng
xử lý ảnh trên endpoint task đang dùng; nếu task server hỗ trợ ảnh, có thể dùng
cùng model ID. Giá trị mặc định `gemini/gemini-2.5-flash` dùng key Google native;
key Gemini gateway không dùng được cho route native đó. Kiểm tra model vision
trước các task có ảnh.

Kiểm tra import của task runner sau khi cài package GAIA. Nếu báo thiếu
`smolagents`, chạy lại lệnh cài editable GAIA ở mục 2:

```bash
cd "$HARNESSFIX_ROOT"
PYTHONPATH=task_agent/open_deep_research/src \
  .venv/bin/python -B task_agent/open_deep_research/run_gaia_entry.py --help
```

```bash
cd "$HARNESSFIX_ROOT"
set -a
source .env
set +a
.venv/bin/python -B data/sample_gaia.py

.venv/bin/python -B run_pipeline_gaia.py \
  --model "$HARNESSFIX_TASK_MODEL" --analysis-model "$LEGACY_ANALYSIS_MODEL" \
  --workers 4 --concurrency 4 --max-iterations 3 --dry-run
.venv/bin/python -B run_pipeline_gaia.py \
  --model "$HARNESSFIX_TASK_MODEL" --analysis-model "$LEGACY_ANALYSIS_MODEL" \
  --workers 4 --concurrency 4 --max-iterations 3
```

Sampler tạo `gaia_train_60`, `gaia_val_30`, `gaia_test_60`, phân tầng theo level
với IDs không trùng, từ official validation có ground truth. Đây là held-out
split local, không phải official test leaderboard. Runner gốc dùng các thư mục
cố định này; không có `--train-dir`/`--val-dir` như SWE.

### 6.3. Terminal-Bench 2.0

Cài Harbor vendored ở mục 2, cấu hình connection YAML ở mục 4.4, và kiểm tra
Docker daemon. Checkout reproduce không kèm `data/terminal_bench_splits.json`;
đường setup mới dưới đây chọn source GitHub chính thức và split theo seed:

```bash
cd "$HARNESSFIX_ROOT"
.venv/bin/python -B data/download_terminal_bench.py --source github --skip-sampling
.venv/bin/python -B data/sample_terminal_bench.py \
  --source data/terminal_bench_2_verified --random-split \
  --train 34 --val 17 --test 34 --seed 20260522

git -C data/terminal_bench_2_verified rev-parse HEAD
```

Ghi lại revision và `instance_ids.txt` để tái lập split. Tên thư mục source là
layout driver hiện tại; command trên lấy source GitHub, không tự chuyển thành
variant ZAI verified. Muốn tái lập split verified cũ, cung cấp manifest có
`source_commit` và các task IDs, rồi dùng:

```bash
.venv/bin/python -B data/download_terminal_bench.py \
  --source zai_verified --manifest /absolute/path/to/terminal_bench_splits.json
```

Chỉ chọn một cách tải source vào cùng destination. Downloader từ chối thư mục
đã tồn tại; với source đã tải, chỉ gọi sampler. Nếu không dùng symlink được,
thêm `--copy` vào `sample_terminal_bench.py` (hoặc `--copy-splits` vào downloader).

```bash
set -a
source .env
set +a
.venv/bin/python -B run_pipeline_terminal_bench.py \
  --model "$HARNESSFIX_TASK_MODEL" --analysis-model "$LEGACY_ANALYSIS_MODEL" \
  --workers 4 --max-iterations 3 --dry-run
.venv/bin/python -B run_pipeline_terminal_bench.py \
  --model "$HARNESSFIX_TASK_MODEL" --analysis-model "$LEGACY_ANALYSIS_MODEL" \
  --workers 4 --max-iterations 3
```

### 6.4. AppWorld

Dockerfile pin AppWorld `0.1.3.post1` trong Python 3.11 và chạy bước install.
Quy trình tải data dựa trên [hướng dẫn AppWorld chính thức](https://github.com/StonyBrookNLP/appworld/blob/main/README.md);
command bên dưới dùng CLI của bản pin trong Dockerfile.

```bash
cd "$HARNESSFIX_ROOT"
export APPWORLD_ROOT="$HARNESSFIX_ROOT/appworld_root"
export APPWORLD_AGENT_IMAGE=appworld-agent-pypi:latest
export APPWORLD_TASK_CACHE="$HARNESSFIX_ROOT/data/appworld_task_cache.json"
mkdir -p "$APPWORLD_ROOT" data

.venv/bin/python - <<'PY'
import os
from dotenv import set_key
for name in ('APPWORLD_ROOT', 'APPWORLD_AGENT_IMAGE', 'APPWORLD_TASK_CACHE'):
    set_key('.env', name, os.environ[name])
PY

docker build -t "$APPWORLD_AGENT_IMAGE" task_agent/appworld_agent
```

Tải data vào root mới. CLI download của bản pin tái tạo `data/`; bỏ qua nếu đã
có dataset bạn muốn giữ để tái lập:

```bash
if [ ! -d "$APPWORLD_ROOT/data/tasks" ]; then
  docker run --rm \
    -v "$APPWORLD_ROOT:/workspace/appworld_root" \
    -e APPWORLD_ROOT=/workspace/appworld_root \
    "$APPWORLD_AGENT_IMAGE" \
    python3 -m appworld.cli download data --root /workspace/appworld_root
fi
```

Tạo cache metadata đúng schema sampler/runner. Dùng package thật trong Docker;
module AppWorld shim vendored trên host không dùng để tải task IDs. Cache chỉ
lấy task instruction/supervisor và các app được phép, không lấy đáp án:

```bash
docker run --rm -i \
  -v "$APPWORLD_ROOT:/workspace/appworld_root" \
  -v "$HARNESSFIX_ROOT/data:/harnessfix-data" \
  -e APPWORLD_ROOT=/workspace/appworld_root \
  "$APPWORLD_AGENT_IMAGE" python3 - <<'PY'
import json
from pathlib import Path
from appworld.common.path_store import path_store
from appworld.task import Task, load_task_ids

path_store.update_root('/workspace/appworld_root')
cache = {}
for split in ('train', 'dev', 'test_normal', 'test_challenge'):
    records = []
    for task_id in load_task_ids(split):
        task = Task.load(task_id, load_ground_truth=False)
        records.append({
            'task_id': task_id,
            'instruction': task.instruction,
            'supervisor': {name: getattr(task.supervisor, name) for name in
                ('first_name', 'last_name', 'email', 'phone_number')},
            'required_apps': task.allowed_apps,
            'difficulty': '',
        })
        task.close()
    cache[split] = records
    print(split, len(records))
Path('/harnessfix-data/appworld_task_cache.json').write_text(json.dumps(cache, indent=2) + '\n')
PY

.venv/bin/python -B data/sample_appworld.py
```

Sampler tạo `appworld_train_90`, `appworld_val_45`, `appworld_test_90` từ
train/dev và test_normal+test_challenge. Lưu cache và checksum khi tái lập:

```bash
sha256sum "$APPWORLD_TASK_CACHE"
```

Sau khi cấu hình connection YAML theo mục 4.4, chạy:

```bash
set -a
source .env
set +a
.venv/bin/python -B run_pipeline_appworld.py \
  --model "$HARNESSFIX_TASK_MODEL" --analysis-model "$LEGACY_ANALYSIS_MODEL" \
  --workers 4 --concurrency 4 --max-iterations 3 --dry-run
.venv/bin/python -B run_pipeline_appworld.py \
  --model "$HARNESSFIX_TASK_MODEL" --analysis-model "$LEGACY_ANALYSIS_MODEL" \
  --workers 4 --concurrency 4 --max-iterations 3
```

## 7. Resume, test cuối và đọc kết quả

### 7.1. OpenHands

Dry-run kiểm tra config, split IDs và model registry, không chạy Docker hay gọi
model; vẫn cần dataset và registry đã setup. Các repository/source prerequisite
như RefactorBench và WebArena phải được kiểm tra riêng ở mục 5.

Mỗi run in ra thư mục mới dưới `artifacts/openhands/`. Lưu đúng đường dẫn đó:

```bash
cd "$HARNESSFIX_ROOT"
export CONFIG=configs/machine_operating_batch4.yaml  # Config của run cần resume.
export RUN_DIR=/absolute/path/to/artifacts/openhands/THE_RUN
.venv/bin/python -B run_pipeline_openhands.py run \
  --config "$CONFIG" --run-dir "$RUN_DIR"
```

Các stage hoàn tất được dùng lại; config, data, model, source và artifact hashes
phải khớp. Chạy lệnh mới không có `--run-dir` tạo thí nghiệm mới. Không tăng
`max_iterations` trên config của run đã đóng rồi coi đó là resume.

Nếu chỉ cập nhật `failure_analysis/aggregate_results.py` sau một aggregate bị
lỗi, có wrapper riêng kiểm tra các stage/input còn lại và ghi provenance cho
planner mới. Nó yêu cầu đúng một aggregate pending đã hoàn tất train analysis:

```bash
.venv/bin/python -B -m scripts.resume_openhands_planner_update \
  --run-dir "$RUN_DIR" --dry-run
.venv/bin/python -B -m scripts.resume_openhands_planner_update \
  --run-dir "$RUN_DIR"
```

Wrapper giữ metadata gốc, lưu source planner mới và log failure cũ ở
`<run>/runtime/planner_resumes/`. Những source thay đổi khác cần thí nghiệm mới
hoặc recovery tương ứng, không xóa stage markers/đổi hash để bỏ kiểm tra.

Aggregate có tối đa 5 retry sau lần đầu (tổng 6 lần sinh), giữ cùng prompt/context
và không đưa lỗi của attempt trước vào feedback. Retry dừng khi plan/spec hợp
lệ. `max_candidate_retries: 2` là cơ chế khác: sinh lại candidate khi import/build
thất bại; validation không cải thiện được xử lý bằng promotion policy.

`run_test: true` tự chạy test sau selection. Với `run_test: false`, chạy test
của candidate đã chọn bằng:

```bash
.venv/bin/python -B run_pipeline_openhands.py test --run-dir "$RUN_DIR"
```

Kết quả nằm ở:

```text
<run>/experiment.json             config/model/split hashes
<run>/provenance.json             revisions và task image
<run>/runs/train_hN/results.json  train outcomes
<run>/runs/val_hN/results.json    validation outcomes
<run>/analysis/*.jsonl            diagnosis train/val regression
<run>/iterations/vN/              plan/spec, audit, comparisons, promotion
<run>/selection.json              candidate được chọn trước test
<run>/summary.json                kết quả selection và test
```

Với test 30 sample x 2 response, `resolved_rollouts / 60` là success rate theo
rollout. Số task có ít nhất một rollout đạt là thống kê khác. Báo cáo riêng
resolved/unresolved/error, timeout và context overflow; pipeline hoàn tất không
có nghĩa task agent giải tốt.

### 7.2. SWE, GAIA, AppWorld, Terminal-Bench

Chạy lại đúng command ban đầu để dùng lại các output hoàn tất. Các driver dùng
`traces/`, `eval/`, `failure_analysis/results/`, `improvement_plans/` và candidate
`task_agent/enhanced_*_vN/`; không nhận `--run-dir` như OpenHands. `--force` chủ
động tái tạo output. `--max-modify-retries` ở các driver gốc là cơ chế redo riêng,
có thể dùng feedback; khác retry sinh aggregate với prompt giữ nguyên.

Kết thúc repair loop, driver in lệnh inference và evaluation test cho candidate
được chọn. Chạy các lệnh đã in để giữ đúng version/model/split; test không tự
được chạy bởi bốn driver này. Giữ test ngoài analysis, aggregate và promotion.

## 8. Các lỗi setup thường gặp

| Hiện tượng | Kiểm tra/cách xử lý |
|---|---|
| Không kết nối Docker | `docker info`; daemon và quyền dùng socket Docker. |
| SDK Docker startup fail | SDK revision/patch mục 3, image đã build; xem health-check/container log của lần chạy đó. |
| Alias model không tồn tại | Cấu hình đúng SLM checkout, load lại `.env`, kiểm tra `LM_DICT`. |
| `api_base` không hợp lệ | Thay placeholder trong profile bằng URL HTTP(S) thực có `/v1`. |
| Gemini request đi đến Qwen endpoint | Kiểm tra connection riêng theo stage, mục 4.2/4.4; CLI model ID không thay mọi YAML legacy. |
| Model không gọi được tool | Chạy tool-call probe; kiểm tra chat template/parser của server. |
| RefactorBench không tìm thấy repo | Export `REFACTORBENCH_REPOS_DIR`, kiểm tra chín repository ở mục 5.2. |
| RefactorBench evaluator thiếu Python | `uv python install 3.11` trên host SLM. |
| WebArena chứa `__SHOPPING_ADMIN__` | Site chưa sẵn sàng khi preprocess; kiểm tra GET và dùng run/rollout mới. |
| AppWorld thiếu cache/root | Dataset phải có `data/tasks`, tạo cache ở mục 6.4 và ghi các đường dẫn tuyệt đối vào `.env`. |
| Terminal-Bench thiếu split manifest | Dùng đường setup GitHub + `--random-split`, hoặc cung cấp manifest verified của thí nghiệm cần tái lập. |
| Resume báo fingerprint/artifact khác | Kiểm tra thay đổi config/model/source; dùng wrapper planner nếu đúng điều kiện, hoặc tạo run mới. |
| Agent timeout/context overflow | Đọc task trace và model completion error; kiểm tra max_time/token budgets và cách xử lý dữ liệu của candidate. |

Các lệnh bridge và từng stage analyze/aggregate/modify/audit/gate nằm ở
[README, mục 8](../README.md#8-run-the-openhands-closed-loop). Dùng pipeline đầy
đủ cho thí nghiệm thường ngày; dùng stage riêng khi cần kiểm tra artifact.
