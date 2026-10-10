Thiết lập HarnessFix với Qwen và Gemini

Hướng dẫn dùng Bash trên Ubuntu/WSL2 và cấu trúc repo hiện tại. Cấu hình
model nằm trong `configs/`, công cụ setup trong `scripts/`, credential trong
`.env`. Benchmark chạy bằng các entry point sẵn có của HarnessFix.

**1. Đường dẫn và công cụ host**

```bash
export HARNESSFIX_ROOT=/root/HarnessFix-reproduce
export BETTER_ROOT=/root/slm-harness-adaptation-reproduce

git --version
uv --version
docker info
docker compose version
```

Nếu thiếu công cụ host hoặc Docker, dùng các command cài đặt trong mục 2.1
của [README](../README.md). Docker cần kết nối được daemon.

**2. Môi trường HarnessFix**

```bash
cd "$HARNESSFIX_ROOT"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt 'swebench==4.1.0'
.venv/bin/python -m swebench.harness.run_evaluation --help
```

Nếu đã cài dependency và chạy được evaluator `--help`, tiếp tục bước 3.
Giữ evaluator ở bản 4.1.0 để khớp tham số và đường dẫn log của pipeline.
Pipeline thu nhận report từ thư mục gốc repo nếu evaluator ghi ở đó thay vì
thư mục `--report_dir`, và tái sử dụng report đã có khi resume.

**3. Cấu hình model**

Điền model ID, endpoint và token budget trong [configs/qwen.yaml](../configs/qwen.yaml).
Qwen tự serve dùng `openai/<ID trả về từ /v1/models>`; prefix này chọn giao
thức API tương thích OpenAI. URL server cần truy cập được từ host lẫn Docker.

Điền model ID, endpoint và token budget trong
[configs/gemini.yaml](../configs/gemini.yaml). Với gateway `/v1`, dùng
`openai/<ID do gateway cung cấp>`; ví dụ `openai/ag/gemini-3.1-pro-low`.
Prefix chọn giao thức chat-completions, backend vẫn là Gemini. Dùng key của
gateway trong `GEMINI_API_KEY`. Với Google AI Studio trực tiếp, dùng
`gemini/<native model-id>` và `api_base: null`.

Cấu hình Qwen chạy agent, Gemini phân tích/lập kế hoạch/sửa harness:

```bash
cd "$HARNESSFIX_ROOT"
export QWEN_API_KEY='EMPTY'  # Thay nếu Qwen server yêu cầu key.
export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'

.venv/bin/python scripts/configure_models.py \
  --config configs/qwen.yaml \
  --analysis-config configs/gemini.yaml \
  --better-root "$BETTER_ROOT"

set -a
source .env
set +a
```

Command này cập nhật `.env` ở hai repo, model registry của SWE-bench,
`<BETTER_ROOT>/configs/models.yaml`, và hai cấu hình phân tích/sửa harness
SWE-bench. Prompt và các alias không liên quan được giữ lại. Key chỉ được
ghi vào `.env`. Với Qwen self-hosted, giá được đặt 0 để không chặn inference
do thiếu pricing; số USD trong báo cáo không đo chi phí server thực tế.

Để dùng một model cho mọi giai đoạn, bỏ `--analysis-config`. Có thể dùng
`--config configs/gemini.yaml` để chọn Gemini cho cả chạy agent và repair.

Để cập nhật riêng model analysis, giữ cấu hình task Qwen và credential hiện có:

```bash
.venv/bin/python scripts/configure_models.py \
  --config configs/gemini.yaml \
  --analysis-only \
  --better-root "$BETTER_ROOT"
```

Nếu analysis gặp lỗi API hoặc trả JSON không hợp lệ, record fallback vẫn được
lưu để debug nhưng được tính là thất bại. Resume sẽ thử lại các record này;
aggregate chỉ đọc diagnosis hợp lệ. Dòng tổng kết và exit code phản ánh trạng
thái analysis thay vì chỉ việc đã ghi được một record.

**4. Môi trường Better Harness/OpenHands**

```bash
cd "$BETTER_ROOT"
git submodule sync --recursive
git submodule update --init --recursive software-agent-sdk LOCA-bench
uv sync --python 3.14

test -e .vertex-ai.json || printf '%s\n' '{}' > .vertex-ai.json

uv run python -c 'from src.utils import LM_DICT; print(sorted(LM_DICT))'
```

Khi dùng hai profile trên, danh sách cần có `qwen-vllm` và `gemini-api`.
HarnessFix và Better Harness có hai virtual environment riêng.

**5. Build môi trường Stock Alert**

```bash
cd "$BETTER_ROOT"
benchmark_uid="$(id -u)"
if [ "$benchmark_uid" -eq 0 ]; then benchmark_uid=1000; fi

BENCHMARK_UID="$benchmark_uid" \
  docker compose build woocommerce_stock_alert_s2l

docker image inspect woocommerce_stock_alert_s2l:latest
```

Phần setup kết thúc ở đây. Các bước sau chạy benchmark bằng pipeline sẵn có.
OpenHands cần model server hỗ trợ native function/tool calls.

**6. Chạy SWE-bench bằng HarnessFix**

Chuẩn bị các split mặc định của repo:

```bash
cd "$HARNESSFIX_ROOT"
.venv/bin/python data/sample_swebench.py
```

Sampler tạo `data/verified_train_100`, `data/verified_val_50` và
`data/verified_test_100`. Test được giữ ngoài vòng sửa harness.

```bash
set -a
source .env
set +a

.venv/bin/python run_pipeline_swe.py \
  --model "$HARNESSFIX_TASK_MODEL" \
  --analysis-model "$HARNESSFIX_ANALYSIS_MODEL"
```

Command dùng các tham số mặc định của pipeline. Có thể điều chỉnh
`--workers`, `--max-iterations`, `--train-dir` và `--val-dir` bằng các option
của entry point. Xem `run_pipeline_swe.py --help` để chọn kích thước/lịch chạy.

**7. Chạy Stock Alert qua bridge HarnessFix/OpenHands**

Để chạy đầy đủ vòng lặp HarnessFix bằng một config, dùng:

```bash
cd /root/HarnessFix-reproduce
.venv/bin/python -B run_pipeline_openhands.py run --config configs/stock_alert.yaml --dry-run
.venv/bin/python -B run_pipeline_openhands.py run --config configs/stock_alert.yaml
```

Config chia test là 30 sample đầu, train là 10 sample tiếp, val là 10 sample tiếp
nữa; train/val chạy 1 rollout mỗi sample, test cuối chạy 2 rollout mỗi sample,
và `agent_batch_size=6`. `execution.n_responses` khai báo riêng số rollout của
`train`, `val`, `test`: mỗi candidate có 10 train và 10 val rollout; candidate
cuối có 60 test rollout. Pipeline tự chạy baseline,
phân tích train, tổng hợp kế hoạch, sửa/audit harness, chạy lại train và val,
quyết định promotion theo cùng policy với SWE, lưu memory và phân tích regression
trên val cho vòng tiếp theo. Mặc định tối đa 3 vòng; test chỉ chạy với candidate
đã được chọn sau cùng. Dùng `--run-dir <thư mục run đã in ra>` để resume cùng config.

Các command bridge và từng stage dưới đây phục vụ chạy thủ công.

Bridge tự bật SDK completion logging cho mỗi rollout mới. Request/response LLM
được lưu tại `<workspace>_logs/llm_completions/`; manifest tham chiếu các file
này qua `llm_completion_paths`, và analysis đọc chúng vào `model_calls`.
Packed runtime giữ callback logging/stats qua bước đăng ký LLM của SDK 1.19,
vốn có thể tạo lại telemetry khi khởi tạo conversation. Bản sửa được mount
cùng file agent nên chạy rollout mới sẽ dùng được với Docker image hiện tại.
Với run cũ chưa bật logging, adapter giữ nội dung/reasoning/tool calls còn trong
events và đánh dấu request là `not_recorded`. Dùng `--rollout-version` mới để
thu thập đầy đủ logs; resume run cũ không khôi phục request đã không được lưu.

Chuẩn bị train/validation/test riêng từ dataset của Better Harness. Các giá
trị bên dưới chọn 30 train, 30 validation và giữ phần còn lại làm test;
có thể thay hai biến số lượng trước khi chạy.

```bash
cd "$HARNESSFIX_ROOT"
export STOCK_TRAIN_N=30
export STOCK_VAL_N=30

.venv/bin/python - <<'PY'
import json
import os
import random
from pathlib import Path
import yaml
from dotenv import dotenv_values

settings = dotenv_values('.env')
better = Path(settings['BETTER_HARNESS_ROOT'])
task = 'woocommerce_stock_alert_s2l'
data = json.loads((better / 'data' / f'{task}.json').read_text())
train_n = int(os.environ['STOCK_TRAIN_N'])
val_n = int(os.environ['STOCK_VAL_N'])
assert train_n > 0 and val_n > 0 and train_n + val_n < len(data)
assert len({row['id'] for row in data}) == len(data)
random.Random(42).shuffle(data)
splits = {
    'train': data[:train_n],
    'val': data[train_n:train_n + val_n],
    'test': data[train_n + val_n:],
}
data_dir = better / 'data' / 'harnessfix' / task
data_dir.mkdir(parents=True, exist_ok=True)
task_dir = better / 'tasks' / task
base = yaml.safe_load((task_dir / 'run.yaml').read_text())
base.pop('agent_file', None)
base['model_name'] = settings['HARNESSFIX_MODEL_NAME']
for split, records in splits.items():
    path = data_dir / f'{split}.json'
    path.write_text(json.dumps(records, indent=2) + '\n')
    config = base | {'data_path': str(path), 'max_examples': len(records),
                     'n_responses': 2 if split == 'test' else 1}
    (task_dir / f'run_harnessfix_{split}.yaml').write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    print(f'{split}: {len(records)} records')
PY
```

Các giới hạn runtime được giữ theo task YAML gốc; train/val dùng 1 rollout mỗi
sample và test dùng 2. Chạy H0:

```bash
set -a
source .env
set +a
export BETTER_ROOT="$BETTER_HARNESS_ROOT"
export TASK=woocommerce_stock_alert_s2l
export ANALYSIS_MODEL="$HARNESSFIX_ANALYSIS_MODEL_NAME"

.venv/bin/python -B -m task_agent.openhands_agent.bridge materialize \
  --better-root "$BETTER_ROOT" \
  --task-id "$TASK" --prompt-name default \
  --output-dir "artifacts/$TASK/h0"

.venv/bin/python -B -m task_agent.openhands_agent.bridge run \
  --better-root "$BETTER_ROOT" \
  --base-config "tasks/$TASK/run_harnessfix_train.yaml" \
  --candidate-dir "artifacts/$TASK/h0" \
  --rollout-version harnessfix_h0_train \
  --normalized-output "artifacts/$TASK/train_h0"
```

Nếu H0 có failed rollouts, tiếp tục các bước analyze → aggregate → modify →
audit → paired validation/gate ở mục 8.3–8.6 của [README](../README.md).
Để dùng các lệnh `python` trong README, kích hoạt môi trường HarnessFix:

```bash
source .venv/bin/activate
```

Các biến `TASK`, `BETTER_ROOT` và `ANALYSIS_MODEL` trên trỏ tới Stock Alert và
Gemini đã cấu hình. Nếu H0 đã giải đúng toàn bộ train, không có failure để sửa.
Khi chạy lại với candidate mới, chọn thư mục candidate và rollout version mới.
