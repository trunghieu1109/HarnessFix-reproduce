# HarnessFix adapter for Better Harness/OpenHands

This prototype keeps Better Harness responsible for task setup, execution, and
official evaluation. HarnessFix receives only normalized references to the
existing OpenHands traces and evaluator results.

## Candidate boundary

Each H0/H1 candidate is a versioned directory with this public contract:

- `agent.py`: required `build_agent(base_dir, llm)` and optional
  `get_workspace_scripts()` / `get_hook_config(workspace_dir)`.
- `config.json`: task identity and candidate-owned runtime configuration.
- `prompts/`, `skills/`, `context/`, `tools/`, `parsers/`, `hooks/`,
  `verification/`, `subagents/`, `monitoring/`, `workspace_scripts/`: editable
  component families selected through HarnessFix's operator registry.

Better Harness accepts one `agent_file` in Docker mode. The bridge therefore
packs the complete candidate directory into a deterministic, self-extracting
Python launcher. It does not patch Better Harness or OpenHands.

## Candidate initialization retries

The closed-loop runner audits each proposed candidate, then imports the packed launcher and
calls `build_agent` in a temporary workspace using Better Harness's Python and SDK. This check
does not start a conversation, create browser/MCP sessions, or evaluate a benchmark sample.
SDK model requests are blocked during the check.
The check preserves the task model's name and generation settings, with placeholder connection
credentials; it does not make a request to the served model.

An import/build failure retries generation of the same version with the original context:
the same accepted base, plan/spec, prompts, references, model settings, and target directory.
Each retry starts a fresh modifier session. No previous candidate error, traceback, diff, or
conversation is added to its input; the pipeline does not pass `--redo-feedback` for these retries.
`pipeline.max_candidate_retries: 2` permits one initial proposal and two retries;
`pipeline.candidate_check_timeout: 60` bounds each initialization check in seconds.
Train/validation scores do not trigger this retry. Audit rejections retain the outer-loop policy.

Each attempt is retained under `iterations/vN/attempts/<attempt>/`, including its candidate,
modifier trajectory, audit, and `candidate_check.json`. The target directory is moved into the
attempt archive after generation so the next modifier starts with a fresh copy at the same path.
After retries are exhausted, the iteration report records `candidate_initialization_failed`
and the candidate is rejected without running train/val. Its failure is available to the next
outer iteration through the previous report and repair memory. Infrastructure failures still
stop the run. Resuming reuses completed attempts and checks.

This changes experiment provenance. Use a new run directory with the updated code; old experiment
snapshots and candidates are not rewritten.

## Model aliases

Both Better Harness and the HarnessFix OpenHands stages read
`<better-root>/configs/models.yaml`. Use `gemini-3.1-pro-low` or `qwen-vllm`
as `model_name` in the Better run YAML and as `--model` in HarnessFix analysis,
aggregation, and modification commands. Pass `--better-root` to those commands.
Put the referenced API credentials in Better Harness's `.env` or environment.

## 1. Materialize H0

Run from the HarnessFix repository root. Use `shopping_admin` as the WebArena
prompt name; the other three task configs currently use `default`.

```powershell
python -B -m task_agent.openhands_agent.bridge materialize `
  --better-root D:\path\to\slm-harness-adaptation `
  --task-id refactorbench `
  --prompt-name default `
  --output-dir artifacts\refactorbench\h0
```

Supported task IDs are `woocommerce_stock_alert_s2l`,
`machine_operating_s2l`, `refactorbench`, and `webarena`.

## 2. Execute and normalize a split

The bridge preserves the selected Better Harness YAML values, including model,
data path, prompt, number of responses, and runtime settings. It overrides only
`agent_file` and `rollout_version`, then invokes the existing `src.collect` and
`src.evaluate` commands.

```powershell
python -B -m task_agent.openhands_agent.bridge run `
  --better-root D:\path\to\slm-harness-adaptation `
  --base-config tasks\refactorbench\run.yaml `
  --candidate-dir artifacts\refactorbench\h0 `
  --rollout-version harnessfix_h0_train `
  --normalized-output artifacts\refactorbench\train_h0
```

The normalized directory contains `results.json` plus one
`traces/<task>__example<N>__rollout<M>/manifest.json` per rollout. The manifest
references the original filtered/raw trace and embeds the official evaluator
result. Multiple rollouts also share a stable `<task>__example<N>`
`task_instance_id` so the aggregate planner can combine their evidence.

## 3. Analyze all failed train rollouts

```powershell
python -B run_pipeline_openhands.py analyze `
  --better-root D:\path\to\slm-harness-adaptation `
  --traces-dir artifacts\refactorbench\train_h0\traces `
  --eval-results artifacts\refactorbench\train_h0\results.json `
  --agent-source-dir artifacts\refactorbench\h0 `
  --output-file artifacts\refactorbench\train_h0_analysis.jsonl `
  --model gemini-3.1-pro-low
```

HTIR pairs actions and observations by `tool_call_id` and adds the official
score/feedback as the outcome anchor. It intentionally does not synthesize a
task-state delta from a successful tool observation.

## 4. Plan, modify, and audit

```powershell
python -B run_pipeline_openhands.py aggregate `
  --better-root D:\path\to\slm-harness-adaptation `
  --results-file artifacts\refactorbench\train_h0_analysis.jsonl `
  --output artifacts\refactorbench\plan.md `
  --spec-output artifacts\refactorbench\plan.json `
  --model gemini-3.1-pro-low

python -B run_pipeline_openhands.py modify `
  --better-root D:\path\to\slm-harness-adaptation `
  --base-dir artifacts\refactorbench\h0 `
  --target-dir artifacts\refactorbench\h1 `
  --plan artifacts\refactorbench\plan.md `
  --spec artifacts\refactorbench\plan.json `
  --model gemini-3.1-pro-low

python -B run_pipeline_openhands.py audit `
  --base-dir artifacts\refactorbench\h0 `
  --candidate-dir artifacts\refactorbench\h1 `
  --spec artifacts\refactorbench\plan.json `
  --output artifacts\refactorbench\h1_audit.json
```

On a later repair round, pass the existing HarnessFix feedback inputs with
`--val-analyses`, `--prev-plan`, and/or `--prev-iteration-report`. Validation
failures used this way are development feedback; keep the final test split
outside the loop.

The audit is the existing HarnessFix plan-to-diff/syntax check extended with
the OpenHands mode. Official task correctness remains the evaluator's job.

## 5. Paired validation gate

Run H0 and H1 with the same validation YAML, task IDs, model settings, and
`n_responses`. Because normalized rollout IDs omit the candidate version, H0
and H1 align on `<task, example, rollout>`. Put those rollout IDs in the
validation IDs file, one per line, then run:

```powershell
python -B run_pipeline_openhands.py gate `
  --baseline-traces artifacts\refactorbench\val_h0\traces `
  --baseline-eval artifacts\refactorbench\val_h0\results.json `
  --current-traces artifacts\refactorbench\val_h1\traces `
  --current-eval artifacts\refactorbench\val_h1\results.json `
  --ids-file artifacts\refactorbench\val_ids.txt `
  --plan-spec artifacts\refactorbench\plan.json `
  --output artifacts\refactorbench\val_gate.json
```

Keep the final test split outside analysis, repair, and validation retries.
