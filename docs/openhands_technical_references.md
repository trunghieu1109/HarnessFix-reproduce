# OpenHands technical reference inputs

HarnessFix's OpenHands analysis, aggregate, and modify stages receive two technical inputs
from `better_root/docs/`:

- `sdk_reference.md`: the SDK overview, embedded in the repair-agent system prompt.
- `read_trajectory.md`: raw SDK event schemas, embedded with a HarnessFix compatibility note.
- `sdk_reference_details/*.md`: SDK signatures/examples available on demand, alongside the overview.

`adaptation.md` and the SLM adaptation-selection policy are excluded. These inputs document
interfaces and evidence formats; HarnessFix's HTIR diagnosis, typed operator registry,
implementation spec, and promotion/stopping policy still govern the repair process.
The references are not installed as task-agent skills and do not change H0 or the test schedule.

The experiment configuration defaults to:

```yaml
repair_references:
  sdk_reference: true
  read_trajectory: true
```

Each input can be disabled independently. No adaptation-guide option is supported.
Only the named Markdown files and SDK detail documents are copied; benchmark overviews,
datasets, teacher trajectories, ground truth, and private evaluator files are excluded.

Snapshots are stored at `<run-dir>/runtime/repair_references/`. File SHA256 hashes are recorded
in `experiment.json` and `runtime/repair_references/manifest.json`, and repair stages verify
them before use. Changing the input configuration or a source document requires a new run
directory. References must remain read-only; SDK examples must be checked against the installed SDK.

Individual `analyze`, `aggregate`, and `modify` commands snapshot the documents next to their
output. Use `--reference-dir` to share a verified snapshot, or `--no-technical-references` to
disable both inputs. The two options cannot be combined.

The raw SDK trace is an event list, while HarnessFix's sanitized trace is an object with
`events` and `model_calls`. HTIR and manifests are separate objects. The compatibility note
directs the agent to the actual supplied paths and HarnessFix trace helpers, rather than the
SLM reference's example workspace paths. `FinishAction` indicates termination; supplied
evaluation feedback determines task success.
