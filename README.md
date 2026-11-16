# studio-engines

Engine kit and engine adapters for Freeweights Studio, Apache-2.0.

Freeweights Studio trains models on your own computer. Each training engine runs
as its own process and speaks **Engine API v1**: loopback HTTP, a per-launch
bearer token, one job at a time, states
`queued -> preparing -> running -> exporting -> succeeded | failed | cancelled`,
events polled with `?after=seq`. The kit implements that API once. An adapter
only turns a job spec v1 into one upstream engine's own config file and command
line, then the kit runs that command as a separate process.

**Not yet tested by Allternit on this hardware.** That label applies to every
engine and every hardware id below. The configs are checked in dry-run against
the engines' own CLI options and against golden files; no trainer has been run
by Allternit with these adapters on any of these computers.

```
kit/studio_engine_kit/        Engine API v1 server, job state machine, spec checks, fit (hardware) checks
adapters/studio_engine_adapters/
    axolotl.py                axolotl train / merge-lora                   (YAML)
    llama_factory.py          llamafactory-cli train / export              (YAML + dataset_info.json)
    torchtune.py              tune run <recipe> --config                   (recipe config)
tests/conformance/            one suite for every adapter, dry-run, plus golden configs and the fit table
pins.json                     upstream version pinned per engine, and how it was checked
tools/spdx_gate.py            licence gate (no AGPL import, SPDX headers)
tools/sync_spec.py            refresh the vendored contract files
```

## Engines and hardware

The hardware ids come from the contract: `apple-silicon`, `cuda-linux`,
`cuda-windows`, `rocm-linux`, `xpu-linux`, `xpu-windows`, `cpu`.

| Engine | Upstream pin | Licence | Runs on | Methods | Status |
|---|---|---|---|---|---|
| axolotl | axolotl 0.21.1 | Apache-2.0 | cuda-linux | lora, qlora, full, dpo | Not yet tested by Allternit on this hardware |
| llama-factory | llamafactory 0.9.5 | Apache-2.0 | cuda-linux, cuda-windows | lora, qlora, full, dpo | Not yet tested by Allternit on this hardware |
| torchtune | torchtune 0.6.1 | BSD-3-Clause | apple-silicon, cuda-linux, rocm-linux, xpu-linux | lora, qlora (Llama 3.2 bases), full, dpo | Not yet tested by Allternit on this hardware. Unmaintained upstream: 0.6.1 is the last release |

torchtune bases: Qwen2.5 and Llama 3.2; other bases are refused. LLaMA-Factory
picks the chat template from the base model's name.

## How a host installs an adapter

1. Install this repository **by pinned commit sha**, never a branch:

   ```
   python3 -m venv ~/.allternit/studio/kit
   ~/.allternit/studio/kit/bin/pip install "studio-engines @ git+https://github.com/Allternit/studio-engines@<commit sha>"
   ```

   The kit is standard library only, so this installs no other package.

2. Install the upstream engine, unmodified, at the version in `pins.json`, in
   its own virtualenv (for example `pip install axolotl==0.21.1` into
   `~/.allternit/studio/axolotl`). Check the wheel's sha256 against `pins.json`.

3. Start the engine process with the launch env the contract defines:

   ```
   STUDIO_ENGINE_TOKEN=<per-launch token> \
   ALLTERNIT_STUDIO_HARDWARE=cuda-linux \
   ALLTERNIT_STUDIO_OUTPUT_DIR=/abs/path/for/outputs \
   STUDIO_ENGINE_BIN_AXOLOTL=~/.allternit/studio/axolotl/bin/axolotl \
   ~/.allternit/studio/kit/bin/python -m studio_engine_kit --engine axolotl --port 0
   ```

   It prints `{"ready": true, "port": N, "hardware": ..., "output_dir": ...}`
   once listening on 127.0.0.1.

- `ALLTERNIT_STUDIO_HARDWARE` is the hardware id the host detected. The engine
  does not classify the computer itself. Without it, jobs are refused with
  503 `hardware_unavailable`.
- `ALLTERNIT_STUDIO_OUTPUT_DIR` is where job folders go (spec, config,
  `command.json`, `out/`). Every output path a finished job reports is
  absolute and inside it.
- `STUDIO_ENGINE_BIN_<ENGINE>` points at the engine CLI in its virtualenv;
  otherwise it is looked up on `PATH`. GGUF export uses llama.cpp's
  `convert_hf_to_gguf.py` and `llama-quantize` from `LLAMA_CPP_DIR`.
- The base model is a local folder in `base.path`. Engines never download
  weights; the host fetches and verifies the model first.

`--dry-run` builds the config and the command, writes them, and runs a stub
process instead of the trainer. The tests use it.

## Refusals

Spec checks use the contract's codes in the contract's order
(`unsupported_spec_version`, `unknown_engine`, `unknown_method`, `invalid_base`,
`invalid_dataset`, `rented_not_available`, `publish_not_available`,
`invalid_spec`), all HTTP 422. Hardware checks follow the contract's fit table,
with the same reasons word for word:

- 422 `hardware_not_supported`: the engine does not run on this hardware id.
- 422 `method_not_supported`: the method does not run here (on `cpu` only LoRA runs).
- 422 `backend_not_supported`: `resources.backend` cannot be used here (MLX off a Mac).
- 422 `output_not_supported`: an output cannot be made here (MLX files off a Mac).
- 503 `hardware_unavailable`: the device cannot be used right now (for NVIDIA,
  `nvidia-smi` is missing or lists no GPU), or the host passed no hardware id.

`GET /v1/engine` returns the engine card with a `fit` object for the hardware id.

## The contract copy

`kit/studio_engine_kit/engines.json` and `tests/conformance/fixtures/*.json` are
copies of the platform's `services/studio-spec` files. `spec-sync.json` records
the commit and the sha256 of each. `tests/conformance/test_spec_sync.py` fails
when a copy is edited, and, with `ALLTERNIT_STUDIO_SPEC_DIR` set to a platform
`services/studio-spec` folder, when the platform has moved on. To refresh:

```
python3 tools/sync_spec.py <platform checkout>/services/studio-spec --commit <sha>
python3 -m pytest -q
```

## Licences

The kit, adapters, tests and tools are Apache-2.0 (`LICENSE`, `NOTICE`). **Each
upstream engine keeps its own licence and is installed unmodified at the pinned
version in `pins.json`**: nothing from an engine is copied, imported or patched
here. No engine may be under AGPL or LGPL, even behind the process boundary:
Unsloth was removed as an engine on 2026-11-15 because its CLI carries AGPL-3.0
headers (Allternit's own trainer, allternit-tune, covers LoRA, QLoRA and GGUF).
`tools/spdx_gate.py` still fails if an Unsloth package is imported or listed as
a dependency.

## Tests

```
python3 -m venv .venv && .venv/bin/pip install pytest pyyaml
.venv/bin/pytest                                  # conformance, dry-run, no GPU, no engines
UPDATE_GOLDEN=1 .venv/bin/pytest tests/conformance/test_golden.py   # after a deliberate change
python3 tools/spdx_gate.py                        # licence gate
```

No real training run is part of the tests. Output locations such as Axolotl's
merge folder and LLaMA-Factory's export folder are the likeliest to need adjustment
after the first run on real hardware.
