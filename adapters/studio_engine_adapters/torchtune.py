# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Adapter for torchtune: writes a recipe config, runs `tune run <recipe>`.

torchtune picks the model through a component name per family and size, so this
adapter carries a small table (Qwen2.5 and Llama 3.2). Other bases are refused
with a plain reason instead of guessed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from studio_engine_kit import yamlish
from studio_engine_kit.hardware import TORCH_DEVICE
from studio_engine_kit.adapter import Adapter, Build, OutputSpec, Step, gguf_outputs, gguf_steps, require_llama_cpp
from studio_engine_kit.spec import Settings, SpecError

from ._common import whole_epochs

CONFIG = "config/torchtune.yaml"
OUT = "out/train"
QWEN_SUFFIXED = {"1_5b", "7b", "14b", "32b", "72b"}      # sizes with _base / _instruct builders
QWEN_SIZES = {"0_5b", "1_5b", "3b", "7b", "14b", "32b", "72b"}
LLAMA_SIZES = {"1b", "3b"}
ATTN = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "o_proj": "output_proj",
        "output_proj": "output_proj"}
MLP = {"gate_proj", "up_proj", "down_proj"}


def family(base: str) -> Tuple[str, str, bool]:
    """(family, size key, is_instruct) from the model name, or refuse."""
    low = Path(base).name.lower() if base.startswith(("/", ".", "~")) else base.lower().split("/")[-1]
    m = re.search(r"qwen2\.?5[-_]?(\d+(?:\.\d+)?)b", low)
    if m:
        size = m.group(1).replace(".", "_") + "b"
        if size in QWEN_SIZES:
            return "qwen2_5", size, "instruct" in low
    m = re.search(r"llama[-_]?3\.2[-_]?(\d+)b", low)
    if m and m.group(1) + "b" in LLAMA_SIZES:
        return "llama3_2", m.group(1) + "b", "instruct" in low
    raise SpecError("invalid_spec", f"torchtune adapter supports Qwen2.5 (0.5B-72B) and Llama 3.2 (1B, 3B) bases; "
                                    f"cannot map '{base}' to a torchtune model")


def components(fam: str, size: str, instruct: bool, method: str) -> Tuple[str, str]:
    """(model component, tokenizer component) for the method."""
    if fam == "qwen2_5":
        name = f"qwen2_5_{size}" + (("_instruct" if instruct else "_base") if size in QWEN_SUFFIXED else "")
        if method == "qlora":
            raise SpecError("method_not_supported", "torchtune ships no QLoRA builder for Qwen2.5; use lora or a Llama 3.2 base")
        prefix = "lora_" if method in ("lora", "dpo") else ""
        return f"torchtune.models.qwen2_5.{prefix}{name}", "torchtune.models.qwen2_5.qwen2_5_tokenizer"
    prefix = {"lora": "lora_", "dpo": "lora_", "qlora": "qlora_", "full": ""}[method]
    return f"torchtune.models.llama3_2.{prefix}llama3_2_{size}", "torchtune.models.llama3.llama3_tokenizer"


class Torchtune(Adapter):
    id = "torchtune"
    pin = "0.6.1"
    executable = "tune"

    def check_spec(self, s: Settings) -> None:
        fam, size, instruct = family(s.base)
        components(fam, size, instruct, s.method)
        whole_epochs(s, "torchtune")
        if s.dataset_revision:
            raise SpecError("invalid_dataset", "dataset.revision is not supported by torchtune")

    def lora_kwargs(self, s: Settings) -> Dict:
        if s.target_modules == "all-linear":
            return {"lora_attn_modules": ["q_proj", "k_proj", "v_proj", "output_proj"], "apply_lora_to_mlp": True}
        bad = [m for m in s.target_modules if m not in ATTN and m not in MLP]
        if bad:
            raise SpecError("invalid_spec", f"torchtune cannot LoRA these modules: {', '.join(bad)}")
        attn = sorted({ATTN[m] for m in s.target_modules if m in ATTN}, key=list(ATTN.values()).index)
        return {"lora_attn_modules": attn, "apply_lora_to_mlp": any(m in MLP for m in s.target_modules)}

    def dataset(self, s: Settings) -> Dict:
        src: Dict = {"source": "json", "data_files": s.dataset_path} if s.dataset_path else {"source": s.dataset_repo}
        if s.dataset_path and s.dataset_path.endswith(".parquet"):
            src["source"] = "parquet"
        fmt = s.dataset_format
        if s.method == "dpo":
            return dict([("_component_", "torchtune.datasets.preference_dataset")] + list(src.items()) +
                        [("column_map", {"chosen": "chosen", "rejected": "rejected"})])
        if fmt == "chat":
            comp = [("_component_", "torchtune.datasets.chat_dataset")] + list(src.items()) + [
                ("conversation_column", "messages"), ("conversation_style", "openai")]
        elif fmt in ("text", "completion"):
            comp = [("_component_", "torchtune.datasets.text_completion_dataset")] + list(src.items()) + [
                ("column", "text")]
        else:  # alpaca, jsonl
            comp = [("_component_", "torchtune.datasets.instruct_dataset")] + list(src.items()) + [
                ("column_map", {"input": "instruction", "output": "output"})]
        return dict(comp)

    def checkpoint_files(self, s: Settings) -> List[str]:
        folder = Path(s.base)
        if folder.is_dir():
            files = sorted(p.name for p in folder.glob("*.safetensors"))
            if files:
                return files
        return ["model.safetensors"]

    def config(self, s: Settings) -> Dict:
        fam, size, instruct = family(s.base)
        model, tok = components(fam, size, instruct, s.method)
        epochs = whole_epochs(s, "torchtune")
        m: Dict = {"_component_": model}
        if s.method != "full":
            m.update(self.lora_kwargs(s))
            m.update({"lora_rank": s.rank, "lora_alpha": s.alpha, "lora_dropout": s.dropout})
        if fam == "qwen2_5":
            tokenizer = {"_component_": tok, "path": f"{s.base}/vocab.json", "merges_file": f"{s.base}/merges.txt",
                         "max_seq_len": s.max_seq_len}
            model_type = "QWEN2"
        else:
            tokenizer = {"_component_": tok, "path": f"{s.base}/original/tokenizer.model",
                         "max_seq_len": s.max_seq_len}
            model_type = "LLAMA3_2"
        cfg: Dict = {
            "output_dir": OUT, "model": m, "tokenizer": tokenizer,
            "checkpointer": {"_component_": "torchtune.training.FullModelHFCheckpointer",
                             "checkpoint_dir": s.base, "checkpoint_files": self.checkpoint_files(s),
                             "recipe_checkpoint": None, "output_dir": OUT, "model_type": model_type},
            "resume_from_checkpoint": False,
        }
        if s.method != "full":
            cfg["save_adapter_weights_only"] = False
        cfg["dataset"] = self.dataset(s)
        cfg["seed"] = s.seed
        cfg["shuffle"] = True
        cfg["batch_size"] = s.batch_size
        cfg["epochs"] = epochs
        if s.max_steps > 0:
            cfg["max_steps_per_epoch"] = s.max_steps
        cfg["gradient_accumulation_steps"] = s.grad_accum
        cfg["optimizer"] = {"_component_": "torch.optim.AdamW", "lr": s.lr, "weight_decay": s.weight_decay,
                            "fused": True}
        cfg["lr_scheduler"] = {"_component_": "torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup",
                               "num_warmup_steps": s.warmup_steps}
        if s.method == "dpo":
            cfg["loss"] = {"_component_": "torchtune.rlhf.loss.DPOLoss", "beta": s.beta, "label_smoothing": 0.0}
        else:
            cfg["loss"] = {"_component_": "torchtune.modules.loss.CEWithChunkedOutputLoss"}
        cfg["device"] = TORCH_DEVICE.get(s.backend or "torch-cuda", "cuda")
        cfg["dtype"] = "bf16"
        cfg["enable_activation_checkpointing"] = s.gradient_checkpointing
        cfg["compile"] = False
        cfg["metric_logger"] = {"_component_": "torchtune.training.metric_logging.DiskLogger",
                                "log_dir": f"{OUT}/logs"}
        cfg["log_every_n_steps"] = 1
        return cfg

    def build(self, s: Settings, job_dir: Path) -> Build:
        recipe = {"lora": "lora_finetune_single_device", "qlora": "lora_finetune_single_device",
                  "full": "full_finetune_single_device", "dpo": "lora_dpo_single_device"}[s.method]
        exe = self.engine_command()
        steps: List[Step] = [Step("train", exe + ["run", recipe, "--config", CONFIG])]
        last = f"{OUT}/epoch_{whole_epochs(s, 'torchtune') - 1}"   # torchtune writes merged HF weights per epoch
        outputs: List[OutputSpec] = []
        notes = []
        if s.lr_scheduler != "cosine":
            notes.append(f"params.lr_scheduler '{s.lr_scheduler}' becomes cosine: the only schedule the torchtune adapter wires")
        if s.method == "full":
            outputs.append(OutputSpec("merged", last))
        else:
            if s.want_adapter:
                outputs.append(OutputSpec("adapter", last + "/adapter_model.safetensors"))
            if s.want_merged:
                outputs.append(OutputSpec("merged", last))
        if s.gguf:
            steps += gguf_steps(last, s.gguf, "out/gguf", require_llama_cpp(s))
            outputs += gguf_outputs(s.gguf, "out/gguf")
        header = "# SPDX-License-Identifier: Apache-2.0\n# Generated by studio-engines (torchtune adapter)"
        return Build({CONFIG: yamlish.dump(self.config(s), header)}, steps, outputs, notes)


ADAPTER = Torchtune()
