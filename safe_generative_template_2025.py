import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import importlib.util
import inspect
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # Fallback if not installed

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore

def set_seeds(base_seed: int, rank_offset: int = 0) -> None:
    seed = (base_seed + rank_offset) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="placeholder/generative")  # display only
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--total-samples", type=int, default=100_000)
    parser.add_argument("--output-dir", type=str, default="./generations_2025")
    parser.add_argument("--wandb-project", type=str, default="safe-generative-run")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--hf-model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="Generic structured content:")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--enable-jsonl-shards", action="store_true")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--no-per-sample-json", action="store_false", dest="save_per_sample_json")
    parser.set_defaults(save_per_sample_json=True)
    parser.add_argument("--custom-generator", type=str, default=None, help="Path to a Python file exposing a generator class")
    parser.add_argument("--generator-class", type=str, default="CustomGenerator", help="Class name to load from custom generator file")
    return parser.parse_args()


def init_distributed_if_needed(local_rank_arg: int) -> None:
    world_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")
    need_init = world_env is not None or local_rank_env is not None
    if not need_init:
        return

    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    # Determine device index
    if local_rank_arg is not None and local_rank_arg >= 0:
        local_rank = local_rank_arg
    else:
        local_rank = int(local_rank_env) if local_rank_env is not None else 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def reduce_sum(value: int) -> int:
    if not is_distributed():
        return value
    t = torch.tensor([value], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


class TextGenerator(nn.Module):
    """
    Safe, domain-agnostic text generator using HuggingFace transformers.
    Produces generic text sequences for demonstration and infrastructure testing.
    """

    def __init__(self, model_name: str, device: torch.device) -> None:
        super().__init__()
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for TextGenerator. Install with: pip install transformers"
            )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        need_resize = False
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                need_resize = True
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(self.device)
        if need_resize:
            self.model.resize_token_embeddings(len(self.tokenizer))
        # Ensure generate uses a valid padding id
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def forward(self, batch_size: int, prompt: str, max_new_tokens: int, temperature: float) -> List[str]:
        prompts = [prompt] * batch_size
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        generations = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        texts = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
        return texts


class JsonlShardWriter:
    """
    Rank-safe JSONL shard writer. Each rank writes to its own rolling shard to avoid contention.
    Writes to a temporary file and atomically renames on rotation for durability.
    """

    def __init__(self, base_dir: Path, rank: int, shard_size: int, start_index: int = 0) -> None:
        self.base_dir = base_dir
        self.rank = rank
        self.shard_size = max(1, shard_size)
        self.current_index = max(0, start_index)
        self.current_count = 0
        self._fh = None  # type: ignore
        self._open_new_shard()

    def _shard_paths(self, index: int) -> (Path, Path):
        final_path = self.base_dir / f"shard_rank{self.rank}_{index:05d}.jsonl"
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        return final_path, tmp_path

    def _open_new_shard(self) -> None:
        if self._fh is not None:
            self._fh.close()
        final_path, tmp_path = self._shard_paths(self.current_index)
        self._fh = open(tmp_path, "a", encoding="utf-8")
        self.current_count = 0

    def write(self, record: dict) -> None:
        assert self._fh is not None
        line = json.dumps(record, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        self.current_count += 1
        if self.current_count >= self.shard_size:
            self._rotate()

    def _rotate(self) -> None:
        assert self._fh is not None
        self._fh.close()
        final_path, tmp_path = self._shard_paths(self.current_index)
        os.replace(tmp_path, final_path)
        self.current_index += 1
        self._open_new_shard()

    def finalize(self) -> None:
        if self._fh is None:
            return
        # Close and finalize the current tmp shard even if not full
        self._fh.close()
        final_path, tmp_path = self._shard_paths(self.current_index)
        if os.path.exists(tmp_path):
            os.replace(tmp_path, final_path)
        self._fh = None


def next_shard_start_index(shards_dir: Path, rank: int) -> int:
    """
    Determine the next shard index for a given rank by scanning existing finalized shard files.
    """
    max_idx = -1
    prefix = f"shard_rank{rank}_"
    for p in shards_dir.glob(f"{prefix}*.jsonl"):
        try:
            stem = p.stem  # shard_rank{rank}_00000
            idx_str = stem.split("_")[-1]
            max_idx = max(max_idx, int(idx_str))
        except Exception:
            continue
    return max_idx + 1


def load_custom_generator(path: str, class_name: str, device: torch.device) -> nn.Module:
    """
    Dynamically load a custom generator class from a file.
    The class should be a torch.nn.Module with a forward method that returns List[str].
    The forward signature may accept (batch_size) or (batch_size, prompt, max_new_tokens, temperature).
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Custom generator file not found: {file_path}")
    spec = importlib.util.spec_from_file_location("custom_generator_mod", str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_generator_mod"] = module
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {file_path.name}")
    cls = getattr(module, class_name)
    if not inspect.isclass(cls):
        raise TypeError(f"'{class_name}' in {file_path.name} is not a class")
    instance = cls()
    if isinstance(instance, nn.Module):
        return instance.to(device)
    # If not nn.Module, wrap in a thin Module
    class _Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        @torch.no_grad()
        def forward(self, *args, **kwargs):
            return self.inner(*args, **kwargs)
    return _Wrapper(instance).to(device)


def call_generator(model: nn.Module, batch_size: int, prompt: str, max_new_tokens: int, temperature: float) -> List[str]:
    """
    Call a generator model with adaptive signature handling.
    Tries (batch_size, prompt, max_new_tokens, temperature) then falls back to (batch_size).
    """
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())
    try:
        if len(params) >= 4:
            return model(batch_size, prompt, max_new_tokens, temperature)  # type: ignore
        return model(batch_size)  # type: ignore
    except TypeError:
        # Try progressively fewer arguments
        try:
            return model(batch_size, prompt)  # type: ignore
        except TypeError:
            return model(batch_size)  # type: ignore


def list_existing_ids(output_dir: Path) -> List[int]:
    ids: List[int] = []
    for p in output_dir.glob("sample_*.json"):
        name = p.stem  # sample_00000001
        try:
            ids.append(int(name.split("_")[1]))
        except Exception:
            continue
    ids.sort()
    return ids


def save_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()

    init_distributed_if_needed(args.local_rank)
    rank = get_rank()
    world_size = get_world_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed, rank_offset=rank)

    # W&B setup (rank 0 only), with graceful fallback
    wandb_run: Optional[object] = None
    if rank == 0:
        if wandb is not None:
            try:
                run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
                wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            except Exception:
                wandb_run = None

    # Model selection (safe, domain-agnostic)
    if args.custom_generator:
        model = load_custom_generator(args.custom_generator, args.generator_class, device)
    else:
        model = TextGenerator(args.hf_model, device).to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shard_writer = None
    if args.enable_jsonl_shards:
        shards_dir.mkdir(parents=True, exist_ok=True)
        start_shard_idx = next_shard_start_index(shards_dir, rank)
        shard_writer = JsonlShardWriter(shards_dir, rank=rank, shard_size=args.shard_size, start_index=start_shard_idx)

    # Discover existing outputs to resume
    if rank == 0:
        existing_ids = list_existing_ids(output_dir) if args.resume else []
        start_idx = (existing_ids[-1] + 1) if len(existing_ids) > 0 else 0
    else:
        start_idx = 0

    if is_distributed():
        t = torch.tensor([start_idx], device=device, dtype=torch.long)
        dist.broadcast(t, src=0)
        start_idx = int(t.item())

    # Progress only on rank 0
    pbar = None
    if rank == 0:
        pbar = tqdm(total=args.total_samples, initial=start_idx)

    step_width = args.batch_size * world_size

    use_autocast = args.mixed_precision in {"fp16", "bf16"} and device.type == "cuda"
    autocast_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    # Main loop
    for global_base in range(start_idx, args.total_samples, step_width):
        local_base = global_base + rank * args.batch_size
        if local_base >= args.total_samples:
            local_effective_bs = 0
        else:
            local_effective_bs = min(args.batch_size, args.total_samples - local_base)

        if local_effective_bs > 0:
            with torch.no_grad():
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        outputs = call_generator(model, local_effective_bs, args.prompt, args.max_new_tokens, args.temperature)
                else:
                    outputs = call_generator(model, local_effective_bs, args.prompt, args.max_new_tokens, args.temperature)
        else:
            outputs = []

        # Save each item if not already present
        local_new = 0
        for i, item in enumerate(outputs):
            idx = local_base + i
            if idx >= args.total_samples:
                continue
            wrote_any = False
            if args.save_per_sample_json:
                save_path = output_dir / f"sample_{idx:08d}.json"
                if not save_path.exists():
                    save_json_atomic(save_path, {"id": idx, "content": item, "global_base": global_base})
                    wrote_any = True
            if shard_writer is not None:
                shard_writer.write({"id": idx, "content": item, "global_base": global_base})
                wrote_any = True
            if wrote_any:
                local_new += 1

        total_new = reduce_sum(local_new)

        if rank == 0:
            if pbar is not None and total_new > 0:
                pbar.update(total_new)
            if wandb_run is not None:
                try:
                    wandb.log({"samples_generated": pbar.n if pbar is not None else 0, "new_this_step": total_new})
                except Exception:
                    pass

            # Periodic checkpoint
            if pbar is not None and pbar.n > 0 and pbar.n % 5000 == 0:
                save_json_atomic(output_dir / "checkpoint.json", {"next_id": pbar.n, "time": datetime.now().isoformat()})

        # End of step
        barrier()

    if rank == 0:
        print(f"Finished! {args.total_samples:,} samples in {output_dir}")
        if wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
    if shard_writer is not None:
        shard_writer.finalize()


if __name__ == "__main__":
    main()


