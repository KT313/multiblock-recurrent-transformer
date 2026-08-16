# Modified from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f.
# Changes (c) 2025-2026 Tobias Kerner: special-token handling and HF tokenizer robustness. See README and git history.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from pathlib import Path
from typing import Optional, Union

import torch


class Tokenizer:
    def __init__(
        self,
        checkpoint_dir: Union[Path, str],
        broken_state_load_fallback=True,
        max_length=-1,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists() and broken_state_load_fallback:
            print("Tokenizer init skipped")
            return

        if not checkpoint_dir.exists():
            try:  # Try downloading the missing tokeinizer
                from huggingface_hub import hf_hub_download

                tokenizer_file = hf_hub_download(repo_id=str(checkpoint_dir), filename="tokenizer.json")
                hf_hub_download(repo_id=str(checkpoint_dir), filename="special_tokens_map.json")
                hf_hub_download(repo_id=str(checkpoint_dir), filename="tokenizer_config.json")
                checkpoint_dir = Path(tokenizer_file).parent
            except (OSError, AssertionError):
                raise NotADirectoryError(f"Tokenizer checkpoint {str(checkpoint_dir)} cannot be loaded.")

        self.checkpoint_dir = checkpoint_dir
        # Use config overrides if provided, otherwise detect from tokenizer
        self.bos_id = bos_token_id
        self.eos_id = eos_token_id
        self.pad_id = pad_token_id
        self._warned_no_bos = False  # Track if we've warned about missing BOS token
        self._warned_no_eos = False  # Track if we've warned about missing EOS token
        self.max_length = max_length

        self.cache_token_id = None
        self.eod_token_id = None
        if (checkpoint_dir / "tokenizer.json").is_file():
            from transformers import AutoTokenizer, PreTrainedTokenizerFast

            try:  # (checkpoint_dir / "config.json").is_file():
                self.processor = AutoTokenizer.from_pretrained(
                    str(checkpoint_dir), add_bos_token=False, add_eos_token=False
                )
            except Exception:
                self.processor = PreTrainedTokenizerFast(
                    tokenizer_file=str((checkpoint_dir / "tokenizer.json")), add_bos_token=False, add_eos_token=False
                )
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                # Only auto-detect if not provided as config override
                if self.bos_id is None:
                    self.bos_id = self.processor.bos_token_id
                if self.eos_id is None:
                    self.eos_id = self.processor.eos_token_id
                if self.pad_id is None:
                    self.pad_id = self.processor.pad_token_id

                # Fallback: try to convert token strings to IDs if IDs are still None
                if self.eos_id is None and hasattr(self.processor, 'eos_token') and self.processor.eos_token:
                    try:
                        self.eos_id = self.processor.convert_tokens_to_ids(self.processor.eos_token)
                    except Exception:
                        pass
                if self.bos_id is None and hasattr(self.processor, 'bos_token') and self.processor.bos_token:
                    try:
                        self.bos_id = self.processor.convert_tokens_to_ids(self.processor.bos_token)
                    except Exception:
                        pass
                if self.pad_id is None and hasattr(self.processor, 'pad_token') and self.processor.pad_token:
                    try:
                        self.pad_id = self.processor.convert_tokens_to_ids(self.processor.pad_token)
                    except Exception:
                        pass
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
                if self.pad_id is None:
                    self.pad_id = config.get("pad_token_id")  # idk if this will always work
        elif "open_llama" in str(checkpoint_dir):
            from transformers import LlamaTokenizer

            self.processor = LlamaTokenizer.from_pretrained(
                str(checkpoint_dir), add_bos_token=False, add_eos_token=False
            )

            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                # Only auto-detect if not provided as config override
                if self.bos_id is None:
                    self.bos_id = self.processor.bos_token_id
                if self.eos_id is None:
                    self.eos_id = self.processor.eos_token_id
                if self.pad_id is None:
                    self.pad_id = self.processor.pad_token_id

                # Fallback: try to convert token strings to IDs if IDs are still None
                if self.eos_id is None and hasattr(self.processor, 'eos_token') and self.processor.eos_token:
                    try:
                        self.eos_id = self.processor.convert_tokens_to_ids(self.processor.eos_token)
                    except Exception:
                        pass
                if self.bos_id is None and hasattr(self.processor, 'bos_token') and self.processor.bos_token:
                    try:
                        self.bos_id = self.processor.convert_tokens_to_ids(self.processor.bos_token)
                    except Exception:
                        pass
                if self.pad_id is None and hasattr(self.processor, 'pad_token') and self.processor.pad_token:
                    try:
                        self.pad_id = self.processor.convert_tokens_to_ids(self.processor.pad_token)
                    except Exception:
                        pass
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
                if self.pad_id is None:
                    self.pad_id = config.get("pad_token_id")  # idk if this will always work
        else:
            raise NotImplementedError("Couldn't load a tokenizer from the given checkpoint directory.")

        # Print special token IDs for debugging
        bos_source = "config" if bos_token_id is not None else "auto-detected"
        eos_source = "config" if eos_token_id is not None else "auto-detected"
        pad_source = "config" if pad_token_id is not None else "auto-detected"
        print(f"Tokenizer initialized with special tokens:")
        print(f"  BOS: {self.bos_id} ({bos_source})")
        print(f"  EOS: {self.eos_id} ({eos_source})")
        print(f"  PAD: {self.pad_id} ({pad_source})")

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size

    def __len__(self) -> int:
        # https://stackoverflow.com/questions/67412925/what-is-the-difference-between-lentokenizer-and-tokenizer-vocab-size#:~:text=Size%20of%20the%20base%20vocabulary%20(without%20the%20added%20tokens).&text=So%20you%20can%20clearly%20see,plus%20the%20len(added_tokens_encoder)%20.
        return len(self.processor)

    def state_dict(self):
        return {"dir": self.checkpoint_dir}

    def load_state_dict(self, state_dict):
        try:
            return Tokenizer(state_dict["dir"])
        except Exception as e:
            print(f"Warning. Could not reload exact tokenizer due to error {e}. Defaulting to tokenizer from config.")
            return self

    def __reduce__(self):
        return (self.__class__, (self.checkpoint_dir,))

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        # Handle None or empty string input
        if string is None:
            raise ValueError("Cannot encode None string")

        tokens = self.processor.encode(string)

        # Defensive check - ensure tokens is not None
        if tokens is None:
            raise ValueError(f"Tokenizer returned None for string: {repr(string[:100])}")

        if bos:
            bos_id = self.bos_id
            if bos_id is None:
                # Some tokenizers (like Qwen) don't use BOS tokens - skip adding it
                if not self._warned_no_bos:
                    print("Warning: BOS token requested but tokenizer has no BOS token defined. Skipping BOS addition.")
                    self._warned_no_bos = True
            else:
                tokens = [bos_id] + tokens
        if eos:
            eos_id = self.eos_id
            if eos_id is None:
                # Some tokenizers might not have EOS properly configured - warn and skip
                if not self._warned_no_eos:
                    print("Warning: EOS token requested but tokenizer has no EOS token defined. Skipping EOS addition.")
                    self._warned_no_eos = True
            else:
                tokens = tokens + [eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        elif self.max_length > 0:
            tokens = tokens[:self.max_length]

        # Final validation before creating tensor
        if not isinstance(tokens, list) or any(t is None for t in tokens):
            raise ValueError(f"Invalid tokens list: {tokens}")

        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor, skip_special_tokens: bool = False) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens, skip_special_tokens=skip_special_tokens)
