# Modified from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f.
# Changes (c) 2025-2026 Tobias Kerner: working chat-template assistant masking, collate/padding changes. See README and git history.
import torch
from torch.utils.data import default_collate

from typing import Optional
from torch.utils.data._utils.collate import collate_tensor_fn
from .tokenizer import Tokenizer
import re


def fix_chat_template_for_masking(tokenizer):
    """Fix chat template to support assistant token masking.

    Llama-2 chat templates don't have {% generation %} tags by default,
    which are required for return_assistant_tokens_mask to work.
    This function adds those tags around assistant responses.

    This is applied once per tokenizer when needed.

    Args:
        tokenizer: HuggingFace tokenizer (usually tokenizer.processor)
    """
    if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
        return  # No chat template to fix

    # Check if already has generation tags
    if '{% generation %}' in tokenizer.chat_template:
        return  # Already fixed

    # Fix Llama-2 style template by adding generation tags around assistant content
    # Original: {% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}
    # Fixed:    {% elif message['role'] == 'assistant' %}{% generation %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endgeneration %}

    original_template = tokenizer.chat_template

    # Replace assistant response section with generation-tagged version
    if "message['role'] == 'assistant'" in original_template:
        # Pattern: finds the assistant block and wraps the output in generation tags
        pattern = r"(\{%\s*elif\s+message\['role'\]\s*==\s*'assistant'\s*%\})(.*?)(\{%\s*endif\s*%\})"

        def add_generation_tags(match):
            prefix = match.group(1)  # {% elif message['role'] == 'assistant' %}
            content = match.group(2)  # {{ ' '  + content.strip() + ' ' + eos_token }}
            suffix = match.group(3)  # {% endif %}
            return f"{prefix}{{% generation %}}{content}{{% endgeneration %}}{suffix}"

        fixed_template = re.sub(pattern, add_generation_tags, original_template, flags=re.DOTALL)

        # Apply the fix
        tokenizer.chat_template = fixed_template
        print("  ✓ Fixed chat template to support assistant token masking")


def pass_text(row, tokenizer, add_bos, add_eos):
    input_string = row.get("text")
    if input_string is None:
        raise ValueError(f"Dataset row has None 'text' field. Row keys: {list(row.keys())}")
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=add_eos)
    label_tokens = input_tokens.clone()
    return (input_tokens, label_tokens)


def concat_input_target(row, tokenizer, add_bos, add_eos):
    input_part = row.get("input")
    target_part = row.get("target")
    if input_part is None or target_part is None:
        raise ValueError(f"Dataset row has None 'input' or 'target' field. Row keys: {list(row.keys())}")
    input_string = input_part + target_part
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=add_eos)
    label_tokens = input_tokens.clone()
    return (input_tokens, label_tokens)


def condition_input_supervise_target(row, tokenizer, add_bos, add_eos):
    input_string = row["input"]
    joint_string = row["input"] + row["target"]
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=False)
    joint_tokens = tokenizer.encode(joint_string, bos=add_bos, eos=add_eos)
    label_tokens = joint_tokens.clone()
    # mask the locations of the input tokens in the joint tokens
    label_tokens[0 : len(input_tokens)] = tokenizer.pad_id
    input_tokens = joint_tokens
    return (input_tokens, label_tokens)


def concatenate_instruction_input_output(row, tokenizer, add_bos, add_eos):
    """Concatenate instruction/input/output fields and supervise only output.

    Format:
    - instruction: The main prompt/question
    - input: Optional context (can be empty string)
    - output: The expected response (what we train on)

    The instruction and input are masked in labels (not trained),
    only the output portion is supervised.
    """
    instruction = row.get("instruction", "")
    input_text = row.get("input", "")
    output = row.get("output", "")

    if instruction is None or output is None:
        raise ValueError(f"Dataset row missing 'instruction' or 'output'. Row keys: {list(row.keys())}")

    # Strip individual parts to avoid excessive newlines
    instruction = instruction.strip()
    input_text = input_text.strip() if input_text else ""
    output = output.strip()

    # Build prompt part (instruction + optional input)
    prompt = instruction
    if input_text:
        prompt = prompt + "\n\n" + input_text

    # Build full string (prompt + output)
    full_string = prompt + "\n\n" + output

    # Tokenize prompt alone to get its length
    prompt_tokens = tokenizer.encode(prompt, bos=add_bos, eos=False)

    # Tokenize full string
    full_tokens = tokenizer.encode(full_string, bos=add_bos, eos=add_eos)

    # Create labels with prompt masked
    label_tokens = full_tokens.clone()
    pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else -100
    label_tokens[0 : len(prompt_tokens)] = pad_id

    return (full_tokens, label_tokens)


def apply_chat_template_supervise_all(row, tokenizer, add_bos, add_eos):
    assert len(row["data_signature"]["keys"]) == 1, (
        "Ambiguous row format for chat template call. data signature should spec the single intended key."
    )
    key = row["data_signature"]["keys"][0]
    input_string = tokenizer.processor.apply_chat_template(row[key], tokenize=False)
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=add_eos)
    label_tokens = input_tokens.clone()
    return (input_tokens, label_tokens)


def apply_chat_template_supervise_assistant(row, tokenizer, add_bos, add_eos):
    """Apply chat template and mask non-assistant tokens.

    Uses the tokenizer's built-in chat template (e.g., Llama-2 style).
    Only trains on assistant responses, masks user prompts.
    """
    # Fix chat template to support assistant masking (idempotent - safe to call multiple times)
    fix_chat_template_for_masking(tokenizer.processor)

    assert len(row["data_signature"]["keys"]) == 1, (
        "Ambiguous row format for chat template call. data signature should spec the single intended key."
    )
    key = row["data_signature"]["keys"][0]

    assert isinstance(row[key], list), "is not in chat format"
    tokenized_string = tokenizer.processor.apply_chat_template(
        row[key],
        tokenize=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Extract input_ids and create labels with proper masking
    input_ids = tokenized_string["input_ids"][0]  # Remove batch dimension
    labels = input_ids.clone()

    # Mask user tokens: where assistant_masks is False (0), set to pad_id
    assistant_mask = torch.tensor(tokenized_string["assistant_masks"][0], dtype=torch.bool)
    labels[~assistant_mask] = tokenizer.pad_id

    return (input_ids, labels)


format_fn_registry = {
    "pass_text": pass_text,
    "concat_input_target": concat_input_target,
    "condition_input_supervise_target": condition_input_supervise_target,
    "concatenate_instruction_input_output": concatenate_instruction_input_output,
    "apply_chat_template_supervise_all": apply_chat_template_supervise_all,
    "apply_chat_template_supervise_assistant": apply_chat_template_supervise_assistant,
}


def apply_formatting(row, tokenizer, add_bos, add_eos):
    # pkds, single tensor
    if isinstance(row, torch.Tensor):
        return row, row.clone()
    # pkds, tuple of tensors
    if isinstance(row, tuple):
        raise NotImplementedError("Tuple format not supported, but direct tensor pairs planned.")
        return row[0], row[1]
    # hfds, dict with format_fn from data signature
    if isinstance(row, dict):
        # we can locally override the add_bos or add_eos args if they exist in the row's data_signature
        if row["data_signature"].get("add_bos") is not None:
            add_bos = row["data_signature"]["add_bos"]
        if row["data_signature"].get("add_eos") is not None:
            add_eos = row["data_signature"]["add_eos"]

        return format_fn_registry[row["data_signature"]["format_fn"]](row, tokenizer, add_bos, add_eos)
    raise ValueError("Row format not recognized.")


def shift_inputs_and_labels(inputs_batch: torch.Tensor, labels_batch: torch.Tensor, tokenizer: Tokenizer):
    seq_len = inputs_batch.shape[1]

    input_ids = inputs_batch[:, 0 : (seq_len - 1)].contiguous().long()
    label_ids = labels_batch[:, 1:(seq_len)].contiguous().long()

    # for the input we need to replace any pad ids with the eos token
    # knowing that they're trailing so they wont contrib to activations
    # but that they do need to be valid indices in the model's embedding layer
    if tokenizer.eos_id is not None:
        input_ids[input_ids == tokenizer.pad_id] = tokenizer.eos_id  # type: ignore
    # Note that we are _not_ doing this operation for the labels,
    # since this is where we actually need the pad tokens to be present for loss to ignore them.

    return input_ids, label_ids

def _infer_cap(tokenizer, block_size):
    if isinstance(block_size, int) and block_size > 0:
        return block_size
    # many HF tokenizers set model_max_length to a huge sentinel when "unlimited"
    # mk = getattr(tokenizer, "model_max_length", None)
    # if isinstance(mk, int) and 0 < mk < 10_000_000:
    #     return mk
    # other common names
    for attr in ("n_ctx", "max_length", "seq_length"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int) and v > 0:
            # print(f"using attr {attr}: {v}", flush=True)
            return v
    # print(f"could not infer cap (block_size: {type(block_size)}, {block_size}), using default", flush=True)
    return 2048  # safe default

def generic_collate_fn(
    batch,
    tokenizer: Tokenizer,
    block_size: Optional[int] = None,
    pad_to_block_size: bool = False,
    sequence_padding_multiple: Optional[int] = None,
    add_bos=True,
    add_eos=True,
    collate_checks_enabled=True,
    all_block_size_tensors=False,
):
    cap = _infer_cap(tokenizer, block_size)

    def compute_padded_length(max_len: int) -> int:
        """Compute padded length based on padding strategy."""
        if pad_to_block_size:
            return cap
        elif sequence_padding_multiple is not None:
            from .utils import find_multiple
            return min(find_multiple(max_len, sequence_padding_multiple), cap)
        else:
            return min(max_len, cap)

    metadata = [None] * len(batch)
    for i, row in enumerate(batch):
        if isinstance(row, dict) and "data_id" in row:
            metadata[i] = row["data_id"]

    # ---------- FAST PATH ----------
    if all_block_size_tensors:
        first = batch[0]
        if isinstance(first, dict):
            try:
                coll = default_collate(batch)  # dict[str, Tensor]
                if "input_ids" in coll and torch.is_tensor(coll["input_ids"]):
                    inputs_batch = coll["input_ids"].to(torch.long)
                    if inputs_batch.size(1) > cap:             # HARD CAP
                        inputs_batch = inputs_batch[:, :cap]
                    labels_batch = inputs_batch.clone()
                else:
                    raise RuntimeError("no input_ids in collated dict")
            except Exception:
                # fallback: format each row then pad/truncate to cap
                tmp = [apply_formatting(row, tokenizer, add_bos, add_eos) for row in batch]  # [(inp, lab)]
                local = compute_padded_length(max(t[0].shape[0] for t in tmp))
                pad_id = tokenizer.pad_id or 0
                inputs_batch = torch.full((len(tmp), local), pad_id, dtype=torch.long)
                labels_batch = torch.full((len(tmp), local), pad_id, dtype=torch.long)
                for i, (inp, lab) in enumerate(tmp):
                    L = min(inp.shape[0], local)
                    inputs_batch[i, :L] = inp[:L].to(torch.long)
                    labels_batch[i, :L] = lab[:L].to(torch.long)
        else:
            # list[Tensor]
            if all(t.shape == batch[0].shape for t in batch):
                inputs_batch = torch.stack(batch, dim=0).to(torch.long)
                if inputs_batch.size(1) > cap:                 # HARD CAP
                    inputs_batch = inputs_batch[:, :cap]
                labels_batch = inputs_batch.clone()
            else:
                max_len = max(t.shape[-1] for t in batch)
                local = compute_padded_length(max_len)
                pad_id = tokenizer.pad_id or 0
                inputs_batch = torch.full((len(batch), local), pad_id, dtype=torch.long)
                labels_batch = torch.full((len(batch), local), pad_id, dtype=torch.long)
                for i, t in enumerate(batch):
                    t = t.to(torch.long)[:local]
                    L = t.shape[-1]
                    inputs_batch[i, :L] = t
                    labels_batch[i, :L] = t

        input_ids, label_ids = shift_inputs_and_labels(inputs_batch, labels_batch, tokenizer)

        # FINAL GUARD: never return longer than cap
        if input_ids.size(1) > cap:
            input_ids = input_ids[:, :cap]
            label_ids = label_ids[:, :cap]

        # CE safety: ignore pad/out-of-range
        ig = -100
        if getattr(tokenizer, "pad_id", None) is not None:
            label_ids[label_ids == tokenizer.pad_id] = ig
        vsz = getattr(tokenizer, "vocab_size", None)
        if callable(vsz):
            vsz = vsz()
        if isinstance(vsz, int):
            label_ids[(label_ids < 0) | (label_ids >= vsz)] = ig

        return input_ids.to(torch.long), label_ids.to(torch.long), metadata

    # ---------- SLOW PATH ----------
    assert cap is not None  # we always have one
    if collate_checks_enabled:
        assert isinstance(batch, list), "Batch must be a list."
        types = set(type(x) for x in batch)
        assert types.issubset({dict, torch.Tensor}), "Batch must contain only expected types."
        if dict in types:
            assert tokenizer is not None and tokenizer.pad_id is not None, \
                "If batch contains dicts, tokenizer and pad_id must be provided."

    batch = [apply_formatting(row, tokenizer, add_bos, add_eos) for row in batch]

    local = compute_padded_length(max(len(x) for row in batch for x in row))

    inputs_batch = torch.full((len(batch), local), tokenizer.pad_id or 0, dtype=torch.long)
    labels_batch = torch.full((len(batch), local), tokenizer.pad_id or 0, dtype=torch.long)
    for i, (inp, lab) in enumerate(batch):
        L_in = min(len(inp), local)
        L_lb = min(len(lab), local)
        inputs_batch[i, :L_in] = inp[:L_in].to(torch.long)
        labels_batch[i, :L_lb] = lab[:L_lb].to(torch.long)

    if torch.all(labels_batch == tokenizer.eos_id) or torch.all(labels_batch == tokenizer.pad_id):
        raise StopIteration("All tokens in batch are padding tokens.")

    input_ids, label_ids = shift_inputs_and_labels(inputs_batch, labels_batch, tokenizer)

    # FINAL GUARD
    if input_ids.size(1) > cap:
        input_ids = input_ids[:, :cap]
        label_ids = label_ids[:, :cap]

    ig = -100
    if getattr(tokenizer, "pad_id", None) is not None:
        label_ids[label_ids == tokenizer.pad_id] = ig
    vsz = getattr(tokenizer, "vocab_size", None)
    if callable(vsz):
        vsz = vsz()
    if isinstance(vsz, int):
        label_ids[(label_ids < 0) | (label_ids >= vsz)] = ig

    return input_ids.to(torch.long), label_ids.to(torch.long), metadata
