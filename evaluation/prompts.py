# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The prompts of the sample generations: the built-in defaults, or a prompts file given to the CLI.

An instruction prompt follows the training template of instruct rows (`training/data/formats.py`): instruction,
an optional input after a blank line, and a trailing blank line where the output starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONTINUATION = "continuation"
INSTRUCTION = "instruction"
PROMPT_KINDS = (CONTINUATION, INSTRUCTION)
FILE_SEPARATOR = "---"  # a line of its own between the prompts of a prompts file


@dataclass(frozen=True)
class Prompt:
    text: str
    kind: str = CONTINUATION


def instruction_prompt(instruction: str, input_text: str = "") -> Prompt:
    """
    An instruction prompt in the training template, ending where the model's output starts.
    """

    text = instruction.strip()
    if input_text.strip():
        text += "\n\n" + input_text.strip()
    return Prompt(text + "\n\n", INSTRUCTION)


DEFAULT_PROMPTS: tuple[Prompt, ...] = (
    Prompt("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It was"),
    Prompt('def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'),
    Prompt("Theorem. Every bounded monotone sequence of real numbers converges.\n\nProof."),
    Prompt("Once upon a time, in a small village by the sea,"),
    Prompt("How to bake a tasty fruit cake:\n"),
    Prompt("How to make a tasty pasta:\n"),
    instruction_prompt("Explain in two sentences why the sky is blue."),
    instruction_prompt("Write a Python function that reverses a string."),
    instruction_prompt("What is 17 + 25? Show your reasoning."),
    instruction_prompt("Translate the following sentence to German.", "The weather is nice today."),
    instruction_prompt("Please tell me funny cat facts."),
)


def load_prompts_file(path: str | Path) -> list[Prompt]:
    """
    Prompts separated by `---` lines; a first line `# instruction` makes the prompt an instruction prompt (its
    first paragraph the instruction, the rest the input), `# continuation` is the default.
    """

    text = Path(path).read_text(encoding="utf-8")
    prompts = []
    for block in text.split(f"\n{FILE_SEPARATOR}\n"):
        block = block.strip("\n")
        if not block.strip():
            continue
        kind = CONTINUATION
        first_line, _, rest = block.partition("\n")
        if first_line.strip().startswith("#"):
            kind = first_line.strip("# ").strip().lower()
            if kind not in PROMPT_KINDS:
                raise ValueError(f"{path}: unknown prompt kind {kind!r}; use one of {PROMPT_KINDS}")
            block = rest
        if kind == INSTRUCTION:
            instruction, _, input_text = block.partition("\n\n")
            prompts.append(instruction_prompt(instruction, input_text))
        else:
            prompts.append(Prompt(block))
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def load_prompts(file: str | Path | None = None) -> list[Prompt]:
    """
    The prompts to sample with: the file when given, else `DEFAULT_PROMPTS`.
    """

    if file is not None:
        return load_prompts_file(file)
    return list(DEFAULT_PROMPTS)
