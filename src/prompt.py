"""SmolVLM prompt construction.

Mirrors the prompt format from ``src/starter_notebook.ipynb`` so that
``.py`` runs and notebook reference behave identically. The choice
letter mapping is fixed: index ``i`` maps to ``CHOICE_LETTERS[i]``.
"""

from __future__ import annotations

import pandas as pd

CHOICE_LETTERS = "ABCDEFGHIJ"


def letter_for(index: int) -> str:
    return CHOICE_LETTERS[index]


def index_for(letter: str) -> int | None:
    letter = letter.strip().upper()
    if not letter:
        return None
    pos = CHOICE_LETTERS.find(letter[0])
    return pos if pos >= 0 else None


def _build_user_text(
    row,
    *,
    include_lecture: bool = True,
    include_hint: bool = True,
    include_solution: bool = False,
    max_context_chars: int | None = 2000,
) -> str:
    """User-text portion of the prompt, without any <image> token or
    chat scaffolding. Used by both the manual builder and the chat-
    template builder.

    ``include_solution`` is the option-(c) recipe from paper-scout
    2026-05-06 grounding: prepend the ScienceQA-style explanation
    (`solution` column) as additional context for *training only*.
    Test rows lack `solution`, so leave it False at inference; if
    a row has no solution column or value the flag is a no-op.
    """
    context_parts: list[str] = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")
    solution = row.get("solution", None) if include_solution else None
    if include_solution and solution is not None and pd.notna(solution) and str(solution).strip():
        context_parts.append(str(solution).strip())
    if include_lecture and pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if include_hint and pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    context_str = "\n".join(context_parts)
    if max_context_chars is not None and len(context_str) > max_context_chars:
        context_str = context_str[:max_context_chars].rstrip() + " ..."

    choices = row["choices"]
    choices_str = "\n".join(f"  {CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices))

    out = ""
    if context_str:
        out += f"Context:\n{context_str}\n\n"
    out += f"Question: {row['question']}\nChoices:\n{choices_str}\nAnswer:"
    return out


def build_chat_prompt(
    processor,
    row,
    *,
    include_answer: bool = False,
    max_context_chars: int | None = 2000,
    include_lecture: bool = True,
    include_hint: bool = True,
    include_solution: bool = False,
) -> str:
    """Build the prompt via ``processor.apply_chat_template`` so the
    model sees its native ``<|im_start|>User:`` / ``Assistant:``
    delimiters. SmolVLM-Instruct is chat-tuned; the manual prompt in
    ``build_prompt`` may have been a distribution-mismatch suppressor.
    """
    user_text = _build_user_text(
        row, include_lecture=include_lecture, include_hint=include_hint,
        include_solution=include_solution, max_context_chars=max_context_chars,
    )
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]},
    ]
    if include_answer:
        answer_letter = CHOICE_LETTERS[int(row["answer"])]
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": answer_letter},
        ]})
        return processor.apply_chat_template(messages, add_generation_prompt=False)
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def build_prompt(
    row,
    *,
    include_answer: bool = False,
    max_context_chars: int | None = 2000,
    include_lecture: bool = True,
    include_hint: bool = True,
    include_solution: bool = False,
) -> str:
    """Build the multimodal QA prompt for a single row.

    ``row`` is a ``pandas.Series`` or compatible mapping with the
    competition columns. The ``<image>`` token is required by the
    SmolVLM processor (it expands the marker to many image tokens
    inside the processor; do not let downstream truncation cut into
    the expanded image span — cap the context here instead).
    """
    context_parts: list[str] = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")
    solution = row.get("solution", None) if include_solution else None
    if include_solution and solution is not None and pd.notna(solution) and str(solution).strip():
        context_parts.append(str(solution).strip())
    if include_lecture and pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if include_hint and pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    context_str = "\n".join(context_parts)
    if max_context_chars is not None and len(context_str) > max_context_chars:
        context_str = context_str[:max_context_chars].rstrip() + " ..."

    choices = row["choices"]
    choices_str = "\n".join(
        f"  {CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices)
    )

    out = "<image>\n"
    if context_str:
        out += f"Context:\n{context_str}\n\n"
    out += f"Question: {row['question']}\n"
    out += f"Choices:\n{choices_str}\n"
    out += "Answer:"

    if include_answer:
        answer_idx = int(row["answer"])
        out += f" {CHOICE_LETTERS[answer_idx]}"

    return out
