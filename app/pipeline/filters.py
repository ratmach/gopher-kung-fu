from __future__ import annotations

import re

from app.models import ShareGPTExample

CODE_FENCE = re.compile(r"```[\s\S]*?```")
CODE_SKILLS = {"write", "review", "debug", "refactor", "idiom"}
IMPLEMENT_SKILLS = {"write", "debug"}
MAX_CHARS = 12_000
MIN_FENCED_CHARS = 24
MAX_IMPLEMENT_PROSE = 500
MIN_FENCE_RATIO = 0.45


def _fenced_len_and_prose(gpt: str) -> tuple[int, int]:
    fenced = sum(len(match.group(0)) for match in CODE_FENCE.finditer(gpt))
    prose = len(CODE_FENCE.sub("", gpt).strip())
    return fenced, prose


def keep_example(example: ShareGPTExample) -> bool:
    if len(example.conversations) < 2:
        return False
    human = example.conversations[0].value.strip()
    gpt = example.conversations[-1].value.strip()
    if not human or not gpt:
        return False
    if len(human) + len(gpt) > MAX_CHARS:
        return False
    skill = str(example.meta.get("skill", ""))
    if skill in CODE_SKILLS and not CODE_FENCE.search(gpt):
        return False
    if skill in IMPLEMENT_SKILLS:
        fenced, prose = _fenced_len_and_prose(gpt)
        if fenced < MIN_FENCED_CHARS:
            return False
        total = fenced + prose
        if prose > MAX_IMPLEMENT_PROSE and (fenced / total if total else 0) < MIN_FENCE_RATIO:
            return False
    return True
