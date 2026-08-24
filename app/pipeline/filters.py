from __future__ import annotations

import re

from app.models import ShareGPTExample

CODE_FENCE = re.compile(r"```[\s\S]*?```")
CODE_SKILLS = {"write", "review", "debug", "refactor", "idiom"}
MAX_CHARS = 12_000


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
    return True
