"""Extract executable Python from model completions."""

import re

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_MARKERS = re.compile(r"(^|\n)\s*(?:async\s+def|def|class|from|import)\s+", re.MULTILINE)
ONLINE_JUDGE_STYLE = "online_judge"


def extract_code(content: str, style: str) -> str:
    content = content.replace("<|im_end|>", "").strip()
    fenced = CODE_FENCE.findall(content)
    if fenced:
        return "\n\n".join(block.strip() for block in fenced)
    if style == ONLINE_JUDGE_STYLE:
        return content
    return content if CODE_MARKERS.search(content) else ""
