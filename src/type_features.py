# type_features.py
from typing import List, Optional

def build_text_with_types(
    label_text: str,
    type_terms: Optional[List[str]],
    mode: str = "prefix",
    max_types: int = 3,
) -> str:
    if not type_terms:
        return label_text
    toks = [str(t).strip() for t in type_terms if str(t).strip()]
    toks = toks[:max_types]
    if not toks:
        return label_text

    types_blk = "[TYPES: " + "; ".join(toks) + "]"
    if mode == "prefix":
        return f"{types_blk} {label_text}".strip()
    if mode == "suffix":
        return f"{label_text} {types_blk}".strip()
    # bag
    return f"{' '.join(toks)} {label_text}".strip()
