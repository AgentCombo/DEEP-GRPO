from typing import Any, List, Optional, Sequence, Tuple


def get_eos_token_id(tokenizer: Any) -> Optional[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, (list, tuple)):
        if not eos_token_id:
            return None
        eos_token_id = eos_token_id[0]
    return int(eos_token_id)


def append_eos_if_missing(
    token_ids: Sequence[int],
    token_mask: Sequence[int],
    tokenizer: Any,
) -> Tuple[List[int], List[int], bool]:
    ids = list(token_ids)
    mask = list(token_mask)
    assert len(ids) == len(mask), (
        f"token_ids length {len(ids)} != token_mask length {len(mask)}"
    )

    eos_token_id = get_eos_token_id(tokenizer)
    if eos_token_id is None:
        return ids, mask, False
    if ids and ids[-1] == eos_token_id:
        return ids, mask, False

    ids.append(eos_token_id)
    mask.append(1)
    return ids, mask, True
