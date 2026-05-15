def parse_atari_tag(tag: str) -> str:
    """Parse atari:Pong or atari:Pong:agent57 -> ALE/Pong-v5."""
    if tag.startswith("atari:"):
        parts = tag.split(":", 1)[1].strip().split(":")
        game = parts[0].split("-")[0]
    elif tag.startswith("ALE/"):
        return tag if "-v" in tag else f"{tag}-v5"
    else:
        raise ValueError(f"Expected atari:Game or ALE/Game-v5, got: {tag}")
    return f"ALE/{game}-v5"


def is_atari_env_tag(env_tag: str) -> bool:
    return str(env_tag).startswith(("atari:", "ALE/"))


def normalize_dm_control_tag(env_tag: str, *, from_pixels: bool) -> str:
    t = str(env_tag)
    if from_pixels and (t.startswith("dm:") or t.startswith("dm_control/")):
        parts = t.split(":")
        if parts[-1] != "pixels":
            return f"{t}:pixels"
    return t
