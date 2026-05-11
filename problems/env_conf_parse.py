def parse_tag_options(tag, from_pixels):
    """Parse shared options from tag. Returns (tag, frozen_noise, from_pixels)."""
    frozen_noise = False
    while ":" in tag:
        x = tag.split(":")
        opt = x[-1]
        if opt == "fn":
            frozen_noise = True
        elif opt == "pixels":
            from_pixels = True if from_pixels is None else from_pixels
        else:
            break
        tag = ":".join(x[:-1])
    return tag, frozen_noise, from_pixels
