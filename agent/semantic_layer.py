def _canonical_name(name):
    return re.sub(r"[^a-z0-9]", "_", name).strip("_")
