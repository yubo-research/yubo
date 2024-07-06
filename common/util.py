def parse_kv(argv):
    d = {}
    for a in argv:
        k, v = a.split("=")
        d[k] = v
    return d
