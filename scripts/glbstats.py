import struct, json, sys

def glb_stats(path):
    with open(path, "rb") as f:
        magic, ver, length = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67, path
        chunks = []
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            clen, ctype = struct.unpack("<II", h)
            chunks.append((ctype, f.read(clen)))
    js = json.loads([c for t, c in chunks if t == 0x4E4F534A][0])
    total_verts = 0
    total_tris = 0
    gmin = [1e9, 1e9, 1e9]
    gmax = [-1e9, -1e9, -1e9]
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pa = prim["attributes"]["POSITION"]
            acc = js["accessors"][pa]
            total_verts += acc["count"]
            if "min" in acc:
                for i in range(3):
                    gmin[i] = min(gmin[i], acc["min"][i])
                    gmax[i] = max(gmax[i], acc["max"][i])
            if "indices" in prim:
                ia = js["accessors"][prim["indices"]]
                total_tris += ia["count"] // 3
    ext = [gmax[i] - gmin[i] for i in range(3)]
    return total_verts, total_tris, gmin, gmax, ext

for p in sys.argv[1:]:
    try:
        v, t, mn, mx, ex = glb_stats(p)
        print(f"{p.split(chr(92))[-1]:22} verts={v:>9} tris={t:>9} "
              f"min={[round(x,3) for x in mn]} max={[round(x,3) for x in mx]} ext={[round(x,3) for x in ex]}")
    except Exception as e:
        print(f"{p}: ERROR {e}")
