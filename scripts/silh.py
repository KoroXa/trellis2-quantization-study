import struct, json, sys
import numpy as np

def load_positions(path, max_pts=60000):
    with open(path, "rb") as f:
        magic, ver, length = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67
        bins = []
        js = None
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            clen, ctype = struct.unpack("<II", h)
            data = f.read(clen)
            if ctype == 0x4E4F534A:
                js = json.loads(data)
            else:
                bins.append(data)
    assert len(bins) == 1, "expected 1 BIN chunk"
    blob = bins[0]
    pts = []
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            acc = js["accessors"][prim["attributes"]["POSITION"]]
            bv = js["bufferViews"][acc["bufferView"]]
            off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            n = acc["count"]
            arr = np.frombuffer(blob, dtype=np.float32, count=n * 3, offset=off).reshape(n, 3)
            pts.append(arr)
    allp = np.concatenate(pts, axis=0)
    if len(allp) > max_pts:
        idx = np.random.default_rng(0).choice(len(allp), max_pts, replace=False)
        allp = allp[idx]
    return allp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sys.argv[1:]
n = len(files)
fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
if n == 1:
    axes = [[axes[0]], [axes[1]]]
for i, p in enumerate(files):
    pts = load_positions(p)
    name = p.replace("\\", "/").split("/")[-1]
    ax = axes[0][i]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="black")
    ax.set_title(f"{name}\nfront (X-Y)")
    ax.set_aspect("equal")
    ax = axes[1][i]
    ax.scatter(pts[:, 0], pts[:, 2], s=0.3, c="black")
    ax.set_title("side (X-Z)")
    ax.set_aspect("equal")
out = sys.argv[0].replace("silh.py", "silh_out.png")
fig.tight_layout()
fig.savefig(out, dpi=80)
print("saved", out)
