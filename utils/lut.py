import numpy as np
from PIL import Image, ImageOps

def load_cube_lut(path):
    size = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    triplets = []

    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            up = s.upper()
            if up.startswith("TITLE"):
                continue
            elif up.startswith("LUT_3D_SIZE"):
                size = int(s.split()[1])
            elif up.startswith("DOMAIN_MIN"):
                domain_min = np.array([float(x) for x in s.split()[1:4]], dtype=np.float32)
            elif up.startswith("DOMAIN_MAX"):
                domain_max = np.array([float(x) for x in s.split()[1:4]], dtype=np.float32)
            else:
                parts = s.split()
                if len(parts) == 3:
                    triplets.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if size is None:
        n = int(round(len(triplets) ** (1.0/3.0)))
        if n**3 != len(triplets):
            raise ValueError(f"Impossibile dedurre LUT_3D_SIZE da {len(triplets)} valori")
        size = n

    lut = np.asarray(triplets, dtype=np.float32).reshape((size, size, size, 3))
    lut = np.transpose(lut, (2, 1, 0, 3))  # [r, g, b, 3]
    return lut, size # , domain_min, domain_max

def apply_cube_lut_preserve(im: Image.Image, lut, size, domain_min=None, domain_max=None):
    if im is None:
        return None
    if domain_min is None:
        domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if domain_max is None:
        domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    w, h = im.size
    im_src = ImageOps.exif_transpose(im)
    alpha = im_src.getchannel("A") if im_src.mode in ("RGBA", "LA") else None

    arr = np.asarray(im_src.convert("RGB"), dtype=np.float32) / 255.0
    # normalizza al dominio del file
    scale = (domain_max - domain_min)
    scale[scale == 0] = 1.0
    arr = (arr - domain_min) / scale
    arr = np.clip(arr, 0.0, 1.0)

    idx = np.clip((arr * (size - 1)).astype(int), 0, size - 1)
    mapped = lut[idx[...,0], idx[...,1], idx[...,2]]
    mapped = np.clip(mapped * 255.0, 0, 255).astype(np.uint8)

    out = Image.fromarray(mapped, "RGB")
    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(alpha)
    if out.size != (w, h):
        out = out.resize((w, h), Image.BICUBIC)
    return out