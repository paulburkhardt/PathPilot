import PIL
import numpy as np

def _resize_pil_seg_mask(img, long_edge_size):
    S = max(img.size)
    interp = PIL.Image.NEAREST
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_seg_mask(img, size, square_ok=False, return_transformation=False):
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_seg_mask(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_seg_mask(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    unnormalized_img=np.asarray(img),

    return unnormalized_img