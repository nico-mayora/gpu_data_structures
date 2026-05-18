import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import numpy as np
import cv2


def hable(x):
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def filter_colour(rgb, exposure=0.5, white=11.2):
    rgb = rgb * exposure
    rgb = hable(rgb) / hable(white)
    return np.sqrt(np.clip(rgb, 0.0, None))


def main():
    if len(sys.argv) < 3:
        print("usage: python tonemap.py <input.exr> <output.png> [exposure]")
        sys.exit(1)

    exposure = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"error: failed to read {sys.argv[1]}")
        sys.exit(1)
    img = img.astype(np.float32)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = filter_colour(img, exposure=exposure)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(sys.argv[2], img)


if __name__ == "__main__":
    main()
