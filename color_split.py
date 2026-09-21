#!/usr/bin/env python3
"""Split an image into distinct areas of different colors.

Usage:
    python color_split.py image.jpg
    python color_split.py image.jpg --k 5 --save-dir out --no-show
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    """Read an image, or exit with a message if the path is unreadable."""
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not read image: {path}")
    return image


def cluster_colors(image, k):
    """Group every pixel into k colors with k-means.

    Returns a (height, width) array of cluster ids and the k cluster colors.
    """
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(image.shape[:2]), centers.astype(np.uint8)


def extract_color(image, labels, index):
    """Keep only the pixels belonging to one cluster; white out the rest."""
    part = np.full_like(image, 255)
    part[labels == index] = image[labels == index]
    return part


def show(image, parts, centers):
    """Display the original alongside each color region."""
    fig, axes = plt.subplots(1, len(parts) + 1, figsize=(4 * (len(parts) + 1), 4))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")

    for ax, part, center in zip(axes[1:], parts, centers):
        b, g, r = center
        ax.imshow(cv2.cvtColor(part, cv2.COLOR_BGR2RGB))
        ax.set_title(f"RGB({r},{g},{b})")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image_path", help="path to the input image")
    parser.add_argument("--k", type=int, default=3,
                        help="number of colors to split into (default: 3)")
    parser.add_argument("--save-dir", default=".",
                        help="directory to write color_N.png into (default: current)")
    parser.add_argument("--no-show", action="store_true",
                        help="skip the display window")
    return parser.parse_args()


def main():
    args = parse_args()

    image = load_image(args.image_path)
    labels, centers = cluster_colors(image, args.k)

    os.makedirs(args.save_dir, exist_ok=True)
    parts = [extract_color(image, labels, i) for i in range(args.k)]
    for i, part in enumerate(parts):
        cv2.imwrite(os.path.join(args.save_dir, f"color_{i}.png"), part)

    if not args.no_show:
        show(image, parts, centers)


if __name__ == "__main__":
    main()
