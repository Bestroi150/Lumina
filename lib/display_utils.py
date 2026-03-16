# -*- coding: utf-8 -*-
"""Utils for display input and output"""

import matplotlib.pyplot as plt
from PIL import Image


def open_image(image_path: str) -> Image.Image:
    """Open an image from a path."""
    return Image.open(image_path)


def display_baselines(image, baselines, boundaries=None):
    """Display baselines and boundaries on an image."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    for idx, baseline in enumerate(baselines):
        baseline_x = [point[0] for point in baseline]
        baseline_y = [point[1] for point in baseline]
        ax.plot(baseline_x, baseline_y, color='blue', linewidth=0.7)
    if boundaries:
        for boundary in boundaries:
            boundary_x = [point[0] for point in boundary]
            boundary_y = [point[1] for point in boundary]
            ax.plot(boundary_x, boundary_y, color='red', linestyle='--', linewidth=1)

    fig_special, ax_special = plt.subplots(figsize=(10, 10))
    ax_special.set_xlim(0, image.size[0])
    ax_special.set_ylim(0, image.size[1])
    ax_special.invert_yaxis()
    for idx, baseline in enumerate(baselines):
        baseline_x = [point[0] for point in baseline]
        baseline_y = [point[1] for point in baseline]
        ax_special.plot(baseline_x, baseline_y, color='blue', linewidth=0.7)
        ax_special.text(baseline_x[0], baseline_y[0], str(idx), fontsize=10, color='red')
    return fig, fig_special


def display_baselines_with_text(image, baselines, lines):
    """Display baselines with text on an image."""
    fig_special, ax_special = plt.subplots(figsize=(10, 10))
    ax_special.set_xlim(0, image.size[0])
    ax_special.set_ylim(0, image.size[1])
    ax_special.invert_yaxis()
    for idx, group in enumerate(zip(lines, baselines)):
        baseline_x = [point[0] for point in group[1]]
        baseline_y = [point[1] for point in group[1]]
        ax_special.text(baseline_x[0], baseline_y[0], f"{str(idx)}: {group[0]}", fontsize=5.5, color='black')
    ax_special.axis('off')
    return fig_special


def prepare_segments(seg_obj):
    """Prepare baselines and boundaries for display."""
    baselines = []
    boundaries = []
    for line in seg_obj.lines:
        baselines.append(line.baseline)
        boundaries.append(line.boundary)
    return baselines, boundaries
