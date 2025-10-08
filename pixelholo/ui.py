"""UI related helpers."""

import cv2
import numpy as np


def overlay_icon(background_img, icon_img, padding: int = 20, scale_divisor: int = 8):
    """Overlay a smaller icon onto the bottom-right of a background image."""
    bg_h, bg_w, _ = background_img.shape
    icon_h, icon_w, _ = icon_img.shape

    scale_factor = (bg_w / scale_divisor) / icon_w
    new_w = int(icon_w * scale_factor)
    new_h = int(icon_h * scale_factor)
    resized_icon = cv2.resize(icon_img, (new_w, new_h))

    alpha = resized_icon[:, :, 3] / 255.0
    bgr_icon = resized_icon[:, :, :3]
    inv_alpha = 1.0 - alpha

    x_offset = bg_w - new_w - padding
    y_offset = bg_h - new_h - padding
    roi = background_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    blended_roi = (
        bgr_icon * alpha[..., np.newaxis] + roi * inv_alpha[..., np.newaxis]
    ).astype(background_img.dtype)

    result_img = background_img.copy()
    result_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = blended_roi
    return result_img
