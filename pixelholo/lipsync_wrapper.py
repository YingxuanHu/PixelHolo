"""Local extensions for the third-party `lipsync` package."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import torch

from lipsync.lipsync import LipSync as _BaseLipSync


class LipSync(_BaseLipSync):
    """Thin wrapper that guards against minor shape mismatches during writing."""

    def _load_model_for_inference(self) -> torch.nn.Module:
        """Load checkpoints that bundle weights under nested keys."""
        from lipsync.models import MODEL_REGISTRY

        model_cls = MODEL_REGISTRY[self.model.lower()]
        model = model_cls()

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if isinstance(checkpoint, dict):
            checkpoint = {
                key.replace("module.", "", 1): value for key, value in checkpoint.items()
            }

        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing:
            print(f"⚠️ Missing checkpoint keys: {len(missing)} skipped during load.")
        if unexpected:
            print(f"⚠️ Unexpected checkpoint keys: {len(unexpected)} ignored.")

        model = model.to(self.device)
        return model.eval()

    @staticmethod
    def _write_predicted_frames(
        pred: torch.Tensor,
        frames: List[np.ndarray],
        coords: List[Tuple[int, int, int, int]],
        out: cv2.VideoWriter,
    ) -> None:
        """Write predictions back into frames while tolerating off-by-one crops."""
        pred_np = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)

        for predicted_face, frame, coord in zip(pred_np, frames, coords):
            y1, y2, x1, x2 = coord

            # Clamp coordinates to frame boundaries to avoid invalid slices.
            y1 = max(0, y1)
            x1 = max(0, x1)
            y2 = min(frame.shape[0], y2)
            x2 = min(frame.shape[1], x2)

            if y2 <= y1 or x2 <= x1:
                continue  # Nothing to write for this frame.

            region = frame[y1:y2, x1:x2]
            target_h, target_w = region.shape[:2]

            resized = cv2.resize(predicted_face, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            if resized.shape != region.shape:
                resized = resized[:target_h, :target_w]

            region[:] = resized
            out.write(frame)
