"""Background removal utilities."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch


def remove_background_advanced(image_path: Path | str, output_path: Path | str) -> bool:
    """Advanced background removal focusing on preserving people's faces and bodies."""
    image_path = str(image_path)
    output_path = str(output_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return False

    try:
        if try_mediapipe_segmentation(img, output_path):
            print(f"Background removed using MediaPipe and saved to: {output_path}")
            return True
    except Exception:
        print("MediaPipe not available, using alternative methods...")

    person_mask = detect_person_region(img)
    if person_mask is not None:
        result = enhanced_grabcut_with_person_detection(img, person_mask)
    else:
        result = enhanced_grabcut_with_skin_detection(img)

    cv2.imwrite(output_path, result)
    print(f"Background removed and saved to: {output_path}")
    return True


def try_mediapipe_segmentation(img, output_path: Path | str) -> bool:
    """Use rembg with GPU acceleration for better background removal."""
    try:
        from rembg import remove, new_session

        if torch.cuda.is_available():
            session = new_session("u2net", providers=["CUDAExecutionProvider"])
        else:
            session = new_session("u2net")

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        output_pil = remove(pil_img, session=session)
        result_array = np.array(output_pil)

        if result_array.shape[2] == 4:
            alpha = result_array[:, :, 3] / 255.0
            result = result_array[:, :, :3] * alpha[:, :, np.newaxis]
        else:
            result = result_array

        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result)
        return True
    except Exception as exc:
        print(f"rembg segmentation failed: {exc}")
        return False


def detect_person_region(img):
    """Detect person/human regions using HOG descriptor or Haar cascades."""
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, _ = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32), scale=1.05)

        if len(boxes) > 0:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for (x, y, w, h) in boxes:
                x = max(0, x - 20)
                y = max(0, y - 20)
                w = min(img.shape[1] - x, w + 40)
                h = min(img.shape[0] - y, h + 40)
                mask[y : y + h, x : x + w] = 255
            return mask
    except Exception as exc:
        print(f"Person detection failed: {exc}")
    return None


def detect_skin_regions(img):
    """Detect skin regions to help identify people."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)

    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)

    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    return skin_mask


def enhanced_grabcut_with_person_detection(img, person_mask):
    """Enhanced GrabCut using detected person regions."""
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[person_mask == 255] = cv2.GC_PR_FGD
    mask[person_mask == 0] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK)

    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    final_mask = refine_mask(final_mask, img)
    return img * final_mask[:, :, np.newaxis]


def enhanced_grabcut_with_skin_detection(img):
    """Enhanced GrabCut using skin detection and improved initialization."""
    height, width = img.shape[:2]
    skin_mask = detect_skin_regions(img)

    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (int(width * 0.15), int(height * 0.05), int(width * 0.7), int(height * 0.9))

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

    mask[skin_mask > 0] = cv2.GC_PR_FGD
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    final_mask = refine_mask(final_mask, img)
    return img * final_mask[:, :, np.newaxis]


def refine_mask(mask, img):
    """Refine mask edges for cleaner results."""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    _, mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)

    edges = cv2.Canny(img, 100, 200)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    edges = edges / 255.0

    mask = np.clip(mask + edges * 0.2, 0, 1)
    return mask


def remove_background_from_video(
    input_video_path: Path | str,
    output_video_path: Path | str,
    *,
    process_every: int = 1,
    ema: float = 0.7,
    mask_downscale_max: int = 768,
    model_name: str | None = None,
) -> bool:
    """Remove background frame-by-frame using rembg.

    Parameters
    - process_every: run the segmenter every N frames; reuse last mask between.
    - ema: exponential moving average factor for temporal mask smoothing (0..1).
    - mask_downscale_max: compute masks on downscaled frames for speed, then upsample.
    - model_name: override rembg model (e.g., "u2netp", "isnet-general-use").
    """
    input_video_path = str(input_video_path)
    output_video_path = str(output_video_path)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        from rembg import remove, new_session

        use_cuda = torch.cuda.is_available()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else None
        chosen_model = model_name or ("isnet-general-use" if use_cuda else "u2netp")

        session = new_session(chosen_model, providers=providers) if providers else new_session(chosen_model)
        print("Using GPU acceleration for background removal" if use_cuda else "Using CPU for background removal")
        print(f"rembg model: {chosen_model}; process_every={process_every}, ema={ema}, downscale_max={mask_downscale_max}")

        last_mask = None  # float32 [H,W] in [0,1]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames}")

            need_infer = (process_every <= 1) or (frame_count % process_every == 1) or (last_mask is None)

            if need_infer:
                h, w = frame.shape[:2]
                # Compute a smaller frame for mask to speed up inference
                scale = 1.0
                if mask_downscale_max > 0:
                    max_dim = max(h, w)
                    if max_dim > mask_downscale_max:
                        scale = mask_downscale_max / float(max_dim)
                if scale < 1.0:
                    small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    small_frame = frame

                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                pil_small = Image.fromarray(rgb_small)
                output_pil = remove(pil_small, session=session)
                out_np = np.array(output_pil)

                if out_np.ndim == 3 and out_np.shape[2] == 4:
                    alpha_small = out_np[:, :, 3].astype(np.float32) / 255.0
                else:
                    # If model returns RGB only, assume full foreground (rare for rembg)
                    alpha_small = np.ones(out_np.shape[:2], dtype=np.float32)

                # Upsample mask back to original resolution if needed
                if alpha_small.shape[0] != h or alpha_small.shape[1] != w:
                    alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_CUBIC)
                else:
                    alpha = alpha_small

                # Temporal EMA smoothing to reduce flicker
                if last_mask is None:
                    smoothed = alpha
                else:
                    smoothed = (ema * alpha + (1.0 - ema) * last_mask)

                # Optional light edge refinement for consistency
                smoothed = np.clip(smoothed, 0.0, 1.0)

                last_mask = smoothed
                current_mask = smoothed
            else:
                current_mask = last_mask

            # Apply mask to current full-res frame
            result_frame = (frame.astype(np.float32) * current_mask[:, :, None]).astype(np.uint8)

            out.write(result_frame)
    except Exception as exc:
        print(f"Error during video background removal: {exc}")
        cap.release()
        out.release()
        return False

    cap.release()
    out.release()
    print(f"Background removal completed. Output saved to: {output_video_path}")
    return True
