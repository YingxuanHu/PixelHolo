"""Background removal utilities."""

import cv2
import numpy as np
from PIL import Image

import torch


def remove_background_advanced(image_path: str, output_path: str) -> bool:
    """Advanced background removal focusing on preserving people's faces and bodies."""
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


def try_mediapipe_segmentation(img, output_path: str) -> bool:
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
        cv2.imwrite(output_path, result)
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


def remove_background_from_video(input_video_path: str, output_video_path: str) -> bool:
    """Remove background frame-by-frame using rembg."""
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

        if torch.cuda.is_available():
            session = new_session("u2net", providers=["CUDAExecutionProvider"])
            print("Using GPU acceleration for background removal")
        else:
            session = new_session("u2net")
            print("Using CPU for background removal")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames}")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            output_pil = remove(pil_frame, session=session)
            result_array = np.array(output_pil)

            if result_array.shape[2] == 4:
                alpha = result_array[:, :, 3] / 255.0
                result_frame = result_array[:, :, :3] * alpha[:, :, np.newaxis]
                result_frame = cv2.cvtColor(result_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                result_frame = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

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
