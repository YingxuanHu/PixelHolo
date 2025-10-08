import time
from pathlib import Path

import torch
from lipsync import LipSync

from pixelholo import config

PROJECT_ROOT = config.PROJECT_ROOT
CHECKPOINT_PATH = config.WEIGHTS_DIR / "wav2lip_gan.pth"
SAMPLE_VIDEO = PROJECT_ROOT / "examples" / "TalkingVideo.mov"
SAMPLE_AUDIO = config.OUTPUTS_DIR / config.OUTPUT_WAV_NAME
OUTPUT_VIDEO = config.OUTPUTS_DIR / "result.mp4"

print("Attempting to load LipSync model...")
print(f"Device: cuda")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

start_time = time.time()

try:
    lip = LipSync(
        model="wav2lip",
        checkpoint_path=str(CHECKPOINT_PATH),
        device="cuda",
        nosmooth=True,
    )
    end_time = time.time()
    print(f"\n✅ SUCCESS: LipSync model loaded in {end_time - start_time:.2f} seconds.")
except Exception as exc:
    print(f"\n❌ FAILED: An error occurred during loading: {exc}")
    raise

lip.sync(
    str(SAMPLE_VIDEO),
    str(SAMPLE_AUDIO),
    str(OUTPUT_VIDEO),
)
