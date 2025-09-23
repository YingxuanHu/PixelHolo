import time
from lipsync import LipSync
import torch

# This is the path to your model file
# Make sure it's correct relative to where you run the script
CHECKPOINT_PATH = 'weights/wav2lip_gan.pth'

print("Attempting to load LipSync model...")
print(f"Device: cuda")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

start_time = time.time()

try:
    lip = LipSync(
        model='wav2lip',
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda", # Forcing CPU
        nosmooth=True
    )
    end_time = time.time()
    print(f"\n✅ SUCCESS: LipSync model loaded in {end_time - start_time:.2f} seconds.")

except Exception as e:
    print(f"\n❌ FAILED: An error occurred during loading: {e}")


lip.sync(
    'TalkingVideo.mov',
    'generated_speech.wav',
    'result.mp4',
)