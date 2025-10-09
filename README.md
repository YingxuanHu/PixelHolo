# Notes for  anyone working on PixelHolo

Overview
---------

- PixelHolo is a program that can turn anyone into an AI-powered holographic avatar that you can converse with! Simply upload a video of someone talking to the camera (at least a minute or two is preferred) and optionally some instructions to fine-tune the personality of your avatar, and PixelHolo will do the rest!

- Uses Chatterbox-TTS for voice cloning.
- Uses the lipsync library for lipsyncing (in one version of the program). Note that this is not allowed to be used in commercial applications and thus a commercially-friendly alternative should  replace this in the future.
- Uses Mediapipe's rembg to remove background from the video and its first frame to isolate the person of interest and place them on a black background.
- Uses Ollama to implement Microsoft's Phi 4 LLM AI model to enable conversation.
- Features a custom-trained ML model to determine if a prompt requires internet consultation. 
- If a question requires internet consultation, then the program will provide some relevant search results as context to the AI model.
- Displays unique icons to notify user which stage of processing the program is currently in.
- Sends signals via PySerial to a 360 degree camera to enable face-tracking, so that the microphone attached on top is always pointed towards the user.

Project layout
---

```
PixelHolo/
├── assets/                 # Static resources (icons, ML models, pretrained weights)
├── docs/                   # Setup notes and other documentation fragments
├── examples/               # Sample media assets for quick testing
├── notes/                  # Dependency snapshots and scratch notes preserved from setup
├── pixelholo/              # Application package (Flask UI, runtime orchestration, utilities)
├── runtime/                # Generated artifacts (uploads, temp files, cached/processed media)
├── scripts/                # Helper scripts for troubleshooting and experiments
├── PixelHolo_Lipsync_and_Voice_Cloning.py      # Entry point with lip-sync enabled
├── PixelHolo_Voice_Cloning_Only.py             # Entry point without lip-sync
├── README.md
└── requirements.txt
```

Info for running the program
----

- The program should be run from a terminal (such as VS Code's integrated terminal) so you can see important debug and processing information.

- The dependency stack has been simplified. If you need to recreate the virtual environment, follow the instructions at the top of `requirements.txt`.

- Prior to running PixelHolo, enter the Python virtual environment created for this project (by running 'source venv/bin/activate' in the terminal while in the same directory as this file).

- When the program starts, a link to a locally hosted web interface will be provided, through which the user will upload a video of a person talking to the camera for at least a minute. The user can optionally also provide text instructions to fine-tune the Ollama AI model that will be used for conversation (Phi-4 is currently being used, but you can of course change this if you wish).
- The uploader UI listens on `http://127.0.0.1:5000` by default. If that port is busy, PixelHolo automatically tries the fallbacks `5001-5003`. To override these values, set `PIXELHOLO_FLASK_HOST`, `PIXELHOLO_FLASK_PORT`, and (optional) `PIXELHOLO_FLASK_PORT_FALLBACKS` before launching the script.

- After clicking the 'Generate' button, you can head back to the terminal to check the initialization process.

- Once initialization is complete, a window will pop up, and if everything went right, it should display a static image of the person on a black background. The program replaces the background with black because when the monitor's screen is put near an acrylic sheet at an angle, the person will appear to float on the acrylic sheet, hence the promise of a 'holographic' avatar.

- At this point, if the conference camera is connected, you should notice it tracking your face. If not, look at the initial debug messages printed in the terminal after you clicked 'Generate' to see what went wrong. The camera currently only serves the purpose of tracking the user's face, but you take it further by implementing something like the ability to differentiate between people if you like.

- Transformers may emit deprecation warnings about attention caches or SDPA; they are non-fatal. If you want to silence them, upgrade `transformers` and set `attn_implementation="eager"` (or disable `output_attentions`) when instantiating models.

- Press the space-bar once to begin recording your prompt (make sure the microphone is connected). You will see a microphone icon pop up in the bottom right corner. You can now say something to the avatar, and simply stop talking when you're done.

- Once the program detects a moment of silence, it will begin processing your prompt. It will display a specific icon in the bottom right depending on what stage of processing it is in. For more information, refer to the terminal, which will detail the process. 

In short, given a prompt, the following is done to generate the avatar's audio/visual response: 
1. Interpret what the user said using Google's speech to text recognition.
2. Generate the avatar's text response using the Ollama AI model.
3. Generate the avatar's vocal speech response using the Chatterbox-TTS model (which was trained on the uploaded video's audio of the person talking).
4. Either generate a video that modifies the uploaded video to make the person's lips match the generated speech, and play it, or simply play/loop the uploaded video along with the generated audio to give the impression that the avatar is talking, depending on which version of the program is run (there are two versions, see next section for more info)

After the avatar is done talking, it will go back to the idle state, at which point the user can provide another prompt.


Info regarding the two versions of PixelHolo
---

- There are two versions of the program, corresponding to two Python programs in this directory, called PixelHolo_Lipsync_and_Voice_Cloning.py and PixelHolo_Voice_Cloning_Only.py

- The former clones the person's voice using the input video's audio (using Chatterbox-TTS) and generates a video that modifies the input video such the person's lips appear to move as if the person was saying the generated speech (using the lipsync library, based on Wav2Lip).

- The latter only clones the person's voice, and simply plays the video when the avatar is to talk.

Environment Setup
---
The shared workstation is provisioned with:

- **CPU**: AMD Ryzen 9 9950X3D (32 threads)
- **GPU**: NVIDIA GeForce RTX 5090 (SM 120)
- **Memory**: 64 GB RAM
- **Storage**: 4 TB NVMe SSD

Ensure the latest NVIDIA driver is installed (`nvidia-smi` should succeed) before continuing. Follow the steps below to create a clean virtual environment tuned for this hardware.

1. **System packages**
   ```bash
   sudo apt update
   sudo apt install -y python3.10 python3.10-venv python3.10-dev \
       libgtk-3-dev pkg-config portaudio19-dev ffmpeg
   # Optional when compiling Python 3.10: sudo apt install liblzma-dev
   ```

2. **Virtual environment**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip wheel
   ```

3. **Install PyTorch (GPU preferred)**
   The RTX 5090 requires the CUDA nightly build (cu128) for full SM 120 support:
   ```bash
   pip install --upgrade --pre --no-cache-dir \
     torch torchaudio torchvision \
     --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
   (If the nightly wheels are temporarily unavailable, fall back to the CUDA 12.1 stable build.)
   ```bash
   pip install --upgrade --no-cache-dir \
     torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 \
     --extra-index-url https://download.pytorch.org/whl/cu121
   ```

   Optional CUDA check:
   ```bash
   python - <<'PY'
   import torch
   print("cuda available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("device:", torch.cuda.get_device_name())
   PY
   ```
   PixelHolo automatically prefers the GPU when `torch.cuda.is_available()` returns `True`; otherwise it will run all workloads on the CPU.

4. **Install Chatterbox-TTS without dragging in conflicting pins**
   ```bash
   pip install chatterbox-tts==0.1.2 --no-deps
   pip install soundfile==0.12.1
   pip install --upgrade transformers==4.45.2 accelerate==1.0.1 einops==0.8.0
   python - <<'PY'
   from importlib.metadata import distribution
   meta = distribution("chatterbox-tts").locate_file("chatterbox_tts-0.1.2.dist-info/METADATA")
   text = meta.read_text()
   text = text.replace("Requires-Dist: torch==2.6.0", "Requires-Dist: torch>=2.10.0.dev20251008")
   text = text.replace("Requires-Dist: torchaudio==2.6.0", "Requires-Dist: torchaudio>=2.8.0.dev20251008")
   text = text.replace("Requires-Dist: librosa==0.11.0", "Requires-Dist: librosa>=0.10.2.post1")
   text = text.replace("Requires-Dist: transformers==4.46.3", "Requires-Dist: transformers>=4.45.2")
   meta.write_text(text)
   PY
   ```
   The runtime now saves audio via `soundfile`, so TorchCodec/FFmpeg compatibility issues are avoided.

5. **Install PixelHolo dependencies**
   ```bash
   pip install -r requirements.txt
   pip check
   ```
   *(Only run this after completing Steps 3–4 so the CUDA 12.8 nightly `torch/torchaudio/torchvision` and patched Chatterbox metadata are already in place. If `pip` complains that `torch>=2.10.0.dev...` is missing, repeat Step 3 to reinstall the nightly build.)*

Ollama Service
---
PixelHolo talks to Ollama at `http://127.0.0.1:11434`.

- Auto-start: the app will try to start Ollama for you if it isn’t running.
- Manual fallback (if auto-start fails or you prefer a service):
  ```bash
  # start the server
  ollama serve    # or: systemctl --user start ollama

  # pull & warm a model
  ollama pull llama3.1
  ollama run llama3.1 "hello"

  # health check
  curl http://127.0.0.1:11434/api/version
  ```

Troubleshooting
- Ensure `ollama` is on PATH: `which ollama || echo "Ollama not on PATH"`
- If running as a managed service, use your system’s service manager instead of `ollama serve`.
- To point PixelHolo at a non-default host/port, edit `pixelholo/config.py` (`OLLAMA_API_URL`).

Final Remarks
---
Thank you for contributing to this important project! If you have any questions, feel free to us know.
