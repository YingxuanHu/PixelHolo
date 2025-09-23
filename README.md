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

Info for running the program
----

- The program should be run from a terminal (such as VS Code's integrated terminal) so you can see important debug and processing information.

- The current versions of PixelHolo have complex dependencies. They have already been set up for you, but if you need to recreate the virtual environment for any reoiason, follow the instructions in the requirements.txt file in the same directory as this file. The process of setting up dependencies by avding many conflicts while allowing some has been found to be complicated, so try to avoid recreating the venv if possible.

- Prior to running PixelHolo, enter the Python virtual environment created for this project (by running 'source venv/bin/activate' in the terminal while in the same directory as this file).

- If you run 'pip check', you will see that there are dependency errors, however, these ones are expected as PixelHolo runs fine anyway. 

- When the program starts, a link to a locally hosted web interface will be provided, through which the user will upload a video of a person talking to the camera for at least a minute. The user can optionally also provide text instructions to fine-tune the Ollama AI model that will be used for conversation (Phi-4 is currently being used, but you can of course change this if you wish).

- After clicking the 'Generate' button, you can head back to the terminal to check the initialization process.

- Once initialization is complete, a window will pop up, and if everything went right, it should display a static image of the person on a black background. The program replaces the background with black because when the monitor's screen is put near an acrylic sheet at an angle, the person will appear to float on the acrylic sheet, hence the promise of a 'holographic' avatar.

- At this point, if the conference camera is connected, you should notice it tracking your face. If not, look at the initial debug messages printed in the terminal after you clicked 'Generate' to see what went wrong. The camera currently only serves the purpose of tracking the user's face, but you take it further by implementing something like the ability to differentiate between people if you like.

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

Final Remarks
---
Thank you for contributing to this important project! If you have any questions, feel free to us know.


