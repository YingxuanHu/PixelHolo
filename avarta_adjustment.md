# Personality (LLM)

## Extra instructions
free text (up to 2000 chars) appended to the system prompt so Ollama stays on a role or style.

## Creativity (temperature)
Ollama sampling temperature (0.1–1.5), same idea as “how wild” the text is.

# Voice (TTS) — mapped to Chatterbox’s generate() knobs:

## Expressiveness (exaggeration, 0.25–1)
Chatterbox wires this into the emotion / delivery conditioning for the speech model (internally it scales an “emotion advance” signal).
Higher → delivery tends to sound more animated or emphasized.
Lower → flatter, more neutral speech (still your cloned voice, just less “acted”).

## Speech variety (temperature, 0.4–1.2)
This is randomness when the model picks the next speech tokens (like “temperature” in text LLMs).
Higher → more variation between runs and phrases—sometimes looser or less predictable.
Lower → steadier, more consistent phrasing and prosody chunk-to-chunk.

## CFG weight (cfg_weight, 0–1)
Classifier-free guidance: the model compares a guided vs unguided path and blends them.
Higher → stronger pull toward the conditioning (voice + text), often clearer / tighter adherence, sometimes at the cost of sounding stiffer or less natural if pushed too far.
Lower (toward 0) → less of that contrastive steering—can sound freer but risk drifting from what you want.

(When cfg_weight is 0, Chatterbox skips the CFG doubling path for text tokens.)

## Repetition penalty (repetition_penalty, 1–1.5)
Penalizes reusing the same speech tokens too often during generation.
Higher → less stuttering / looping sounds, but if too high can sometimes make delivery less stable.
Lower (near 1) → weaker penalty; the model may repeat syllables or patterns more easily.


## Short cheat sheet

|Control | Think of it as… |
|--|--|
|Expressiveness | How “dramatic” or flat the delivery is |
|Speech variety | How random vs predictable each phrase is |
|CFG weight | How hard to stick to conditioning vs drift|
|Repetition penalty | How much to avoid repeating the same sounds|

If you tell me whether you care more about naturalness vs consistency, we can suggest starting ranges—but the UI sliders are there so you can tune by ear.

# Video look & pace

## Playback speed
playbackRate on both stacked <video> elements (lip-sync chunks stay in sync because audio is baked into the file).

## Brightness / contrast / saturation
CSS filter on #videoContainer (poster + video), no re-encode.