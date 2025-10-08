"""Entry point for the voice-only PixelHolo experience."""

from pixelholo.app import main


if __name__ == "__main__":
    main(enable_lipsync=False)
