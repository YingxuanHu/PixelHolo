"""Entry point for the PixelHolo interactive clone application."""

import argparse
from pixelholo.app import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PixelHolo - Interactive Voice Clone with Lip-sync")
    parser.add_argument(
        "--enable-lipsync",
        action="store_true",
        default=True,
        help="Enable lip-sync video generation (default: True)"
    )
    parser.add_argument(
        "--no-lipsync",
        action="store_true",
        help="Disable lip-sync video generation (audio only)"
    )
    parser.add_argument(
        "--enable-timing",
        action="store_true",
        help="Enable performance timing reports (shows execution time for each operation)"
    )

    args = parser.parse_args()

    # Handle lipsync flag
    enable_lipsync = args.enable_lipsync and not args.no_lipsync

    main(enable_lipsync=enable_lipsync, enable_timing=args.enable_timing)
