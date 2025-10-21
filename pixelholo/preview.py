"""HTML preview generation for rendered videos."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>PixelHolo Preview</title>
    <style>
      body {{
        background: #111;
        color: #f5f5f5;
        font-family: system-ui, sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        margin: 0;
      }}
      h1 {{
        font-size: 1.4rem;
        margin-bottom: 1rem;
      }}
      video {{
        max-width: 90vw;
        max-height: 80vh;
        border-radius: 12px;
        box-shadow: 0 0 24px rgba(0, 0, 0, 0.6);
      }}
      p {{
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #bbb;
      }}
      a {{
        color: #61dafb;
      }}
    </style>
  </head>
  <body>
    <h1>PixelHolo Output Preview</h1>
    <video controls autoplay>
      <source src="{video_src}" type="video/mp4" />
      Your browser does not support the video tag.
    </video>
    <p>File: <a href="{video_src}" download>{video_name}</a></p>
  </body>
</html>
"""


def generate_preview_html(video_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
    """Render a lightweight HTML page that plays the given video.

    Returns the written HTML path, or ``None`` if the video is missing.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return None

    target_dir = video_path.parent
    preview_path = output_path or (config.OUTPUTS_DIR / "preview.html")
    if preview_path.parent != target_dir:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            video_src = video_path.relative_to(preview_path.parent)
        except ValueError:
            video_src = video_path.as_uri()
    else:
        video_src = video_path.name

    html = HTML_TEMPLATE.format(video_src=video_src, video_name=video_path.name)
    preview_path.write_text(html, encoding="utf-8")
    return preview_path
