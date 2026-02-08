#!/usr/bin/env python3
"""Wrestling Frame Bot for Bluesky.

Posts a random compelling frame from a wrestling PPV or TV episode (Raw/SmackDown)
in Plex to Bluesky, with the title and timestamp.
"""

import argparse
import base64
import io
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import anthropic

import cv2
import numpy as np
from atproto import Client as BskyClient
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from plexapi.server import PlexServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_PATH = Path(__file__).parent / "config.env"
load_dotenv(ENV_PATH)

PLEX_URL = os.environ.get("PLEX_URL", "")
PLEX_TOKEN = os.environ.get("PLEX_TOKEN", "")
PLEX_PPV_LIBRARY = os.environ.get("PLEX_PPV_LIBRARY", "Wrestling (PPVs)")
PLEX_TV_LIBRARY = os.environ.get("PLEX_TV_LIBRARY", "Wrestling (TV)")
TV_SHOW_FILTERS = [f.strip() for f in os.environ.get("TV_SHOW_FILTERS", "Raw,SmackDown").split(",") if f.strip()]
PPV_WEIGHT = float(os.environ.get("PPV_WEIGHT", "0.25"))
BSKY_HANDLE = os.environ.get("BSKY_HANDLE", "")
BSKY_PASSWORD = os.environ.get("BSKY_PASSWORD", "")
HUGO_SITE_PATH = os.environ.get("HUGO_SITE_PATH", "")

HUGO_SITE_PUSH = os.environ.get("HUGO_SITE_PUSH", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Plex helpers
# ---------------------------------------------------------------------------


def connect_plex():
    """Connect to Plex and return the server object."""
    return PlexServer(PLEX_URL, PLEX_TOKEN)


def pick_random_ppv(plex):
    """Pick a random item from the wrestling library."""
    library = plex.library.section(PLEX_PPV_LIBRARY)
    items = library.all()
    if not items:
        raise RuntimeError(f"No items found in Plex library '{PLEX_PPV_LIBRARY}'")
    return random.choice(items)


def pick_random_tv_episode(plex):
    """Pick a random episode from filtered TV shows (Raw/SmackDown)."""
    library = plex.library.section(PLEX_TV_LIBRARY)
    shows = library.all()
    matching = [
        s for s in shows
        if any(f.lower() in s.title.lower() for f in TV_SHOW_FILTERS)
    ]
    if not matching:
        raise RuntimeError(
            f"No shows matching {TV_SHOW_FILTERS} in '{PLEX_TV_LIBRARY}'"
        )
    # Pool all episodes across matching shows to avoid bias
    all_episodes = []
    for show in matching:
        all_episodes.extend(show.episodes())
    if not all_episodes:
        raise RuntimeError("Matching shows have no episodes")
    return random.choice(all_episodes)


def pick_random_item(plex):
    """Pick a random PPV or TV episode. Returns (item, source).

    PPV_WEIGHT controls the split: 0 = TV only, 1 = PPV only, 0-1 = mix.
    """
    if PPV_WEIGHT >= 1:
        return pick_random_ppv(plex), "ppv"
    if PPV_WEIGHT <= 0:
        return pick_random_tv_episode(plex), "tv"
    if random.random() < PPV_WEIGHT:
        return pick_random_ppv(plex), "ppv"
    return pick_random_tv_episode(plex), "tv"


def format_item_title(item, source):
    """Format a display title for a PPV or TV episode."""
    if source == "ppv":
        title = item.title
        year = getattr(item, "year", None) or getattr(
            getattr(item, "originallyAvailableAt", None), "year", None
        )
        if year:
            title = f"{title} ({year})"
        return title
    # TV episode: "WWE Raw (January 6, 2025)"
    show_name = getattr(item, "grandparentTitle", None) or "Unknown Show"
    air_date = getattr(item, "originallyAvailableAt", None)
    if air_date:
        return f"{show_name} ({air_date.strftime('%B %-d, %Y')})"
    # Fallback: "WWE Raw - S32E05 (2025)"
    season = getattr(item, "parentIndex", None)
    episode = getattr(item, "index", None)
    year = getattr(item, "year", None)
    if season is not None and episode is not None:
        tag = f"S{season:02d}E{episode:02d}"
        if year:
            return f"{show_name} - {tag} ({year})"
        return f"{show_name} - {tag}"
    return show_name


def get_stream_url(item):
    """Build a direct stream URL for ffmpeg from a Plex media item."""
    part = item.media[0].parts[0]
    base = PLEX_URL.rstrip("/")
    return f"{base}{part.key}?X-Plex-Token={PLEX_TOKEN}"


def get_duration_seconds(item):
    """Return the item's duration in seconds."""
    return item.duration / 1000.0


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frame(stream_url, timestamp_secs, output_path):
    """Extract a single frame at the given timestamp using ffmpeg (JPEG).

    Returns True if extraction succeeded, False otherwise.
    """
    ts_str = format_timestamp_ffmpeg(timestamp_secs)
    cmd = [
        "ffmpeg", "-y",
        "-ss", ts_str,
        "-i", stream_url,
        "-vsync", "0",
        "-vf", "yadif=deint=interlaced",
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0


def extract_frame_png(stream_url, timestamp_secs, output_path):
    """Extract a single frame as lossless PNG at the given timestamp.

    Returns True if extraction succeeded, False otherwise.
    """
    ts_str = format_timestamp_ffmpeg(timestamp_secs)
    cmd = [
        "ffmpeg", "-y",
        "-ss", ts_str,
        "-i", stream_url,
        "-vsync", "0",
        "-vf", "yadif=deint=interlaced",
        "-frames:v", "1",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0


def format_timestamp_ffmpeg(seconds):
    """Format seconds as HH:MM:SS.mmm for ffmpeg."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def format_timestamp_display(seconds):
    """Format seconds as H:MM:SS for display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def is_dead_frame(image_path):
    """Check if a frame is nearly all black or all white (fade transition)."""
    img = Image.open(image_path).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float64)
    avg = arr.mean()
    return avg < 20 or avg > 240


def compute_sharpness(image_path):
    """Return the Laplacian variance of an image (higher = sharper)."""
    img = Image.open(image_path).convert("L")
    laplacian = ImageFilter.Kernel(
        size=(3, 3),
        kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
        scale=1, offset=128,
    )
    edges = img.filter(laplacian)
    arr = np.array(edges, dtype=np.float64)
    return float(np.var(arr - 128.0))


def is_blurry(image_path, threshold=150.0):
    """Check if a frame is blurry using Laplacian variance.

    Low variance of the Laplacian = few edges = blurry/motion-blurred.
    """
    return compute_sharpness(image_path) < threshold


def _deduplicate_timestamps(timestamps, min_gap=10.0):
    """Remove timestamps within min_gap seconds of each other."""
    if not timestamps:
        return []
    timestamps.sort()
    deduped = [timestamps[0]]
    for ts in timestamps[1:]:
        if ts - deduped[-1] >= min_gap:
            deduped.append(ts)
    return deduped


def detect_scene_changes(stream_url, duration_secs, num_segments=8, segment_duration=60,
                         scene_threshold=0.3, pre_cut_offset=1.5):
    """Detect scene changes by sampling segments of the video with ffmpeg.

    Samples num_segments evenly-spaced windows of segment_duration seconds each,
    runs scene change detection on each, and returns timestamps offset back by
    pre_cut_offset seconds (to grab the "money shot" before the cut).

    Returns a list of timestamps (floats) clamped to [5, duration-5].
    """
    margin = 5.0
    usable = duration_secs - 2 * margin
    if usable <= 0:
        return []

    # Evenly space segment start points across the video
    if num_segments == 1:
        starts = [margin]
    else:
        step = max(usable - segment_duration, 0) / (num_segments - 1)
        starts = [margin + i * step for i in range(num_segments)]

    timestamps = []
    for seg_start in starts:
        seg_start = min(seg_start, duration_secs - margin - segment_duration)
        if seg_start < margin:
            seg_start = margin

        cmd = [
            "ffmpeg",
            "-ss", format_timestamp_ffmpeg(seg_start),
            "-i", stream_url,
            "-t", str(segment_duration),
            "-an",
            "-vf", f"scale=320:-1,select='gt(scene\\,{scene_threshold})',showinfo",
            "-f", "null", "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            stderr = result.stderr
        except subprocess.TimeoutExpired:
            print(f"  Scene detection timed out for segment at {format_timestamp_display(seg_start)}", file=sys.stderr)
            continue

        # Parse pts_time from showinfo output lines
        for m in re.finditer(r"pts_time:(\d+\.?\d*)", stderr):
            pts = float(m.group(1))
            # pts is relative to segment start; convert to absolute and offset back
            abs_ts = seg_start + pts - pre_cut_offset
            abs_ts = max(margin, min(abs_ts, duration_secs - margin))
            timestamps.append(abs_ts)

    timestamps = _deduplicate_timestamps(timestamps)
    print(f"  Scene detection found {len(timestamps)} candidate timestamps across {num_segments} segments", file=sys.stderr)
    return timestamps


# ---------------------------------------------------------------------------
# Stage 1: Candidate generation
# ---------------------------------------------------------------------------


def generate_candidates(stream_url, duration_secs, tmpdir, target=10, min_viable=5):
    """Generate candidate frames using scene detection + random fallback.

    Phase 1: Use scene change detection to find interesting timestamps,
             extract frames and reject dead/blurry ones.
    Phase 2: If Phase 1 didn't yield enough, fill remaining slots randomly.

    Returns list of (timestamp_secs, image_path) tuples.
    """
    candidates = []
    used_timestamps = []
    frame_idx = 0

    def _try_extract(ts):
        """Extract a frame at ts, rejecting dead/blurry. Returns True if kept."""
        nonlocal frame_idx
        out_path = os.path.join(tmpdir, f"candidate_{frame_idx:03d}.png")
        frame_idx += 1

        if not extract_frame_png(stream_url, ts, out_path):
            return False

        if is_dead_frame(out_path):
            os.remove(out_path)
            return False

        if is_blurry(out_path):
            os.remove(out_path)
            return False

        candidates.append((ts, out_path))
        used_timestamps.append(ts)
        return True

    # Phase 1: Scene detection candidates
    print("  Phase 1: Scene change detection...", file=sys.stderr)
    scene_timestamps = detect_scene_changes(stream_url, duration_secs)

    if scene_timestamps:
        random.shuffle(scene_timestamps)
        pool = scene_timestamps[:target * 2]
        for ts in pool:
            if len(candidates) >= target:
                break
            _try_extract(ts)
        print(f"  Phase 1: {len(candidates)} candidates from scene detection", file=sys.stderr)

    # Phase 2: Random fallback for remaining slots
    remaining = target - len(candidates)
    if remaining > 0:
        print(f"  Phase 2: Filling {remaining} remaining slots randomly...", file=sys.stderr)
        attempts = 0
        max_attempts = remaining * 3

        while len(candidates) < target and attempts < max_attempts:
            ts = random.uniform(5, duration_secs - 5)
            attempts += 1

            # Skip if too close to an already-selected timestamp
            if any(abs(ts - used) < 10 for used in used_timestamps):
                continue

            _try_extract(ts)

        print(f"  Phase 2: {len(candidates)} total candidates now", file=sys.stderr)

    if len(candidates) < min_viable:
        print(f"Warning: only {len(candidates)} viable candidates (wanted {min_viable})", file=sys.stderr)

    return candidates


# ---------------------------------------------------------------------------
# Stage 2: Visual scoring
# ---------------------------------------------------------------------------


def score_brightness_variance(img_array):
    """Score based on brightness variance — higher = more interesting lighting."""
    gray = np.mean(img_array, axis=2)
    return float(np.var(gray))


def score_edge_density(pil_image):
    """Score based on edge density — more edges = more action/detail."""
    gray = pil_image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges, dtype=np.float64)
    return float(arr.mean())


def score_color_diversity(pil_image):
    """Score based on distinct hue buckets in HSV — colorful > monochrome."""
    hsv = pil_image.convert("HSV")
    arr = np.array(hsv)
    hues = arr[:, :, 0].flatten()
    saturations = arr[:, :, 1].flatten()
    # Only count hues where saturation is meaningful
    meaningful = hues[saturations > 30]
    if len(meaningful) == 0:
        return 0.0
    # Count hue buckets (divide 0-255 range into 24 buckets)
    buckets = np.unique(meaningful // 11)
    return float(len(buckets))


def score_center_weight(pil_image):
    """Score ratio of edge energy in center 50% crop vs full frame.
    Close-ups with centered subjects score higher than uniform wide shots."""
    gray = pil_image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges, dtype=np.float64)
    h, w = arr.shape
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4
    full_mean = arr.mean()
    if full_mean < 1.0:
        return 0.0
    center_mean = arr[y1:y2, x1:x2].mean()
    return float(center_mean / full_mean)


def score_face_presence(pil_image):
    """Score based on detected face presence and size — larger faces = close-ups."""
    arr = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) == 0:
        return 0.0
    h = gray.shape[0]
    largest = max(faces, key=lambda f: f[2] * f[3])
    size_ratio = largest[3] / h  # face height vs frame height
    return 0.5 + size_ratio * 2.0  # base boost + size bonus


def score_frame(image_path):
    """Compute a composite visual interest score for a frame."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float64)

    bv = score_brightness_variance(arr)
    ed = score_edge_density(img)
    cd = score_color_diversity(img)
    sh = compute_sharpness(image_path)
    cw = score_center_weight(img)
    fc = score_face_presence(img)

    # Normalize roughly to similar scales and combine
    # Brightness variance: typically 0-10000+, divide by 2000
    # Edge density: typically 0-80, divide by 40 (halved to reduce wide-shot bias)
    # Color diversity: typically 0-24, divide by 8
    # Sharpness: typically 0-2000+, divide by 500
    # Center weight: typically 0.8-1.8, * 0.5 (boosts center-heavy close-ups)
    # Face presence: 0.0 (none) to ~2.0+ (large close-up), * 0.6
    score = (bv / 2000.0) + (ed / 40.0) + (cd / 8.0) + (sh / 500.0) + (cw * 0.5) + (fc * 0.6)
    return score, {"brightness_var": bv, "edge_density": ed, "color_diversity": cd, "sharpness": sh, "center_weight": cw, "face_score": fc}


def rank_candidates(candidates, top_n=3):
    """Score and rank candidates, returning the top N."""
    scored = []
    for ts, path in candidates:
        score, details = score_frame(path)
        scored.append((score, ts, path, details))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Stage 3: Sharpness refinement
# ---------------------------------------------------------------------------


def refine_frame_sharpness(top_frames, stream_url, duration_secs, tmpdir,
                           window=1.0, step=0.05):
    """Refine each top candidate by seeking nearby sharper frames.

    For each candidate, extracts frames in a +/- window around the original
    timestamp at the given step interval, picks the sharpest one, then
    re-ranks all refined candidates using the full composite score.

    Returns a new top_frames list (same format as rank_candidates output).
    """
    margin = 5.0
    refined = []

    for rank, (orig_score, orig_ts, orig_path, orig_details) in enumerate(top_frames):
        best_sharpness = compute_sharpness(orig_path)
        best_ts = orig_ts
        best_path = orig_path
        print(f"  Candidate #{rank+1}: original sharpness={best_sharpness:.1f} "
              f"at ts={format_timestamp_display(orig_ts)}", file=sys.stderr)

        # Generate offsets around the original timestamp
        offsets = []
        t = -window
        while t <= window + step / 2:
            if abs(t) > step / 2:  # skip offset=0, we already have the original
                offsets.append(t)
            t += step

        checked = 0
        for offset in offsets:
            probe_ts = orig_ts + offset
            if probe_ts < margin or probe_ts > duration_secs - margin:
                continue

            probe_path = os.path.join(tmpdir, f"refine_{rank}_{checked:03d}.png")
            if not extract_frame_png(stream_url, probe_ts, probe_path):
                continue
            checked += 1

            if is_dead_frame(probe_path):
                os.remove(probe_path)
                continue

            sh = compute_sharpness(probe_path)
            if sh > best_sharpness:
                best_sharpness = sh
                best_ts = probe_ts
                best_path = probe_path

        print(f"    -> best sharpness={best_sharpness:.1f} at ts={format_timestamp_display(best_ts)} "
              f"(checked {checked} nearby frames)", file=sys.stderr)
        refined.append((best_ts, best_path))

    # Re-rank refined candidates using full composite score
    re_scored = []
    for ts, path in refined:
        score, details = score_frame(path)
        re_scored.append((score, ts, path, details))

    re_scored.sort(key=lambda x: x[0], reverse=True)
    return re_scored


# ---------------------------------------------------------------------------
# Dry-run review composite
# ---------------------------------------------------------------------------


def build_review_composite(winner_path, runner_up_entries, post_text, winner_score):
    """Build a single review image: winner on top, post text, runner-ups below."""
    pad = 20
    winner_w = 800
    runner_w = 400
    bg_color = (30, 30, 30)
    text_color = (255, 255, 255)
    label_color = (0, 200, 80)
    font = ImageFont.load_default()

    # --- Winner ---
    winner_img = Image.open(winner_path).convert("RGB")
    ratio = winner_w / winner_img.width
    winner_img = winner_img.resize(
        (winner_w, int(winner_img.height * ratio)), Image.LANCZOS
    )
    # Draw label on winner
    draw_w = ImageDraw.Draw(winner_img)
    label = f"WINNER  score={winner_score:.2f}"
    draw_w.text((10, 10), label, fill=label_color, font=font)

    # --- Runner-ups ---
    runner_imgs = []
    for i, (score, ts, path, _) in enumerate(runner_up_entries):
        img = Image.open(path).convert("RGB")
        r = runner_w / img.width
        img = img.resize((runner_w, int(img.height * r)), Image.LANCZOS)
        draw_r = ImageDraw.Draw(img)
        draw_r.text(
            (10, 10), f"#{i+2}  score={score:.2f}", fill=text_color, font=font
        )
        runner_imgs.append(img)

    # --- Measure post text ---
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_bbox = dummy.multiline_textbbox((0, 0), post_text, font=font)
    text_h = text_bbox[3] - text_bbox[1] + pad

    # --- Layout ---
    runners_row_w = sum(im.width for im in runner_imgs) + pad * max(len(runner_imgs) - 1, 0)
    runners_row_h = max((im.height for im in runner_imgs), default=0)

    total_w = max(winner_w, runners_row_w) + pad * 2
    total_h = pad + winner_img.height + pad + text_h + pad + runners_row_h + pad

    composite = Image.new("RGB", (total_w, total_h), bg_color)

    # Place winner (centered)
    x_win = (total_w - winner_img.width) // 2
    y = pad
    composite.paste(winner_img, (x_win, y))
    y += winner_img.height + pad

    # Draw post text
    draw_c = ImageDraw.Draw(composite)
    draw_c.multiline_text((pad, y), post_text, fill=text_color, font=font)
    y += text_h + pad

    # Place runner-ups (centered row)
    x = (total_w - runners_row_w) // 2
    for img in runner_imgs:
        composite.paste(img, (x, y))
        x += img.width + pad

    return composite


# ---------------------------------------------------------------------------
# Image prep for Bluesky
# ---------------------------------------------------------------------------


def prepare_image_for_upload(image_path, max_bytes=976_000):
    """Ensure image is under Bluesky's ~1MB limit. Returns image bytes.

    Strategy: cap width at 1920px, then binary-search for the highest JPEG
    quality (60-92) that fits. Only downscale resolution as a last resort.
    Never goes below quality 60 to avoid visible blocking artifacts.
    """
    img = Image.open(image_path).convert("RGB")

    # Cap width at 1920px, preserving aspect ratio
    max_width = 1920
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)

    def _encode(image, quality):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    # Binary search for highest quality that fits
    lo, hi = 60, 92
    best = _encode(img, lo)  # fallback: lowest acceptable quality

    while lo <= hi:
        mid = (lo + hi) // 2
        data = _encode(img, mid)
        if len(data) <= max_bytes:
            best = data
            lo = mid + 1  # try higher quality
        else:
            hi = mid - 1  # need lower quality

    if len(best) <= max_bytes:
        return best

    # Last resort: downscale resolution in steps until it fits at quality 60
    for scale in [0.85, 0.7, 0.55]:
        scaled = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        data = _encode(scaled, 60)
        if len(data) <= max_bytes:
            return data

    # Final fallback
    img.thumbnail((1280, 720), Image.LANCZOS)
    return _encode(img, 60)


# ---------------------------------------------------------------------------
# Alt text generation
# ---------------------------------------------------------------------------


def generate_alt_text(image_bytes, item_title, timestamp_secs):
    """Use Claude Haiku vision to generate descriptive alt text for a frame."""
    ts_display = format_timestamp_display(timestamp_secs)
    prefix = f"A frame from {item_title} at {ts_display}."
    try:
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"This is a frame from the wrestling event \"{item_title}\". "
                                "Write a concise visual description of this image "
                                "(1-2 sentences, under 300 characters). Describe what is visible: "
                                "wrestlers, action, crowd, ring, graphics, etc. "
                                "Do not start with \"This image shows\" or similar preamble."
                            ),
                        },
                    ],
                }
            ],
        )
        description = response.content[0].text.strip()
        alt_text = f"{prefix} {description}"
        # Bluesky alt text limit is ~1000 chars; keep it concise
        if len(alt_text) > 500:
            alt_text = alt_text[:497] + "..."
        return alt_text
    except Exception as e:
        print(f"  Alt text generation failed: {e}", file=sys.stderr)
        return prefix


# ---------------------------------------------------------------------------
# Hugo site integration
# ---------------------------------------------------------------------------


def save_to_hugo_site(item_title, source, timestamp_secs, image_bytes, alt_text, push=True):
    """Save a posted frame to the Hugo gallery site as a page bundle.

    Skips entirely if HUGO_SITE_PATH is not set.
    """
    if not HUGO_SITE_PATH:
        return None

    site_path = Path(HUGO_SITE_PATH)
    if not site_path.is_dir():
        print(f"  Hugo site path not found: {site_path}", file=sys.stderr)
        return None

    from datetime import date

    today = date.today()
    ts_display = format_timestamp_display(timestamp_secs)

    # Build slug: {date}-{slugified event}-{timestamp}
    event_slug = re.sub(r'[^a-z0-9]+', '-', item_title.lower()).strip('-')
    ts_slug = ts_display.replace(':', '-')
    slug = f"{today.isoformat()}-{event_slug}-{ts_slug}"

    bundle_dir = site_path / "content" / "frames" / slug
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Write frame image
    (bundle_dir / "frame.jpg").write_bytes(image_bytes)

    # Extract show name: strip trailing parenthetical
    show_name = re.sub(r'\s*\(.*\)$', '', item_title)

    # Build frontmatter
    year = str(today.year)
    frontmatter = (
        f"---\n"
        f'title: "{item_title}"\n'
        f"date: {today.isoformat()}T12:00:00Z\n"
        f'timestamp: "{ts_display}"\n'
        f'source: "{source}"\n'
        f"shows:\n"
        f'  - "{show_name}"\n'
        f"years:\n"
        f'  - "{year}"\n'
        f"episodes:\n"
        f'  - "{item_title}"\n'
        f"---\n"
        f"{alt_text}\n"
    )
    (bundle_dir / "index.md").write_text(frontmatter)

    # Git add, commit, optionally push
    git_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    git_args = {"cwd": site_path, "capture_output": True, "text": True, "env": git_env}

    subprocess.run(["git", "pull", "--rebase"], **git_args)
    subprocess.run(["git", "add", str(bundle_dir.relative_to(site_path))], **git_args)
    commit_msg = f"Add frame: {item_title} ({ts_display})"
    subprocess.run(["git", "commit", "-m", commit_msg], **git_args)

    if push:
        result = subprocess.run(["git", "push"], **git_args)
        if result.returncode != 0:
            print(f"  Git push failed: {result.stderr}", file=sys.stderr)
    else:
        print(f"  [DRY RUN] Committed but skipped push", file=sys.stderr)

    print(f"  Hugo page bundle: {bundle_dir}")
    return bundle_dir


# ---------------------------------------------------------------------------
# Bluesky posting
# ---------------------------------------------------------------------------


def post_to_bluesky(item_title, timestamp_secs, image_bytes, image_alt=None):
    """Post the frame to Bluesky with the item title and timestamp."""
    client = BskyClient()
    client.login(BSKY_HANDLE, BSKY_PASSWORD)

    ts_display = format_timestamp_display(timestamp_secs)
    text = f"{item_title}\n\n— {ts_display}"

    if image_alt is None:
        image_alt = f"A frame from {item_title} at {ts_display}"

    client.send_image(
        text=text,
        image=image_bytes,
        image_alt=image_alt,
    )
    print(f"Posted: {text}")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def run_bot(dry_run=False):
    """Main bot pipeline: pick PPV/episode -> extract frames -> score -> post top frame."""
    print("Connecting to Plex...")
    plex = connect_plex()

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        print("Picking a random item...")
        item, source = pick_random_item(plex)
        item_title = format_item_title(item, source)
        duration = get_duration_seconds(item)
        print(f"Selected: {item_title} ({format_timestamp_display(duration)} long)")

        stream_url = get_stream_url(item)

        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name

        # Stage 1: Generate candidates
        print("Stage 1: Generating candidate frames...")
        candidates = generate_candidates(stream_url, duration, tmpdir)
        if candidates:
            break
        print(f"  No viable candidates (attempt {attempt}/{MAX_RETRIES}), retrying with a new video...",
              file=sys.stderr)
        tmpdir_obj.cleanup()
    else:
        raise RuntimeError(f"No viable candidate frames found after {MAX_RETRIES} attempts")

    with tmpdir_obj:
        print(f"  {len(candidates)} viable candidates")

        # Stage 2: Visual scoring
        print("Stage 2: Scoring frames visually...")
        top_frames = rank_candidates(candidates, top_n=3)
        for i, (score, ts, path, details) in enumerate(top_frames):
            print(f"  #{i+1}: score={score:.2f} ts={format_timestamp_display(ts)} {details}")

        # Stage 3: Sharpness refinement
        print("Stage 3: Refining sharpness on top candidates...")
        top_frames = refine_frame_sharpness(top_frames, stream_url, duration, tmpdir)
        for i, (score, ts, path, details) in enumerate(top_frames):
            print(f"  #{i+1}: score={score:.2f} ts={format_timestamp_display(ts)} {details}")

        # Pick the top-scored frame
        chosen_score, chosen_ts, chosen_path, _ = top_frames[0]
        print(f"  Winner: #{1} (ts={format_timestamp_display(chosen_ts)})")

        # Prep image
        image_bytes = prepare_image_for_upload(chosen_path)
        print(f"Image size: {len(image_bytes) / 1024:.0f}KB")

        # Generate alt text
        print("Generating alt text...")
        alt_text = generate_alt_text(image_bytes, item_title, chosen_ts)
        print(f"  Alt text: {alt_text}")

        # Post
        if dry_run:
            out_dir = Path(__file__).parent / "dry_run_output"
            out_dir.mkdir(exist_ok=True)
            frame_dest = out_dir / "frame.jpg"
            frame_dest.write_bytes(image_bytes)

            for i, (_, _, runner_path, _) in enumerate(top_frames[1:], start=1):
                runner_bytes = prepare_image_for_upload(runner_path)
                (out_dir / f"runner_up_{i}.jpg").write_bytes(runner_bytes)

            ts_display = format_timestamp_display(chosen_ts)
            post_text = f"{item_title}\n{ts_display}"
            (out_dir / "post.txt").write_text(post_text)
            (out_dir / "alt_text.txt").write_text(alt_text)

            composite = build_review_composite(
                chosen_path, top_frames[1:], post_text, chosen_score
            )
            review_path = out_dir / "review.jpg"
            composite.save(review_path, "JPEG", quality=90)
            subprocess.run(["open", str(review_path)])

            print(f"[DRY RUN] Saved to {out_dir}/")
            print(f"  Frame: {frame_dest}")
            print(f"  Runner-ups: {len(top_frames) - 1}")
            print(f"  Review composite: {review_path}")
            print(f"  Post text:\n{post_text}")
            print(f"  Alt text: {alt_text}")

            # Save to Hugo site (commit only, no push)
            save_to_hugo_site(item_title, source, chosen_ts, image_bytes, alt_text, push=False)

        else:
            print("Posting to Bluesky...")
            post_to_bluesky(item_title, chosen_ts, image_bytes, image_alt=alt_text)

            save_to_hugo_site(item_title, source, chosen_ts, image_bytes, alt_text, push=HUGO_SITE_PUSH)


    print("Done!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Wrestling PPV Frame Bot for Bluesky")
    parser.add_argument("--dry-run", action="store_true", help="Run the pipeline but don't post to Bluesky")
    args = parser.parse_args()

    run_bot(dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
