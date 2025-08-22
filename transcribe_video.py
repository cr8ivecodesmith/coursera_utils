import os
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import Dict, List, Optional, Tuple

import json
import re
from datetime import datetime, timezone
from tempfile import gettempdir

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import make_chunks


def load_client() -> OpenAI:
    """Initialize OpenAI client using environment variables.

    Looks for `OPENAI_API_KEY`. Supports loading from a local `.env`.
    """
    # Load from a local .env if present
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found in environment. Set it or add to .env"
        )
    return OpenAI(api_key=api_key)


def find_video_files(target: Path, recursive: bool = False) -> List[Path]:
    """Return a flat list of `.mp4` files for the given target.

    - If `target` is a file, validates extension and returns [target].
    - If `target` is a directory, returns only top-level `.mp4` files.
    - Subfolders are NOT traversed.
    """
    if target.is_file():
        if target.suffix.lower() != ".mp4":
            raise ValueError("Only .mp4 files are supported")
        return [target]

    if not target.exists():
        raise FileNotFoundError(f"Target not found: {target}")

    if not target.is_dir():
        raise ValueError(f"Target must be a file or directory: {target}")

    if recursive:
        return sorted([p for p in target.rglob("*.mp4") if p.is_file()])
    else:
        files = [p for p in sorted(target.iterdir()) if p.is_file() and p.suffix.lower() == ".mp4"]
        return files


def default_names_cache_path(target_root: Path) -> Path:
    """Return a default path to store names cache (editable by user).

    Preference order:
    - If target_root is a directory, store a hidden file under it.
    - If it's a file, store under its parent.
    - Fallback to a temp dir file keyed by root name.
    """
    base = target_root if target_root.is_dir() else target_root.parent
    if base.exists() and base.is_dir():
        return base.joinpath(".transcribe_video_names.json")
    # Fallback to temp with a deterministic filename
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(target_root))
    return Path(gettempdir()).joinpath(f"transcribe_video_names_{safe}.json")


def load_names_cache(cache_path: Path) -> Dict[str, str]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        names = data.get("names", {})
        if isinstance(names, dict):
            return {str(Path(k)): str(v) for k, v in names.items()}
    except Exception:
        return {}
    return {}


def save_names_cache(cache_path: Path, root: Path, names: Dict[Path, str], meta: Optional[Dict] = None) -> None:
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "names": {str(Path(k)): v for k, v in names.items()},
    }
    if meta:
        payload["meta"] = meta
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _clean_segment(text: str) -> str:
    """Normalize a path segment for use in a smart name."""
    s = text
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    # drop extension if present
    s = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", s)
    # remove leading ordering like "01 - ", "1.", "m01 -"
    s = re.sub(r"^(?i:m\d+|\d{1,3})(?:\s*[-–.:]|\))\s*", "", s)
    return s


def heuristic_smart_name(video_path: Path, root: Path) -> str:
    """Build a smart base name from directory structure and file stem.

    Uses up to the last two directories plus cleaned file stem.
    """
    try:
        rel = video_path.relative_to(root)
    except Exception:
        rel = video_path.name
    parts = list(rel.parts) if isinstance(rel, Path) else [rel]
    if parts and parts[-1] == video_path.name:
        parts = parts[:-1]
    segments = [_clean_segment(p) for p in parts][-2:]  # last two folders
    stem = _clean_segment(video_path.stem)
    pieces = [p for p in segments + [stem] if p]
    # Ensure not too long
    base = " - ".join(pieces)
    return base[:120] if len(base) > 120 else base


def ai_smart_name(client: OpenAI, video_path: Path, root: Path) -> Optional[str]:
    """Attempt to generate a concise descriptive name using OpenAI.

    Falls back to None on any error.
    """
    prompt = (
        "Generate a concise, file-name-safe, human-friendly title (<= 80 chars) "
        "for a course video based only on its directory path and file name. "
        "Use important folder names (e.g., module/week/section) and the file stem. "
        "Avoid quotes; avoid slashes; return only the title.\n\n"
        f"Root: {root}\n"
        f"Path: {video_path}\n"
    )
    try:
        # Prefer a lightweight model if available
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_TITLE_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You create concise, file-name-safe titles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=64,
        )
        choice = resp.choices[0].message.content.strip()
        # Basic sanitization
        choice = re.sub(r"[\r\n]+", " ", choice)
        choice = re.sub(r"[\\/:*?\"<>|]", "-", choice)
        return choice[:120]
    except Exception:
        return None


def build_name_mapping(
    files: List[Path],
    root: Path,
    use_ai: bool,
    client: Optional[OpenAI],
) -> Dict[Path, str]:
    """Return mapping of video path -> smart base name (no extension).

    Ensures uniqueness by suffixing duplicates with an index.
    """
    mapping: Dict[Path, str] = {}
    seen: Dict[str, int] = {}
    for p in files:
        base = heuristic_smart_name(p, root)
        if use_ai and client is not None:
            ai_name = ai_smart_name(client, p, root)
            if ai_name:
                base = ai_name
        # prevent empty
        if not base:
            base = p.stem
        # ensure uniqueness
        key = base
        if key in seen:
            seen[key] += 1
            key = f"{base} ({seen[base]})"
        else:
            seen[key] = 0
        mapping[p] = key
    return mapping


def split_video_to_audio_segments(file_path: Path, exist_delete: bool = True) -> List[Path]:
    """Extract audio from an mp4 and split into ~10 minute mp3 segments.

    Returns a list of mp3 segment file paths. Creates a transient directory
    `<video_stem>_segments` in the current working directory and fills it with
    the mp3 chunks. Caller is responsible for removing this directory when
    done.
    """
    # Load audio track from the video (requires ffmpeg via pydub)
    full_audio = AudioSegment.from_file(file_path, format="mp4")
    segment_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
    chunks = make_chunks(full_audio, segment_ms)

    chunk_dir = Path(f"{file_path.stem}_segments")
    if exist_delete and chunk_dir.exists():
        rmtree(chunk_dir)
    chunk_dir.mkdir(mode=0o755, exist_ok=True)

    segment_files: List[Path] = []
    for idx, chunk in enumerate(chunks):
        audio_chunk = chunk_dir.joinpath(f"{file_path.stem}_segment_{idx:02d}.mp3")
        chunk.export(audio_chunk, format="mp3")
        segment_files.append(audio_chunk)

    return segment_files


def transcribe_audio_file(client: OpenAI, audio_path: Path) -> str:
    """Transcribe a single audio file using Whisper-1 and return plain text."""
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_path.open("rb"),
        response_format="text",
    )
    # SDK returns a plain string when response_format='text'
    return response.strip() if isinstance(response, str) else str(response).strip()


def transcribe_video_file(client: OpenAI, video_path: Path) -> str:
    """Transcribe an mp4 video by chunking its audio and concatenating results."""
    print(f"Splitting audio for {video_path.name} ...")
    segments = split_video_to_audio_segments(video_path)
    print(f"Segments directory: {segments[0].parent if segments else 'N/A'}")

    transcripts: List[str] = []
    try:
        for seg in segments:
            print(f"Transcribing {seg.name} ...")
            text = transcribe_audio_file(client, seg)
            transcripts.append(text)
            # Gentle pacing to avoid hammering the API
            sleep(1)
    finally:
        # Cleanup segment directory regardless of success
        if segments:
            print("Cleaning up segments ...")
            rmtree(segments[0].parent)

    return "\n".join(transcripts)


def sanitize_filename(name: str) -> str:
    """Remove or replace characters not safe for common filesystems."""
    name = re.sub(r"[\\/:*?\"<>|]", "-", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def make_output_filename(
    video_path: Path,
    index: int,
    prefix: Optional[str] | None,
    smart_base: Optional[str] = None,
) -> str:
    """Return output filename per spec.

    - Default: `<video_filename_stem>_transcript.txt`
    - With prefix: `<prefix>_NN_transcript.txt` (1-based, 2 digits)
    - With smart_base: `<smart_base>_transcript.txt` (overrides prefix)
    """
    if smart_base:
        return f"{sanitize_filename(smart_base)}_transcript.txt"
    if prefix:
        return f"{prefix}_{index:02d}_transcript.txt"
    return f"{video_path.stem}_transcript.txt"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe mp4 video(s) using Whisper-1")
    parser.add_argument("TARGET", help="Path to an .mp4 file or a directory containing .mp4 files")
    parser.add_argument("-o", "--output-dir", dest="output_dir", help="Directory to write transcripts (default: cwd)")
    parser.add_argument("-p", "--prefix", dest="prefix", help="Prefix for output files; auto-numbered as <prefix>_NN_transcript.txt")
    parser.add_argument("-l", "--list", dest="list_only", action="store_true", help="List discovered .mp4 files and exit")
    parser.add_argument("-r", "--recursive", dest="recursive", action="store_true", help="Traverse subfolders of the target directory")
    parser.add_argument("--smart-names", dest="smart_names", action="store_true", help="Generate smart output names from directory structure")
    parser.add_argument("--use-ai", dest="use_ai", action="store_true", help="Use OpenAI to refine smart names (optional)")
    parser.add_argument("--names-file", dest="names_file", help="Path to cache file for proposed names (defaults to a hidden file in target root)")

    args = parser.parse_args()

    target_path = Path(args.TARGET).expanduser().resolve()
    try:
        video_files = find_video_files(target_path, recursive=args.recursive)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    if args.list_only:
        if not video_files:
            print("No .mp4 files found.")
            return
        if args.smart_names:
            # Build and save proposed names
            client = load_client() if args.use_ai else None
            root = target_path if target_path.is_dir() else target_path.parent
            mapping = build_name_mapping(video_files, root, args.use_ai, client)
            cache_path = Path(args.names_file).expanduser().resolve() if args.names_file else default_names_cache_path(root)
            save_names_cache(cache_path, root, mapping, meta={"use_ai": args.use_ai})
            print("Proposed names (saved). Edit the cache file to adjust:")
            print(f"Cache file: {cache_path}")
            for p in video_files:
                base = mapping.get(p, p.stem)
                out_name = make_output_filename(p, 0, None, smart_base=base)
                print(f"{p} -> {out_name}")
        else:
            for p in video_files:
                print(str(p))
        return

    if not video_files:
        print("No .mp4 files found to transcribe.")
        raise SystemExit(1)

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = load_client()

    # If using smart names, try to load from cache (user may have edited)
    smart_mapping: Dict[Path, str] = {}
    if args.smart_names:
        root = target_path if target_path.is_dir() else target_path.parent
        cache_path = Path(args.names_file).expanduser().resolve() if args.names_file else default_names_cache_path(root)
        raw = load_names_cache(cache_path)
        # Convert to Path keys
        smart_mapping = {Path(k): v for k, v in raw.items()}
        # If mapping incomplete, fill missing using heuristic (avoid AI to keep fast unless explicitly requested)
        missing = [p for p in video_files if p not in smart_mapping]
        if missing:
            fill = build_name_mapping(missing, root, args.use_ai, client if args.use_ai else None)
            smart_mapping.update(fill)
            save_names_cache(cache_path, root, smart_mapping, meta={"use_ai": args.use_ai, "note": "auto-filled missing entries"})

    for idx, video in enumerate(video_files, start=1):
        print(f"Processing: {video.name}")
        try:
            transcript_text = transcribe_video_file(client, video)
        except Exception as exc:
            print(f"Failed to transcribe {video.name}: {exc}")
            continue

        smart_base = smart_mapping.get(video) if args.smart_names else None
        out_name = make_output_filename(video, idx, args.prefix, smart_base=smart_base)
        out_path = out_dir.joinpath(out_name)
        print(f"Saving transcript to: {out_path}")
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(transcript_text)

    print("Done!")


if __name__ == "__main__":
    main()
