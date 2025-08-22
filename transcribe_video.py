import os
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import List

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


def find_video_files(target: Path) -> List[Path]:
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

    files = [p for p in sorted(target.iterdir()) if p.is_file() and p.suffix.lower() == ".mp4"]
    return files


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


def make_output_filename(video_path: Path, index: int, prefix: str | None) -> str:
    """Return output filename per spec.

    - Default: `<video_filename_stem>_transcript.txt`
    - With prefix: `<prefix>_NN_transcript.txt` (1-based, 2 digits)
    """
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

    args = parser.parse_args()

    target_path = Path(args.TARGET).expanduser().resolve()
    try:
        video_files = find_video_files(target_path)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    if args.list_only:
        if not video_files:
            print("No .mp4 files found.")
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

    for idx, video in enumerate(video_files, start=1):
        print(f"Processing: {video.name}")
        try:
            transcript_text = transcribe_video_file(client, video)
        except Exception as exc:
            print(f"Failed to transcribe {video.name}: {exc}")
            continue

        out_name = make_output_filename(video, idx, args.prefix)
        out_path = out_dir.joinpath(out_name)
        print(f"Saving transcript to: {out_path}")
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(transcript_text)

    print("Done!")


if __name__ == "__main__":
    main()
