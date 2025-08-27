import json
from pathlib import Path

import app.transcribe_video as tv


def run_main_with_args(args):
    # Helper to call main() with custom argv
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["transcribe_video.py"] + args
        tv.main()
    finally:
        sys.argv = old_argv


def test_list_smart_names_saves_final_names(tmp_path, capsys):
    # Build a small tree with two videos
    root = tmp_path / 'course'
    root.mkdir()
    (root / 'm01').mkdir()
    v1 = root / 'm01' / '01 - Intro.mp4'
    v1.parent.mkdir(parents=True, exist_ok=True)
    v1.write_text('')
    v2 = root / 'm01' / '02 - Basics.mp4'
    v2.write_text('')

    cache_path = root / '.transcribe_video_names.json'

    # Preview with smart names and a prefix
    run_main_with_args([
        str(root),
        '--list',
        '--recursive',
        '--smart-names',
        '--names-file', str(cache_path),
        '-p', 'text:PRE-',
        '-p', 'counter:NN',
        '-p', 'text:-',
    ])

    out = capsys.readouterr().out
    assert '->' in out  # shows mapping
    assert cache_path.exists()

    data = json.loads(cache_path.read_text())
    # Both files should have final names with the prefix applied
    e1 = data['names'][str(v1)]
    e2 = data['names'][str(v2)]
    assert e1['final'].startswith('PRE-01-')
    assert e2['final'].startswith('PRE-02-')
