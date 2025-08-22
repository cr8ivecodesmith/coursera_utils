from pathlib import Path
import json

import transcribe_video as tv


def test_save_and_load_cache_v2_roundtrip(tmp_path):
    cache = tmp_path / '.transcribe_video_names.json'
    root = tmp_path
    video = tmp_path / 'a.mp4'
    video.touch()
    mapping = {video: {"base": "Base A", "final": "P01-Base A.txt"}}
    tv.save_names_cache(cache, root, mapping, meta={"use_ai": False})

    data = json.loads(cache.read_text())
    assert data["version"] == 2
    assert data["names"][str(video)]["base"] == "Base A"
    assert data["names"][str(video)]["final"] == "P01-Base A.txt"

    loaded = tv.load_names_cache(cache)
    assert str(video) in loaded
    assert loaded[str(video)]["base"] == "Base A"


def test_upgrade_from_v1_and_save(tmp_path):
    cache = tmp_path / '.transcribe_video_names.json'
    root = tmp_path
    video = tmp_path / 'b.mp4'
    video.touch()
    # Old v1 format: names map to plain strings (base only)
    cache.write_text(json.dumps({
        "version": 1,
        "root": str(root),
        "names": {str(video): "Old Base Name"}
    }))

    loaded = tv.load_names_cache(cache)
    # v1 returns plain string entries
    assert loaded[str(video)] == "Old Base Name"

    # After saving, it should be v2 with dict entries
    tv.save_names_cache(cache, root, {video: loaded[str(video)]})
    data = json.loads(cache.read_text())
    assert data["version"] == 2
    assert data["names"][str(video)]["base"] == "Old Base Name"

