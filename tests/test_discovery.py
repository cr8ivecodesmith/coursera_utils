from pathlib import Path

import study_utils.transcribe_video as tv


def test_find_video_files_top_level_vs_recursive(tmp_path):
    # layout:
    # root/
    #   a.mp4, b.txt
    #   sub/
    #     c.mp4
    a = tmp_path / 'a.mp4'
    a.write_text('')
    (tmp_path / 'b.txt').write_text('')
    sub = tmp_path / 'sub'
    sub.mkdir()
    c = sub / 'c.mp4'
    c.write_text('')

    top = tv.find_video_files(tmp_path, recursive=False)
    assert a in top and c not in top

    rec = tv.find_video_files(tmp_path, recursive=True)
    assert a in rec and c in rec
