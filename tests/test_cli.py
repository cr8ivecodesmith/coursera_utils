import sys
import types
from pathlib import Path

import pytest

from study_utils import cli
from study_utils.core import config_templates
from study_utils.core import workspace as workspace_mod
from study_utils.generate_document import config as gd_config
import study_utils.generate_document.cli as generate_document_cli


@pytest.fixture(autouse=True)
def reset_metadata(monkeypatch):
    """Ensure metadata.version is controllable during tests."""

    def fake_version(name: str) -> str:
        assert name == "study-utils"
        return "0.0-test"

    monkeypatch.setattr(cli.metadata, "version", fake_version)
    yield


def test_version_command_handles_missing_package(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(cli.metadata.PackageNotFoundError()),
    )
    code = cli.main(["version"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "unknown"


def test_no_args_prints_usage_and_returns_error(capsys):
    code = cli.main([])
    captured = capsys.readouterr()
    assert code == 2
    assert "Usage: study" in captured.out
    assert "Available commands:" in captured.out


def test_help_flag_shows_usage(capsys):
    code = cli.main(["--help"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Usage: study" in captured.out


def test_help_command_without_target(capsys):
    code = cli.main(["help"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Usage: study" in captured.out


def test_list_outputs_command_table(capsys):
    code = cli.main(["list"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Available commands:" in captured.out
    # The table should include the known command names.
    assert "transcribe-video" in captured.out
    assert "markdown-to-pdf" in captured.out
    assert "convert-markdown" in captured.out
    assert "init" in captured.out


def test_help_known_command(capsys):
    code = cli.main(["help", "transcribe-video"])
    captured = capsys.readouterr()
    assert code == 0
    assert "transcribe-video" in captured.out
    assert "Run `study transcribe-video --help`" in captured.out


def test_help_unknown_command(capsys):
    code = cli.main(["help", "does-not-exist"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Unknown command" in captured.err
    assert "Available commands:" in captured.err


def test_version_command(capsys):
    code = cli.main(["version"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "0.0-test"


def test_version_flag(capsys):
    code = cli.main(["--version"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "0.0-test"


def test_unknown_command_errors(capsys):
    code = cli.main(["bogus"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Unknown command" in captured.err
    assert "Available commands:" in captured.err


def test_dispatch_invokes_module_main_with_passthrough(monkeypatch):
    before = list(sys.argv)
    captured: dict[str, list[str]] = {}

    def fake_import(module_name: str):
        assert module_name == "study_utils.transcribe_video"

        def stub_main(argv):
            captured["argv"] = list(argv)
            captured["sys_argv"] = list(sys.argv)
            return 7

        return types.SimpleNamespace(main=stub_main)

    monkeypatch.setattr(cli, "import_module", fake_import)
    code = cli.main(["transcribe-video", "--", "--flag"])
    assert code == 7
    assert captured["argv"] == ["--", "--flag"]
    assert captured["sys_argv"][0] == "study transcribe-video"
    assert captured["sys_argv"][1:] == ["--", "--flag"]
    assert list(sys.argv) == before


def test_dispatch_supports_main_without_parameters(monkeypatch):
    called = {"count": 0}

    def fake_import(module_name: str):
        assert module_name == "study_utils.transcribe_video"

        def stub_main():
            called["count"] += 1
            assert sys.argv[0] == "study transcribe-video"

        return types.SimpleNamespace(main=stub_main)

    monkeypatch.setattr(cli, "import_module", fake_import)
    code = cli.main(["transcribe-video"])
    assert code == 0
    assert called["count"] == 1


def test_dispatch_propagates_system_exit_code(monkeypatch):
    def fake_import(module_name: str):
        assert module_name == "study_utils.transcribe_video"

        def stub_main(argv):
            raise SystemExit(5)

        return types.SimpleNamespace(main=stub_main)

    monkeypatch.setattr(cli, "import_module", fake_import)
    code = cli.main(["transcribe-video"])
    assert code == 5


def test_dispatch_handles_system_exit_message(monkeypatch, capsys):
    def fake_import(module_name: str):
        assert module_name == "study_utils.transcribe_video"

        def stub_main(argv):
            raise SystemExit("boom")

        return types.SimpleNamespace(main=stub_main)

    monkeypatch.setattr(cli, "import_module", fake_import)
    code = cli.main(["transcribe-video"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == "boom"


def test_dispatch_treats_system_exit_none_as_success(monkeypatch):
    def fake_import(module_name: str):
        assert module_name == "study_utils.transcribe_video"

        def stub_main(argv):
            raise SystemExit()

        return types.SimpleNamespace(main=stub_main)

    monkeypatch.setattr(cli, "import_module", fake_import)
    code = cli.main(["transcribe-video"])
    assert code == 0


def test_dispatch_normalizes_non_int_return(monkeypatch):
    def fake_import(module_name: str):
        assert module_name == "study_utils.transcribe_video"

        def stub_main(argv):
            return "done"

        return types.SimpleNamespace(main=stub_main)

    monkeypatch.setattr(cli, "import_module", fake_import)
    code = cli.main(["transcribe-video"])
    assert code == 0


def test_study_cli_runs_init_end_to_end(tmp_path, capsys):
    target = tmp_path / "workspace"

    code = cli.main(["init", "--path", str(target)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Workspace ready" in captured.out
    assert target.is_dir()
    for entry in ("config", "logs", "converted"):
        assert (target / entry).is_dir()


def test_study_cli_runs_convert_markdown_config_init(tmp_path, capsys):
    destination = tmp_path / "convert_markdown.toml"

    code = cli.main(
        [
            "convert-markdown",
            "config",
            "init",
            "--path",
            str(destination),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert destination.exists()
    template = config_templates.get_template("convert_markdown")
    assert destination.read_text(encoding="utf-8") == template.read_text()
    assert str(destination.resolve()) in captured.out


def test_generate_document_cli_config_init_writes_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = config_templates.get_template("generate_document")
    code = cli.main(
        [
            "generate-document",
            "config",
            "init",
            "--workspace",
            str(tmp_path),
        ]
    )

    assert code == 0
    target = tmp_path / "config" / gd_config.CONFIG_FILENAME
    assert target.exists()
    assert target.read_text(encoding="utf-8") == template.read_text()

    captured = capsys.readouterr()
    assert str(target) in captured.out
    assert captured.err == ""


def test_generate_document_cli_config_init_requires_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = config_templates.get_template("generate_document")
    custom = tmp_path / "custom.toml"
    custom.write_text("existing", encoding="utf-8")

    code = cli.main(
        [
            "generate-document",
            "config",
            "init",
            "--path",
            str(custom),
        ]
    )

    assert code == 1
    captured = capsys.readouterr()
    assert "Config already exists" in captured.err
    assert custom.read_text(encoding="utf-8") == "existing"

    code = cli.main(
        [
            "generate-document",
            "config",
            "init",
            "--path",
            str(custom),
            "--force",
        ]
    )

    assert code == 0
    assert custom.read_text(encoding="utf-8") == template.read_text()
    captured = capsys.readouterr()
    assert str(custom) in captured.out


def test_generate_document_cli_config_init_relative_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    code = cli.main(
        [
            "generate-document",
            "config",
            "init",
            "--path",
            "local.toml",
        ]
    )

    assert code == 0
    target = tmp_path / "local.toml"
    assert target.exists()
    captured = capsys.readouterr()
    assert str(target.resolve()) in captured.out


def test_generate_document_cli_config_init_workspace_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_workspace(*_args, **_kwargs):
        raise workspace_mod.WorkspaceError("workspace boom")

    monkeypatch.setattr(
        generate_document_cli.workspace_mod,
        "ensure_workspace",
        fail_workspace,
    )

    code = cli.main(
        [
            "generate-document",
            "config",
            "init",
        ]
    )

    assert code == 1
    captured = capsys.readouterr()
    assert "workspace boom" in captured.err


def test_cli_generate_document_invokes_subcommand_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv(
        workspace_mod.WORKSPACE_ENV,
        str(workspace_root),
    )
    layout = workspace_mod.ensure_workspace(path=workspace_root)
    workspace_cfg = layout.path_for("config") / gd_config.CONFIG_FILENAME
    workspace_cfg.write_text(
        "[keywords]\nprompt='Workspace'\n",
        encoding="utf-8",
    )

    project = tmp_path / "project"
    project.mkdir()
    local_cfg = project / gd_config.CONFIG_FILENAME
    local_cfg.write_text(
        "[keywords]\nprompt='Local'\n",
        encoding="utf-8",
    )

    src_dir = project / "refs"
    src_dir.mkdir()
    ref = src_dir / "note.txt"
    ref.write_text("Reference", encoding="utf-8")
    out_path = project / "out.md"

    captured: dict[str, object] = {}

    def fake_generate_document(**kwargs):
        captured.update(kwargs)
        return 2

    monkeypatch.setattr(
        "study_utils.generate_document.cli.generate_document",
        fake_generate_document,
    )
    monkeypatch.chdir(project)

    code = cli.main(
        [
            "generate-document",
            "keywords",
            str(out_path),
            str(src_dir),
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "Generated document" in out
    assert captured["extensions"] == {"txt", "md", "markdown"}
    assert captured["inputs"] == [src_dir.resolve()]
    assert captured["output_path"] == out_path.resolve()
    assert captured["config_path"] == workspace_cfg.resolve()


def test_cli_text_combiner_invokes_subcommand_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import study_utils.text_combiner as tc

    source = tmp_path / "a.txt"
    source.write_text("Alpha", encoding="utf-8")
    out_path = tmp_path / "combined.txt"

    captured: dict[str, object] = {}

    def fake_iter_text_files(inputs, extensions, level_limit):
        captured["iter_extensions"] = set(extensions)
        return [source]

    def fake_order_files(files, order_by):
        captured["order_by"] = order_by
        return list(files)

    def fake_combine_files(files, output_path, options):
        captured["options"] = options
        output_path.write_text("done", encoding="utf-8")
        return len(files)

    monkeypatch.setattr(tc, "iter_text_files", fake_iter_text_files)
    monkeypatch.setattr(tc, "order_files", fake_order_files)
    monkeypatch.setattr(tc, "combine_files", fake_combine_files)

    code = cli.main(["text-combiner", str(out_path), str(source)])

    assert code == 0
    out = capsys.readouterr().out
    assert "Combined 1 file(s)" in out
    assert captured["iter_extensions"] == {"txt"}
    options = captured["options"]
    assert options.extensions == {"txt"}
    assert options.combine_by == "NEW"
    assert options.level_limit == 0
    assert out_path.read_text(encoding="utf-8") == "done"


def test_cli_markdown_to_pdf_invokes_subcommand_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import study_utils.markdown_to_pdf as mdp

    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "doc.md"
    doc.write_text("# Title", encoding="utf-8")
    out_path = tmp_path / "out.pdf"

    captured: dict[str, object] = {}

    def fake_iter_text_files(paths, extensions, level_limit):
        captured["extensions"] = set(extensions)
        yield doc

    monkeypatch.setattr(mdp, "iter_text_files", fake_iter_text_files)
    monkeypatch.setattr(mdp, "default_highlight_css", lambda *_: "css")
    monkeypatch.setattr(mdp, "build_markdown_it", lambda _: object())
    monkeypatch.setattr(
        mdp,
        "_render_markdown_parts",
        lambda files, md: ([("doc", "<p>Body</p>")], "sample"),
    )
    monkeypatch.setattr(
        mdp, "_build_title_page_html", lambda args, sample: None
    )
    monkeypatch.setattr(mdp, "_print_dry_run", lambda args, files, out: None)

    code = cli.main(
        [
            "markdown-to-pdf",
            str(out_path),
            str(docs),
            "--dry-run",
        ]
    )

    assert code == 0
    assert captured["extensions"] == {"md", "markdown"}


def test_dispatch_supports_varargs_main(monkeypatch):
    captured = {}

    def fake_import(module_name: str):
        assert module_name == "study_utils.transcribe_video"

        def stub_main(*received):
            captured["received"] = received
            return 3

        return types.SimpleNamespace(main=stub_main)

    monkeypatch.setattr(cli, "import_module", fake_import)
    code = cli.main(["transcribe-video", "--foo", "bar"])
    assert code == 3
    assert captured["received"] == (["--foo", "bar"],)


def test_pending_command_without_handler_reports_error(monkeypatch, capsys):
    spec = cli.CommandSpec(
        name="pending",
        summary="Placeholder command.",
        handler=None,
    )
    monkeypatch.setitem(cli.COMMANDS, "pending", spec)
    code = cli.main(["pending"])
    captured = capsys.readouterr()
    assert code == 2
    assert "not yet implemented" in captured.err


def test_positional_parameter_count_handles_signature_failure(monkeypatch):
    def boom(_func):
        raise TypeError("no signature")

    monkeypatch.setattr(cli.inspect, "signature", boom)

    def stub():
        return None

    assert cli._positional_parameter_count(stub) == 0


def test_dispatch_handles_system_exit_unknown_payload(monkeypatch, capsys):
    class Payload:
        pass

    def fake_import(module_name: str):
        assert module_name == "study_utils.transcribe_video"

        def stub_main(argv):
            raise SystemExit(Payload())

        return types.SimpleNamespace(main=stub_main)

    monkeypatch.setattr(cli, "import_module", fake_import)
    code = cli.main(["transcribe-video"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err == ""
