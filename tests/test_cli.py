import sys
import types

import pytest

from study_utils import cli


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
