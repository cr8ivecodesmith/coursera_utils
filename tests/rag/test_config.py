from __future__ import annotations

from pathlib import Path

import pytest

from study_utils.rag import config as config_mod


def _write_config(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


BASE_CONFIG = """
[providers]
default = "openai"

[providers.openai]
chat_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-large"
max_input_tokens = 6000
max_output_tokens = 2000
temperature = 0.2
request_timeout_seconds = 60

[ingestion]
max_workers = 4
file_batch_size = 32

[ingestion.chunking]
tokenizer = "tiktoken"
encoding = "cl100k_base"
tokens_per_chunk = 300
token_overlap = 30
fallback_delimiter = "\\n\\n"

[ingestion.dedupe]
strategy = "checksum"
checksum_algorithm = "sha256"

[retrieval]
top_k = 5
max_context_tokens = 1800
score_threshold = 0.2

[chat]
max_history_turns = 200
response_tokens = 800
stream = true

[logging]
level = "INFO"
verbose = false
"""

CHUNKING_BLOCK = """
[ingestion.chunking]
tokenizer = "tiktoken"
encoding = "cl100k_base"
tokens_per_chunk = 300
token_overlap = 30
fallback_delimiter = "\\n\\n"

"""


def test_config_template_parses_as_toml():
    template = config_mod.config_template()
    import tomllib

    data = tomllib.loads(template)
    assert data["providers"]["default"] == "openai"


def test_write_template_honours_overwrite(tmp_path):
    target = tmp_path / "rag.toml"
    config_mod.write_template(target)

    with pytest.raises(config_mod.ConfigError):
        config_mod.write_template(target)

    config_mod.write_template(target, overwrite=True)


def test_load_config_merges_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data-home"))
    content = f"""
[paths]
data_home = "{tmp_path / "alt-home"}"

[providers]
default = "openai"

[providers.openai]
chat_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-large"
max_input_tokens = 8000
max_output_tokens = 3000
temperature = 0.5
request_timeout_seconds = 30

[ingestion]
max_workers = 2
file_batch_size = 8

[ingestion.chunking]
tokenizer = "tiktoken"
encoding = "cl100k_base"
tokens_per_chunk = 512
token_overlap = 64
fallback_delimiter = "\\n\\n"

[ingestion.dedupe]
strategy = "checksum"
checksum_algorithm = "sha512"

[retrieval]
top_k = 7
max_context_tokens = 2048
score_threshold = 0.3

[chat]
max_history_turns = 120
response_tokens = 600
stream = false

[logging]
level = "debug"
verbose = true
"""
    target = tmp_path / "rag.toml"
    _write_config(target, content)

    cfg = config_mod.load_config(explicit_path=target)

    assert cfg.data_home == (tmp_path / "alt-home").resolve()
    assert cfg.providers.openai.max_input_tokens == 8000
    assert cfg.ingestion.chunking.tokens_per_chunk == 512
    assert cfg.ingestion.dedupe.checksum_algorithm == "sha512"
    assert cfg.retrieval.top_k == 7
    assert cfg.chat.stream is False
    assert cfg.logging.level == "DEBUG"
    assert cfg.logging.verbose is True


def test_load_config_uses_default_data_home_when_not_overridden(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "dh"))
    content = BASE_CONFIG
    target = tmp_path / "rag.toml"
    _write_config(target, content)

    cfg = config_mod.load_config(explicit_path=target)

    assert cfg.paths.data_home_override is None
    assert cfg.data_home == (tmp_path / "dh").resolve()


def test_resolve_config_path_prefers_env(tmp_path, monkeypatch):
    expected = tmp_path / "custom" / "rag.toml"
    monkeypatch.setenv(config_mod.CONFIG_PATH_ENV, str(expected))
    result = config_mod.resolve_config_path()
    assert result == expected.resolve()


def test_default_tree_returns_deep_copy():
    tree = config_mod.default_tree()
    tree["ingestion"]["max_workers"] = 99
    new_tree = config_mod.default_tree()
    assert new_tree["ingestion"]["max_workers"] == 4


def test_load_config_unknown_key_raises(tmp_path):
    content = (
        BASE_CONFIG
        + """
[extra]
bogus = 1
"""
    )
    target = tmp_path / "config.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_load_config_rejects_invalid_overlap(tmp_path):
    content = BASE_CONFIG.replace(
        "token_overlap = 30",
        "token_overlap = 128",
    ).replace("tokens_per_chunk = 300", "tokens_per_chunk = 128")
    target = tmp_path / "bad.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_load_config_requires_checksum_strategy(tmp_path):
    content = BASE_CONFIG.replace(
        'strategy = "checksum"',
        'strategy = "hash"',
    )
    target = tmp_path / "bad-strategy.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_load_config_validates_logging_level(tmp_path):
    content = BASE_CONFIG.replace('level = "INFO"', 'level = "TRACE"')
    target = tmp_path / "bad-log.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_write_template_sets_permissions_when_possible(tmp_path):
    target = tmp_path / "perm" / "rag.toml"
    result = config_mod.write_template(target)
    assert result == target


def test_load_config_requires_tables(tmp_path):
    target = tmp_path / "bad.toml"
    target.write_text("name = 'value'\n", encoding="utf-8")

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_resolve_config_path_with_explicit_value(tmp_path):
    explicit = tmp_path / "explicit.toml"
    resolved = config_mod.resolve_config_path(explicit_path=explicit)
    assert resolved == explicit.resolve()


def test_load_config_requires_chunking_table(tmp_path):
    content = BASE_CONFIG.replace(CHUNKING_BLOCK, "ingestion.chunking = 3\n\n")
    target = tmp_path / "bad-table.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_load_config_rejects_empty_api_base(tmp_path):
    content = BASE_CONFIG.replace(
        "request_timeout_seconds = 60",
        'request_timeout_seconds = 60\napi_base = ""',
    )
    target = tmp_path / "bad-api.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_load_config_rejects_invalid_boolean(tmp_path):
    content = BASE_CONFIG.replace("stream = true", 'stream = "yes"')
    target = tmp_path / "bad-bool.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_load_config_rejects_non_positive_int(tmp_path):
    content = BASE_CONFIG.replace("top_k = 5", "top_k = 0")
    target = tmp_path / "bad-int.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_load_config_rejects_score_threshold_out_of_range(tmp_path):
    content = BASE_CONFIG.replace(
        "score_threshold = 0.2",
        "score_threshold = 1.5",
    )
    target = tmp_path / "bad-score.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_load_config_rejects_empty_data_home_override(tmp_path):
    content = '[paths]\ndata_home = ""\n\n' + BASE_CONFIG
    target = tmp_path / "bad-home.toml"
    _write_config(target, content)

    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=target)


def test_merge_dict_requires_table():
    base = config_mod.default_tree()
    with pytest.raises(config_mod.ConfigError):
        config_mod._merge_dict(base, {"ingestion": 5})


def test_require_non_negative_int_rejects_negative():
    with pytest.raises(config_mod.ConfigError):
        config_mod._require_non_negative_int(
            -1,
            field="ingestion.token_overlap",
        )


def test_require_float_range_rejects_non_number():
    with pytest.raises(config_mod.ConfigError):
        config_mod._require_float_range(
            "bad",
            field="retrieval.score_threshold",
            min_value=0,
            max_value=1,
        )


def test_require_string_type_enforcement():
    with pytest.raises(config_mod.ConfigError):
        config_mod._require_string(123, field="providers.openai.chat_model")


def test_require_string_allow_whitespace_empty():
    with pytest.raises(config_mod.ConfigError):
        config_mod._require_string(
            "",
            field="ingestion.chunking.fallback_delimiter",
            allow_whitespace=True,
        )


def test_require_string_rejects_blank_after_trim():
    with pytest.raises(config_mod.ConfigError):
        config_mod._require_string("   ", field="providers.default")


def test_coerce_optional_string_trims_value():
    result = config_mod._coerce_optional_string(
        " https://api.example.com ",
        field="providers.openai.api_base",
    )
    assert result == "https://api.example.com"


def test_build_providers_requires_openai_table():
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_providers({"default": "openai"})


def test_build_providers_rejects_non_openai_default():
    data = {
        "default": "azure",
        "openai": config_mod.default_tree()["providers"]["openai"],
    }
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_providers(data)


def test_build_ingestion_requires_chunking_table():
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_ingestion({"dedupe": {}})


def test_build_ingestion_requires_dedupe_table():
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_ingestion({"chunking": {}})


def test_build_config_requires_mappings():
    tree = config_mod.default_tree()
    tree["paths"] = []
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_config(tree)


def test_build_config_requires_providers_mapping():
    tree = config_mod.default_tree()
    tree["providers"] = []
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_config(tree)


def test_build_config_requires_ingestion_mapping():
    tree = config_mod.default_tree()
    tree["ingestion"] = []
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_config(tree)


def test_build_config_requires_retrieval_mapping():
    tree = config_mod.default_tree()
    tree["retrieval"] = []
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_config(tree)


def test_build_config_requires_chat_mapping():
    tree = config_mod.default_tree()
    tree["chat"] = []
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_config(tree)


def test_build_config_requires_logging_mapping():
    tree = config_mod.default_tree()
    tree["logging"] = []
    with pytest.raises(config_mod.ConfigError):
        config_mod._build_config(tree)


def test_load_toml_missing_file(tmp_path):
    with pytest.raises(config_mod.ConfigError):
        config_mod._load_toml(tmp_path / "missing.toml")


def test_load_toml_bad_content(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('x = "y"\n]', encoding="utf-8")
    with pytest.raises(config_mod.ConfigError):
        config_mod._load_toml(path)


def test_resolve_config_path_defaults_to_data_dir(tmp_path):
    env = {config_mod.data_dir.DATA_HOME_ENV: str(tmp_path / "data")}
    path = config_mod.resolve_config_path(env=env)
    assert path.parent == (tmp_path / "data" / "config").resolve()


def test_load_config_requires_mapping_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "_load_toml", lambda path: [])
    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(explicit_path=tmp_path / "fake.toml")


def test_write_template_handles_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "perm" / "rag.toml"

    def raise_permission(path_self, mode):  # noqa: ARG001
        raise PermissionError

    monkeypatch.setattr(Path, "chmod", raise_permission)
    result = config_mod.write_template(target)
    assert result == target
