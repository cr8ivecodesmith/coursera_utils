"""Health diagnostics for the Study RAG workspace."""

from __future__ import annotations

import importlib
import os
import stat
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Dict, Mapping, Tuple

from . import config as config_mod
from . import data_dir
from . import ingest as ingest_mod
from . import vector_store

_DEPENDENCIES: Tuple[Tuple[str, str, str], ...] = (
    ("faiss", "faiss", "faiss-cpu"),
    ("numpy", "numpy", "numpy"),
    ("tiktoken", "tiktoken", "tiktoken"),
    ("openai", "openai", "openai"),
)

_SUBDIR_NAMES = {
    "config": "config",
    "vector_dbs": "rag_dbs",
    "sessions": "rag_sessions",
    "logs": "logs",
}


@dataclass(frozen=True)
class DirectoryStatus:
    """Represents the health of a managed directory."""

    name: str
    path: Path
    exists: bool
    is_dir: bool
    mode: int | None
    severity: str
    message: str | None


@dataclass(frozen=True)
class DependencyStatus:
    """Represents availability of an optional/runtime dependency."""

    name: str
    module: str
    status: str
    version: str | None
    error: str | None


@dataclass(frozen=True)
class StorageUsage:
    """Disk usage summary for an artefact directory."""

    name: str
    path: Path
    size_bytes: int
    issues: Tuple[str, ...]


@dataclass(frozen=True)
class TokenizerStatus:
    """State of the configured tokenizer implementation."""

    tokenizer: str
    encoding: str
    status: str
    message: str | None


@dataclass(frozen=True)
class DoctorReport:
    """Aggregate diagnostics for the Study RAG workspace."""

    data_home: Path
    data_home_severity: str
    data_home_message: str | None
    env_overrides: Dict[str, str]
    directories: Tuple[DirectoryStatus, ...]
    config_path: Path
    config_exists: bool
    config_error: str | None
    dependencies: Tuple[DependencyStatus, ...]
    tokenizer: TokenizerStatus
    vector_stores: Tuple[StorageUsage, ...]
    sessions: Tuple[StorageUsage, ...]


def generate_report(env: Mapping[str, str] | None = None) -> DoctorReport:
    """Collect diagnostic information about the current installation."""

    env_map = dict(os.environ if env is None else env)
    overrides: Dict[str, str] = {}
    if data_dir.DATA_HOME_ENV in env_map:
        overrides[data_dir.DATA_HOME_ENV] = env_map[data_dir.DATA_HOME_ENV]
    if config_mod.CONFIG_PATH_ENV in env_map:
        overrides[config_mod.CONFIG_PATH_ENV] = env_map[
            config_mod.CONFIG_PATH_ENV
        ]

    data_home, data_home_severity, data_home_message = _resolve_data_home(
        env_map
    )

    config_path, config_exists, config_error, cfg = _resolve_config(env_map)

    directories = _collect_directories(data_home)
    dependencies = _check_dependencies()
    tokenizer = _determine_tokenizer(cfg)
    vector_stores = _collect_vector_stores(
        data_home / _SUBDIR_NAMES["vector_dbs"]
    )
    sessions = _collect_sessions(data_home / _SUBDIR_NAMES["sessions"])

    return DoctorReport(
        data_home=data_home,
        data_home_severity=data_home_severity,
        data_home_message=data_home_message,
        env_overrides=overrides,
        directories=directories,
        config_path=config_path,
        config_exists=config_exists,
        config_error=config_error,
        dependencies=dependencies,
        tokenizer=tokenizer,
        vector_stores=vector_stores,
        sessions=sessions,
    )


def has_errors(report: DoctorReport) -> bool:
    """Return ``True`` when critical issues were detected."""

    if report.data_home_severity == "error":
        return True
    if report.config_error:
        return True
    if any(item.severity == "error" for item in report.directories):
        return True
    if any(dep.status == "error" for dep in report.dependencies):
        return True
    if report.tokenizer.status == "error":
        return True
    if any(usage.issues for usage in report.vector_stores):
        return True
    if any(usage.issues for usage in report.sessions):
        return True
    return False


def format_report(report: DoctorReport) -> str:
    lines: list[str] = []
    header = "Study RAG Doctor Report"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")
    lines.extend(_format_data_home_section(report))
    lines.append("")
    lines.extend(_format_config_section(report))
    lines.append("")
    lines.extend(_format_dependency_section(report))
    lines.append("")
    lines.extend(_format_tokenizer_section(report))
    lines.append("")
    lines.extend(_format_vector_store_section(report))
    lines.append("")
    lines.extend(_format_session_section(report))
    return "\n".join(lines)


def _format_data_home_section(report: DoctorReport) -> list[str]:
    lines = [f"Data home: {report.data_home}"]
    if report.data_home_message:
        lines.append(f"  status: {report.data_home_message}")
    if report.env_overrides:
        lines.append("  environment overrides:")
        for key in sorted(report.env_overrides):
            lines.append(f"    - {key} = {report.env_overrides[key]}")
    lines.append("  directories:")
    for item in report.directories:
        mode_text = (
            f"mode {oct(item.mode)}" if item.mode is not None else "mode ?"
        )
        status = item.message if item.message else "ok"
        lines.append(
            "    - {name}: {status} ({mode})".format(
                name=item.name,
                status=status,
                mode=mode_text,
            )
        )
    return lines


def _format_config_section(report: DoctorReport) -> list[str]:
    lines = [f"Config path: {report.config_path}"]
    lines.append(f"  exists: {'yes' if report.config_exists else 'no'}")
    if report.config_error:
        lines.append(f"  error: {report.config_error}")
    return lines


def _format_dependency_section(report: DoctorReport) -> list[str]:
    lines = ["Dependencies:"]
    for dep in report.dependencies:
        if dep.status == "ok":
            suffix = f" ({dep.version})" if dep.version else ""
            lines.append(f"  - {dep.name}: ok{suffix}")
        else:
            reason = dep.error or "unknown error"
            lines.append(f"  - {dep.name}: missing ({reason})")
    if not report.dependencies:
        lines.append("  (none)")
    return lines


def _format_tokenizer_section(report: DoctorReport) -> list[str]:
    message = report.tokenizer.message or "ready"
    return [
        "Tokenizer:",
        "  - {tokenizer} / {encoding}: {status}".format(
            tokenizer=report.tokenizer.tokenizer,
            encoding=report.tokenizer.encoding,
            status=message,
        ),
    ]


def _format_vector_store_section(report: DoctorReport) -> list[str]:
    lines = ["Vector stores: {count}".format(count=len(report.vector_stores))]
    if report.vector_stores:
        total = sum(item.size_bytes for item in report.vector_stores)
        lines.append(f"  total disk usage: {_format_bytes(total)}")
        for item in report.vector_stores:
            suffix = f" issues: {', '.join(item.issues)}" if item.issues else ""
            lines.append(
                "  - {name}: {size}{suffix}".format(
                    name=item.name,
                    size=_format_bytes(item.size_bytes),
                    suffix=suffix,
                )
            )
    else:
        lines.append("  (none)")
    return lines


def _format_session_section(report: DoctorReport) -> list[str]:
    lines = ["Sessions: {count}".format(count=len(report.sessions))]
    if report.sessions:
        total = sum(item.size_bytes for item in report.sessions)
        lines.append(f"  total disk usage: {_format_bytes(total)}")
        for item in report.sessions:
            suffix = f" issues: {', '.join(item.issues)}" if item.issues else ""
            lines.append(
                "  - {name}: {size}{suffix}".format(
                    name=item.name,
                    size=_format_bytes(item.size_bytes),
                    suffix=suffix,
                )
            )
    else:
        lines.append("  (none)")
    return lines


def _resolve_data_home(env: Mapping[str, str]) -> tuple[Path, str, str | None]:
    try:
        data_home = data_dir.get_data_home(env=env, create=False)
    except data_dir.DataDirError as exc:
        raw = env.get(data_dir.DATA_HOME_ENV)
        candidate = (
            Path(raw).expanduser() if raw else data_dir.DEFAULT_DATA_HOME
        )
        message = str(exc)
        return candidate, "error", message

    if not data_home.exists():
        return (
            data_home,
            "warning",
            "missing (will be created automatically)",
        )
    if not data_home.is_dir():
        return data_home, "error", "path is not a directory"
    mode = _safe_mode(data_home)
    if mode is not None and os.name != "nt" and mode != 0o700:
        message = f"permissions expected 0o700, found {oct(mode)}"
        return data_home, "warning", message
    return data_home, "ok", None


def _resolve_config(
    env: Mapping[str, str],
) -> tuple[Path, bool, str | None, config_mod.RagConfig | None]:
    try:
        config_path = config_mod.resolve_config_path(env=env)
    except config_mod.ConfigError as exc:
        raw = env.get(config_mod.CONFIG_PATH_ENV)
        fallback = (
            Path(raw).expanduser()
            if raw
            else data_dir.DEFAULT_DATA_HOME
            / _SUBDIR_NAMES["config"]
            / "rag.toml"
        )
        return fallback, False, str(exc), None

    config_exists = config_path.is_file()
    try:
        cfg = config_mod.load_config(env=env)
        return config_path, config_exists, None, cfg
    except config_mod.ConfigError as exc:
        return config_path, config_exists, str(exc), None


def _collect_directories(data_home: Path) -> Tuple[DirectoryStatus, ...]:
    entries = []
    for name, subdir in (
        ("config", "config"),
        ("vector_dbs", "rag_dbs"),
        ("sessions", "rag_sessions"),
        ("logs", "logs"),
    ):
        path = data_home / subdir
        status = _directory_status(name, path)
        entries.append(status)
    return tuple(entries)


def _directory_status(name: str, path: Path) -> DirectoryStatus:
    if not path.exists():
        return DirectoryStatus(
            name=name,
            path=path,
            exists=False,
            is_dir=False,
            mode=None,
            severity="warning",
            message="missing",
        )
    if not path.is_dir():
        return DirectoryStatus(
            name=name,
            path=path,
            exists=True,
            is_dir=False,
            mode=_safe_mode(path),
            severity="error",
            message="not a directory",
        )
    mode = _safe_mode(path)
    if mode is not None and os.name != "nt" and mode != 0o700:
        return DirectoryStatus(
            name=name,
            path=path,
            exists=True,
            is_dir=True,
            mode=mode,
            severity="warning",
            message=f"permissions {oct(mode)} (expected 0o700)",
        )
    return DirectoryStatus(
        name=name,
        path=path,
        exists=True,
        is_dir=True,
        mode=mode,
        severity="ok",
        message=None,
    )


def _check_dependencies() -> Tuple[DependencyStatus, ...]:
    results = []
    for name, module, package in _DEPENDENCIES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - depends on env
            results.append(
                DependencyStatus(
                    name=name,
                    module=module,
                    status="error",
                    version=None,
                    error=str(exc),
                )
            )
            continue
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:  # pragma: no cover - rare
            version = None
        results.append(
            DependencyStatus(
                name=name,
                module=module,
                status="ok",
                version=version,
                error=None,
            )
        )
    return tuple(results)


def _determine_tokenizer(
    cfg: config_mod.RagConfig | None,
) -> TokenizerStatus:
    if cfg is None:
        return TokenizerStatus(
            tokenizer="unknown",
            encoding="unknown",
            status="warning",
            message="config unavailable",
        )
    chunk_cfg = cfg.ingestion.chunking
    try:
        chunker = ingest_mod.TextChunker(
            tokenizer=chunk_cfg.tokenizer,
            encoding=chunk_cfg.encoding,
            tokens_per_chunk=chunk_cfg.tokens_per_chunk,
            token_overlap=chunk_cfg.token_overlap,
            fallback_delimiter=chunk_cfg.fallback_delimiter,
        )
    except Exception as exc:
        return TokenizerStatus(
            tokenizer=chunk_cfg.tokenizer,
            encoding=chunk_cfg.encoding,
            status="error",
            message=str(exc),
        )

    encoder = getattr(chunker, "_encoder", None)
    if chunk_cfg.tokenizer.lower() == "tiktoken" and encoder is None:
        return TokenizerStatus(
            tokenizer=chunk_cfg.tokenizer,
            encoding=chunk_cfg.encoding,
            status="warning",
            message="tiktoken unavailable; using fallback splitter",
        )
    return TokenizerStatus(
        tokenizer=chunk_cfg.tokenizer,
        encoding=chunk_cfg.encoding,
        status="ok",
        message=None,
    )


def _collect_vector_stores(root: Path) -> Tuple[StorageUsage, ...]:
    usages = []
    if not root.exists() or not root.is_dir():
        return tuple()
    repository = vector_store.VectorStoreRepository(root)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        issues = []
        try:
            repository.load_manifest(child.name)
        except vector_store.VectorStoreError as exc:
            issues.append(str(exc))
        size = _directory_size(child)
        usages.append(
            StorageUsage(
                name=child.name,
                path=child,
                size_bytes=size,
                issues=tuple(issues),
            )
        )
    return tuple(usages)


def _collect_sessions(root: Path) -> Tuple[StorageUsage, ...]:
    usages = []
    if not root.exists() or not root.is_dir():
        return tuple()
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        issues = []
        session_file = child / "session.json"
        if not session_file.is_file():
            issues.append("missing session.json")
        size = _directory_size(child)
        usages.append(
            StorageUsage(
                name=child.name,
                path=child,
                size_bytes=size,
                issues=tuple(issues),
            )
        )
    return tuple(usages)


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:  # pragma: no cover - depends on filesystem
                continue
    return total


def _safe_mode(path: Path) -> int | None:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return None


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    units = ["KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        value /= 1024.0
        if value < 1024:
            if value < 10:
                return f"{value:.1f} {unit}"
            return f"{value:.0f} {unit}"
    value /= 1024.0
    if value < 10:
        return f"{value:.1f} TB"
    return f"{value:.0f} TB"
