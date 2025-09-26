set shell := ["bash", "-lc"]
set dotenv-load := true
set dotenv-filename := ".env"
set export := true


## Config

HERE := justfile_directory()
MARKER_DIR := HERE
VERSION := `awk -F\" '/^version/{print $2}' pyproject.toml`
PYTHON_VERSION := trim(read(".python-version"))
PYTHON_VERSIONS := `awk -F'[^0-9]+' '/requires-python/{for(i=$3;i<$5;)printf(i-$3?" ":"")$2"."i++}' pyproject.toml`
BUILD_WHEEL_FILE := `ls dist/study_utils-*.whl 2>/dev/null | head -n 1`


## Recipes

@environment:
    env | sort
    echo $TERMUX_ENV

@version:
    echo "{{ VERSION }}"

@python-versions:
    echo "Development Python version: {{ PYTHON_VERSION }}"
    echo "Supported Python versions: {{ PYTHON_VERSIONS }}"

@setup-dev:
    if [ "$TERMUX_ENV" == "true" ];\
        then uv pip install --link-mode=copy -e .[dev]\
        && uv sync --link-mode=copy --extra dev;\
        else uv pip install -e .[dev]\
        && uv sync --extra dev;\
    fi

@build-clean:
    rm -rf dist build


@build:
    just build-clean
    uv build


@install:
    if [ -z "{{ BUILD_WHEEL_FILE }}" ]; then echo "No wheel found. Run 'just build' first."; exit 1; fi
    if [ "$TERMUX_ENV" == "true" ];\
        then uv pip install --link-mode=copy --force-reinstall {{BUILD_WHEEL_FILE }}\
        && just build-clean;\
        else uv pip install --force-reinstall {{ BUILD_WHEEL_FILE }}\
        && just build-clean;\
    fi

@uninstall:
    uv pip uninstall study_utils

@set-version version:
    python -c "import pathlib, re, sys; version='{{version}}'; version or sys.exit('Provide a version, e.g. `just set-version 0.2.0`.'); path = pathlib.Path('pyproject.toml'); text = path.read_text(); pattern = r'(?m)^version\\s*=\\s*\"[^\"]+\"'; re.search(pattern, text) or sys.exit('Could not find version field in pyproject.toml.'); new_text, count = re.subn(pattern, 'version = \"' + version + '\"', text, count=1); count == 1 or sys.exit('Expected to update exactly one version field.'); path.write_text(new_text); print('Updated project version to ' + version + '.')"

test:
    uv run pytest

lint:
    uv run ruff check --fix src tests
    uv run ruff format src tests

