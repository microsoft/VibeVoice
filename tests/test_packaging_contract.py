"""Keep the documented vLLM installation path aligned with package metadata."""

from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_vllm_extra_pins_the_supported_runtime() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["optional-dependencies"]["vllm"] == ["vllm==0.14.1"]


def test_vllm_launcher_installs_the_declared_extra() -> None:
    launcher = (ROOT / "vllm_plugin" / "scripts" / "start_server.py").read_text(
        encoding="utf-8"
    )

    assert '"-e", "/app[vllm]"' in launcher
