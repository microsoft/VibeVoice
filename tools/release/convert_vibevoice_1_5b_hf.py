#!/usr/bin/env python3
"""Create the official Transformers-native VibeVoice 1.5B release artifact."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable


OFFICIAL_SOURCE_REPOSITORY = "microsoft/VibeVoice-1.5B"
OFFICIAL_SOURCE_REVISION = "c00898d257e6b46004e3e2866a47534085fb685a"
TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers.git"
TRANSFORMERS_REVISION = "640a08a597034221ca1c4fc0c129cf0118179225"
QWEN_TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-1.5B"
QWEN_TOKENIZER_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
NATIVE_REFERENCE_REPOSITORY = "vibevoice/VibeVoice-1.5B-hf"
NATIVE_REFERENCE_REVISION = "edc39f80f5cae656da37baf8faa8f5502bf7081f"
EXPECTED_TENSOR_COUNT = 1204

REQUIRED_OUTPUT_FILES = (
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class ConversionError(RuntimeError):
    """Raised when a release input or converted artifact is not exact."""


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConversionError(f"{path} must contain a JSON object.")
    return value


def read_safetensors_header(path: Path) -> dict[str, TensorMetadata]:
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise ConversionError(f"{path} is not a valid safetensors file.")
        header_size = int.from_bytes(header_size_bytes, "little")
        if header_size <= 0 or header_size > path.stat().st_size - 8:
            raise ConversionError(f"{path} has an invalid safetensors header size.")
        header = json.loads(handle.read(header_size))

    if not isinstance(header, dict):
        raise ConversionError(f"{path} has an invalid safetensors header.")

    tensors: dict[str, TensorMetadata] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ConversionError(f"{path} contains an invalid tensor entry.")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        if not isinstance(dtype, str) or not isinstance(shape, list) or any(
            not isinstance(dimension, int) or dimension < 0 for dimension in shape
        ):
            raise ConversionError(f"{path} has invalid metadata for {name!r}.")
        tensors[name] = TensorMetadata(shape=tuple(shape), dtype=dtype)
    return tensors


def checkpoint_tensor_metadata(
    checkpoint_dir: Path, expected_tensor_count: int | None = None
) -> dict[str, TensorMetadata]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ConversionError(f"Missing safetensors index: {index_path}")

    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()
    ):
        raise ConversionError(f"{index_path} has an invalid weight_map.")
    if expected_tensor_count is not None and len(weight_map) != expected_tensor_count:
        raise ConversionError(
            f"{index_path} has {len(weight_map)} tensors; expected {expected_tensor_count}."
        )

    tensors: dict[str, TensorMetadata] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = checkpoint_dir / shard_name
        if not shard_path.is_file():
            raise ConversionError(f"Index references missing shard: {shard_path}")
        shard_tensors = read_safetensors_header(shard_path)
        for name, metadata in shard_tensors.items():
            if name in tensors:
                raise ConversionError(f"Tensor {name!r} occurs in more than one shard.")
            tensors[name] = metadata

    indexed_names = set(weight_map)
    actual_names = set(tensors)
    if indexed_names != actual_names:
        missing = sorted(indexed_names - actual_names)
        extra = sorted(actual_names - indexed_names)
        raise ConversionError(
            f"Safetensors index/header mismatch: missing={missing[:3]}, extra={extra[:3]}."
        )
    return tensors


def mapping_digest(mapping: dict[str, tuple[str, TensorMetadata]]) -> str:
    payload = [
        {
            "native_name": native_name,
            "source_name": source_name,
            "shape": list(metadata.shape),
            "dtype": metadata.dtype,
        }
        for native_name, (source_name, metadata) in sorted(mapping.items())
    ]
    return sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def assert_tensor_metadata_matches(
    source: dict[str, TensorMetadata],
    native: dict[str, TensorMetadata],
    map_key: Callable[[str], str],
    expected_tensor_count: int = EXPECTED_TENSOR_COUNT,
) -> str:
    if len(source) != expected_tensor_count or len(native) != expected_tensor_count:
        raise ConversionError(
            f"Expected {expected_tensor_count} tensors, got source={len(source)}, native={len(native)}."
        )

    mapped: dict[str, tuple[str, TensorMetadata]] = {}
    for source_name, metadata in source.items():
        native_name = map_key(source_name)
        if native_name in mapped:
            raise ConversionError(f"Two source tensors map to {native_name!r}.")
        mapped[native_name] = (source_name, metadata)

    missing = sorted(set(mapped) - set(native))
    extra = sorted(set(native) - set(mapped))
    if missing or extra:
        raise ConversionError(
            f"Tensor key mismatch after conversion: missing={missing[:3]}, extra={extra[:3]}."
        )

    mismatches = [
        name
        for name, (_, metadata) in mapped.items()
        if native[name].shape != metadata.shape or native[name].dtype != metadata.dtype
    ]
    if mismatches:
        name = mismatches[0]
        raise ConversionError(
            f"Tensor metadata mismatch for {name!r}: "
            f"source={asdict(mapped[name][1])}, native={asdict(native[name])}."
        )
    return mapping_digest(mapped)


def assert_transformers_revision(transformers_source: Path) -> Path:
    transformers_source = transformers_source.resolve()
    converter_path = (
        transformers_source
        / "src"
        / "transformers"
        / "models"
        / "vibevoice"
        / "convert_vibevoice_to_hf.py"
    )
    if not converter_path.is_file():
        raise ConversionError(f"Pinned VibeVoice converter is absent: {converter_path}")

    result = subprocess.run(
        ["git", "-C", str(transformers_source), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ConversionError(f"Cannot resolve the Transformers checkout: {result.stderr.strip()}")
    actual_revision = result.stdout.strip()
    if actual_revision != TRANSFORMERS_REVISION:
        raise ConversionError(
            f"Transformers must be at {TRANSFORMERS_REVISION}, got {actual_revision}."
        )

    clean_result = subprocess.run(
        ["git", "-C", str(transformers_source), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if clean_result.returncode:
        raise ConversionError(f"Cannot inspect Transformers checkout: {clean_result.stderr.strip()}")
    if clean_result.stdout:
        raise ConversionError("Transformers checkout has tracked changes; use a clean pinned checkout.")
    return converter_path


def import_canonical_converter(transformers_source: Path) -> Any:
    converter_path = assert_transformers_revision(transformers_source)
    source_root = transformers_source.resolve() / "src"
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location("pinned_vibevoice_converter", converter_path)
    if spec is None or spec.loader is None:
        raise ConversionError(f"Cannot import canonical converter: {converter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import transformers

    transformers_path = Path(transformers.__file__).resolve()
    if source_root not in transformers_path.parents:
        raise ConversionError(f"Converter imported Transformers from {transformers_path}, not {source_root}.")
    return module


def download_official_checkpoint() -> Path:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise ConversionError("Install huggingface_hub before running this converter.") from error

    info = HfApi().model_info(
        repo_id=OFFICIAL_SOURCE_REPOSITORY,
        revision=OFFICIAL_SOURCE_REVISION,
    )
    if info.sha != OFFICIAL_SOURCE_REVISION:
        raise ConversionError(
            f"Hub resolved {OFFICIAL_SOURCE_REPOSITORY} to {info.sha}, "
            f"not {OFFICIAL_SOURCE_REVISION}."
        )
    return Path(
        snapshot_download(
            repo_id=OFFICIAL_SOURCE_REPOSITORY,
            revision=OFFICIAL_SOURCE_REVISION,
            allow_patterns=(
                "config.json",
                "preprocessor_config.json",
                "model.safetensors.index.json",
                "model-*.safetensors",
            ),
        )
    )


def convert_with_canonical_tool(
    converter: Any, source_dir: Path, output_dir: Path
) -> None:
    canonical_snapshot_download = converter.snapshot_download
    canonical_tokenizer_loader = converter.Qwen2TokenizerFast.from_pretrained

    def use_pinned_official_snapshot(repo_id: str, **_: Any) -> str:
        if repo_id != OFFICIAL_SOURCE_REPOSITORY:
            raise ConversionError(f"Canonical converter requested unexpected checkpoint: {repo_id}")
        return str(source_dir)

    def load_pinned_qwen_tokenizer(
        pretrained_model_name_or_path: str, *args: Any, **kwargs: Any
    ) -> Any:
        if pretrained_model_name_or_path != QWEN_TOKENIZER_REPOSITORY:
            raise ConversionError(
                "Canonical converter requested unexpected tokenizer source: "
                f"{pretrained_model_name_or_path}"
            )
        kwargs["revision"] = QWEN_TOKENIZER_REVISION
        return canonical_tokenizer_loader(pretrained_model_name_or_path, *args, **kwargs)

    converter.snapshot_download = use_pinned_official_snapshot
    converter.Qwen2TokenizerFast.from_pretrained = load_pinned_qwen_tokenizer
    try:
        converter.convert_checkpoint(
            OFFICIAL_SOURCE_REPOSITORY,
            str(output_dir),
            push_to_hub=None,
            bfloat16=True,
            max_shard_size="2GB",
        )
    finally:
        converter.snapshot_download = canonical_snapshot_download
        converter.Qwen2TokenizerFast.from_pretrained = canonical_tokenizer_loader


def assert_native_assets(output_dir: Path) -> dict[str, TensorMetadata]:
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing:
        raise ConversionError(f"Converted artifact is missing required files: {', '.join(missing)}")
    if list(output_dir.rglob("*.py")):
        raise ConversionError("Converted artifact must not include Python sidecar code.")

    config = load_json(output_dir / "config.json")
    if config.get("model_type") != "vibevoice":
        raise ConversionError("Converted config must use model_type='vibevoice'.")
    if "VibeVoiceForConditionalGeneration" not in config.get("architectures", []):
        raise ConversionError("Converted config does not declare VibeVoiceForConditionalGeneration.")

    for name in ("config.json", "processor_config.json", "tokenizer_config.json"):
        auto_map = load_json(output_dir / name).get("auto_map")
        if auto_map:
            raise ConversionError(f"{name} requests remote custom code through auto_map.")
    if load_json(output_dir / "processor_config.json").get("processor_class") != "VibeVoiceProcessor":
        raise ConversionError("Converted processor must be VibeVoiceProcessor.")
    return checkpoint_tensor_metadata(output_dir, EXPECTED_TENSOR_COUNT)


def write_model_card(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        f"""---
license: mit
library_name: transformers
pipeline_tag: text-to-speech
base_model: {OFFICIAL_SOURCE_REPOSITORY}
---

# VibeVoice 1.5B (Transformers-native)

This checkpoint is the official Transformers-native publication of
[`{OFFICIAL_SOURCE_REPOSITORY}`](https://huggingface.co/{OFFICIAL_SOURCE_REPOSITORY}).
It was converted from source revision
[`{OFFICIAL_SOURCE_REVISION}`](https://huggingface.co/{OFFICIAL_SOURCE_REPOSITORY}/tree/{OFFICIAL_SOURCE_REVISION})
with the canonical Transformers converter at
[`{TRANSFORMERS_REVISION}`](https://github.com/huggingface/transformers/commit/{TRANSFORMERS_REVISION}).

Load it with `AutoProcessor` and `AutoModelForTextToWaveform`; it does not require
`trust_remote_code` or a sidecar repository. `conversion-manifest.json` records the
complete release provenance and strict tensor-alignment result.
""",
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, mapping_sha256: str) -> None:
    manifest = {
        "format_version": 1,
        "source": {
            "repository": OFFICIAL_SOURCE_REPOSITORY,
            "revision": OFFICIAL_SOURCE_REVISION,
        },
        "canonical_converter": {
            "repository": TRANSFORMERS_REPOSITORY,
            "revision": TRANSFORMERS_REVISION,
        },
        "tokenizer_source": {
            "repository": QWEN_TOKENIZER_REPOSITORY,
            "revision": QWEN_TOKENIZER_REVISION,
        },
        "native_reference": {
            "repository": NATIVE_REFERENCE_REPOSITORY,
            "revision": NATIVE_REFERENCE_REVISION,
            "purpose": "Independent key-layout evidence only; never downloaded by this workflow.",
        },
        "tensor_alignment": {
            "source_tensor_count": EXPECTED_TENSOR_COUNT,
            "native_tensor_count": EXPECTED_TENSOR_COUNT,
            "source_to_native_mapping_sha256": mapping_sha256,
        },
        "output_files": sorted(
            path.relative_to(output_dir).as_posix()
            for path in output_dir.iterdir()
            if path.name != "conversion-manifest.json"
        ),
    }
    (output_dir / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assert_native_auto_loading(output_dir: Path) -> None:
    from transformers import AutoModelForTextToWaveform, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        output_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    if processor.__class__.__name__ != "VibeVoiceProcessor":
        raise ConversionError(f"AutoProcessor loaded {processor.__class__.__name__}, not VibeVoiceProcessor.")

    model = AutoModelForTextToWaveform.from_pretrained(
        output_dir,
        dtype="auto",
        local_files_only=True,
        trust_remote_code=False,
    )
    if model.__class__.__name__ != "VibeVoiceForConditionalGeneration":
        raise ConversionError(
            f"AutoModelForTextToWaveform loaded {model.__class__.__name__}, "
            "not VibeVoiceForConditionalGeneration."
        )
    del model
    gc.collect()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transformers-source",
        type=Path,
        required=True,
        help=f"Transformers checkout pinned to {TRANSFORMERS_REVISION}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, empty local directory for the owner-upload artifact.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ConversionError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = import_canonical_converter(args.transformers_source)
    print(f"Downloading {OFFICIAL_SOURCE_REPOSITORY}@{OFFICIAL_SOURCE_REVISION}")
    source_dir = download_official_checkpoint()
    source_metadata = checkpoint_tensor_metadata(source_dir, EXPECTED_TENSOR_COUNT)

    convert_with_canonical_tool(converter, source_dir, output_dir)
    write_model_card(output_dir)
    native_metadata = assert_native_assets(output_dir)
    mapping_sha256 = assert_tensor_metadata_matches(
        source_metadata,
        native_metadata,
        converter.map_old_key_to_new,
    )
    write_manifest(output_dir, mapping_sha256)
    assert_native_auto_loading(output_dir)
    print(f"Validated {EXPECTED_TENSOR_COUNT} tensors and wrote {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConversionError as error:
        raise SystemExit(f"Conversion failed: {error}") from error
