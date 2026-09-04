from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "release" / "convert_vibevoice_1_5b_hf.py"
SPEC = importlib.util.spec_from_file_location("publish_vibevoice", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PUBLISH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLISH
SPEC.loader.exec_module(PUBLISH)


def write_safetensors_header(path: Path, tensors: dict[str, dict[str, object]]) -> None:
    header = json.dumps(tensors, separators=(",", ":")).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def write_checkpoint(directory: Path, tensors: dict[str, dict[str, object]]) -> None:
    shard = "model-00001-of-00001.safetensors"
    write_safetensors_header(directory / shard, tensors)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}),
        encoding="utf-8",
    )


class PublishVibeVoiceTests(unittest.TestCase):
    def test_pinned_revisions_are_full_git_hashes(self) -> None:
        for revision in (
            PUBLISH.OFFICIAL_SOURCE_REVISION,
            PUBLISH.TRANSFORMERS_REVISION,
            PUBLISH.QWEN_TOKENIZER_REVISION,
            PUBLISH.NATIVE_REFERENCE_REVISION,
        ):
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_rejects_dirty_transformers_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkout = Path(temporary_directory)
            converter = (
                checkout
                / "src"
                / "transformers"
                / "models"
                / "vibevoice"
                / "convert_vibevoice_to_hf.py"
            )
            converter.parent.mkdir(parents=True)
            converter.touch()
            with patch.object(
                PUBLISH.subprocess,
                "run",
                side_effect=[
                    PUBLISH.subprocess.CompletedProcess(
                        args=[],
                        returncode=0,
                        stdout=f"{PUBLISH.TRANSFORMERS_REVISION}\n",
                    ),
                    PUBLISH.subprocess.CompletedProcess(
                        args=[],
                        returncode=0,
                        stdout=" M src/transformers/models/vibevoice/modular_vibevoice.py\n",
                    ),
                ],
            ):
                with self.assertRaisesRegex(PUBLISH.ConversionError, "uncommitted changes"):
                    PUBLISH.assert_transformers_revision(checkout)

    def test_rejects_untracked_transformers_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkout = Path(temporary_directory)
            converter = (
                checkout
                / "src"
                / "transformers"
                / "models"
                / "vibevoice"
                / "convert_vibevoice_to_hf.py"
            )
            converter.parent.mkdir(parents=True)
            converter.touch()
            with patch.object(
                PUBLISH.subprocess,
                "run",
                side_effect=[
                    PUBLISH.subprocess.CompletedProcess(
                        args=[],
                        returncode=0,
                        stdout=f"{PUBLISH.TRANSFORMERS_REVISION}\n",
                    ),
                    PUBLISH.subprocess.CompletedProcess(
                        args=[],
                        returncode=0,
                        stdout="?? src/transformers/sidecar.py\n",
                    ),
                ],
            ):
                with self.assertRaisesRegex(PUBLISH.ConversionError, "uncommitted changes"):
                    PUBLISH.assert_transformers_revision(checkout)

    def test_records_clean_release_tool_checkout(self) -> None:
        expected_revision = "a" * 40
        with patch.object(
            PUBLISH.subprocess,
            "run",
            side_effect=[
                PUBLISH.subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=f"{expected_revision}\n",
                ),
                PUBLISH.subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
            ],
        ):
            self.assertEqual(PUBLISH.release_tool_revision(), expected_revision)

    def test_reads_metadata_from_indexed_safetensors_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            write_checkpoint(
                checkpoint,
                {
                    "tensor_a": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
                    "tensor_b": {"dtype": "F32", "shape": [], "data_offsets": [12, 16]},
                },
            )

            metadata = PUBLISH.checkpoint_tensor_metadata(checkpoint, expected_tensor_count=2)

        self.assertEqual(metadata["tensor_a"].shape, (2, 3))
        self.assertEqual(metadata["tensor_a"].dtype, "BF16")
        self.assertEqual(metadata["tensor_b"].shape, ())

    def test_rejects_index_header_key_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            write_safetensors_header(
                checkpoint / "model-00001-of-00001.safetensors",
                {"actual": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}},
            )
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"indexed": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PUBLISH.ConversionError, "index/header mismatch"):
                PUBLISH.checkpoint_tensor_metadata(checkpoint, expected_tensor_count=1)

    def test_rejects_dtype_or_shape_mismatch(self) -> None:
        source = {"original": PUBLISH.TensorMetadata(shape=(2, 3), dtype="BF16")}
        native = {"native": PUBLISH.TensorMetadata(shape=(2, 3), dtype="F32")}

        with self.assertRaisesRegex(PUBLISH.ConversionError, "metadata mismatch"):
            PUBLISH.assert_tensor_metadata_matches(
                source,
                native,
                lambda _: "native",
                expected_tensor_count=1,
            )

    def test_rejects_key_mismatch(self) -> None:
        source = {"original": PUBLISH.TensorMetadata(shape=(2, 3), dtype="BF16")}
        native = {"other": PUBLISH.TensorMetadata(shape=(2, 3), dtype="BF16")}

        with self.assertRaisesRegex(PUBLISH.ConversionError, "Tensor key mismatch"):
            PUBLISH.assert_tensor_metadata_matches(
                source,
                native,
                lambda _: "native",
                expected_tensor_count=1,
            )

    def test_rejects_remote_code_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            for name in PUBLISH.REQUIRED_OUTPUT_FILES:
                (output / name).write_text("{}", encoding="utf-8")
            (output / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "vibevoice",
                        "architectures": ["VibeVoiceForConditionalGeneration"],
                    }
                ),
                encoding="utf-8",
            )
            (output / "processor_config.json").write_text(
                json.dumps({"processor_class": "VibeVoiceProcessor"}),
                encoding="utf-8",
            )
            (output / "tokenizer_config.json").write_text(
                json.dumps({"auto_map": {"AutoTokenizer": "untrusted.Module"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PUBLISH.ConversionError, "remote custom code"):
                PUBLISH.assert_native_assets(output)

    def test_alignment_digest_is_deterministic(self) -> None:
        source = {
            "first": PUBLISH.TensorMetadata(shape=(1,), dtype="BF16"),
            "second": PUBLISH.TensorMetadata(shape=(2,), dtype="F32"),
        }
        native = {
            "native.first": source["first"],
            "native.second": source["second"],
        }

        first = PUBLISH.assert_tensor_metadata_matches(
            source,
            native,
            lambda name: f"native.{name}",
            expected_tensor_count=2,
        )
        second = PUBLISH.assert_tensor_metadata_matches(
            dict(reversed(list(source.items()))),
            native,
            lambda name: f"native.{name}",
            expected_tensor_count=2,
        )

        self.assertEqual(first, second)

    def test_manifest_records_pinned_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "config.json").write_text("{}", encoding="utf-8")
            PUBLISH.write_manifest(output, "f" * 64, "a" * 40)
            manifest = json.loads((output / "conversion-manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["source"]["revision"],
            PUBLISH.OFFICIAL_SOURCE_REVISION,
        )
        self.assertEqual(
            manifest["canonical_converter"]["revision"],
            PUBLISH.TRANSFORMERS_REVISION,
        )
        self.assertEqual(manifest["release_tool"]["revision"], "a" * 40)
        self.assertEqual(manifest["tensor_alignment"]["source_tensor_count"], 1204)
        self.assertEqual(
            manifest["native_reference"]["revision"],
            PUBLISH.NATIVE_REFERENCE_REVISION,
        )


if __name__ == "__main__":
    unittest.main()
