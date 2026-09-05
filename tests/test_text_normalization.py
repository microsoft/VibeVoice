import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROCESSOR_PATH = (
    Path(__file__).resolve().parent.parent
    / "vibevoice"
    / "processor"
    / "vibevoice_processor.py"
)
MODULE_NAME = "vibevoice.processor.vibevoice_processor_for_test"
CHINESE_SCRIPT = "Speaker 1:你好，世界。她问：“今天好吗？”我说：‘很好！’"
NORMALIZED_SCRIPT = [(0, ' 你好,世界.她问:"今天好吗?"我说:\'很好!\'')]


def stub_module(name, **attributes):
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


def load_processor_module():
    numpy = stub_module("numpy", ndarray=object)
    torch = stub_module("torch", Tensor=object, device=object, dtype=object)
    tokenization = stub_module(
        "transformers.tokenization_utils_base",
        BatchEncoding=dict,
        PaddingStrategy=object,
        PreTokenizedInput=str,
        TextInput=str,
        TruncationStrategy=object,
    )
    transformers_logging = stub_module(
        "transformers.utils.logging",
        get_logger=lambda _name: Mock(),
    )
    transformers_utils = stub_module(
        "transformers.utils",
        TensorType=object,
        logging=transformers_logging,
    )
    tokenizer_processor = stub_module(
        "vibevoice.processor.vibevoice_tokenizer_processor",
        AudioNormalizer=object,
    )
    dependency_stubs = {
        "numpy": numpy,
        "torch": torch,
        "transformers": stub_module("transformers"),
        "transformers.tokenization_utils_base": tokenization,
        "transformers.utils": transformers_utils,
        "transformers.utils.logging": transformers_logging,
        "vibevoice": stub_module("vibevoice"),
        "vibevoice.processor": stub_module("vibevoice.processor"),
        "vibevoice.processor.vibevoice_tokenizer_processor": tokenizer_processor,
    }

    spec = importlib.util.spec_from_file_location(MODULE_NAME, PROCESSOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load processor module from {PROCESSOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependency_stubs):
        spec.loader.exec_module(module)
    return module


class TextNormalizationTest(unittest.TestCase):
    def test_normalizes_chinese_punctuation_for_synthesis(self):
        module = load_processor_module()
        processor = module.VibeVoiceProcessor.__new__(module.VibeVoiceProcessor)

        self.assertEqual(
            processor._parse_script(CHINESE_SCRIPT),
            NORMALIZED_SCRIPT,
        )


if __name__ == "__main__":
    unittest.main()
