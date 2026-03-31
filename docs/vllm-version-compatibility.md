# vLLM Version Compatibility

VibeVoice's vLLM plugin (`vllm_plugin/model.py`) tracks vLLM's evolving multimodal API surface. This document describes the known breaking changes across vLLM releases and how the plugin handles them.

## Supported vLLM Versions

| vLLM Version | Status | Notes |
|---|---|---|
| 0.14.x | ✅ Supported | Original target version. Backward compat shim applied at import time. |
| 0.16.x | ✅ Supported | Requires `get_data_parser` on `ProcessingInfo` instead of `_get_data_parser` on `Processor`. |
| 0.18.x | ✅ Supported | `ProcessorInputs` constructor signature changed; base class handles it correctly. |

## API Changes

### vLLM 0.16: `_get_data_parser` → `get_data_parser`

In vLLM 0.16, the multimodal data parser was moved from the processor to the processing info class:

- **Before (0.14):** `BaseMultiModalProcessor._get_data_parser()` — override on the processor class.
- **After (0.16):** `BaseProcessingInfo.get_data_parser()` — override on the processing info class.

vLLM 0.16 raises `ValueError` if `_get_data_parser` is still defined on the processor:

```
ValueError: BaseMultiModalProcessor._get_data_parser has been moved to
BaseProcessingInfo.build_data_parser in v0.16
```

**Our fix:** The canonical `get_data_parser()` now lives on `VibeVoiceProcessingInfo`. For vLLM < 0.16, a monkey-patch adds `_get_data_parser` back onto `VibeVoiceMultiModalProcessor` at import time via version detection.

### vLLM 0.18: `ProcessorInputs` signature change

In vLLM 0.18, `ProcessorInputs.__init__()` no longer accepts the `mm_data` keyword argument (replaced by `mm_data_items` internally):

```
TypeError: ProcessorInputs.__init__() got an unexpected keyword argument 'mm_data'
```

**Our fix:** We removed the custom `get_dummy_processor_inputs` override from `VibeVoiceDummyInputsBuilder`. The base class (`BaseDummyInputsBuilder`) already constructs `ProcessorInputs` correctly for both 0.14 and 0.16+ by using `parse_mm_data()` internally. Our `get_dummy_text()` and `get_dummy_mm_data()` methods are still called by the base implementation.

## Test Verification

Tested on:
- **vLLM:** 0.16.0
- **GPU:** NVIDIA L40S (single card)
- **Workload:** 20-minute audio file
- **Result:** Completed in 197 seconds
- **Quality:** Output matches 0.14.x baseline
