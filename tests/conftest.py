# Lightweight test shim for environments without heavy ML dependencies.
# When running tests in environments where packages like 'torch' are not available
# we provide minimal stubs so unit tests that don't rely on full ML behavior can run.
# NOTE: The shim is intentionally gated behind an env var - set VIBEVOICE_USE_SHIMS=1 to enable it.
import sys
import os
import types

USE_SHIMS = str(os.getenv('VIBEVOICE_USE_SHIMS', '')).lower() in ['1', 'true', 'yes']

if USE_SHIMS:
    try:
        import torch  # noqa: F401
    except Exception:
        torch_stub = types.ModuleType('torch')
        # minimal helpers used by imports
        torch_stub.set_num_threads = lambda n: None
        torch_stub.device = lambda *args, **kwargs: None
        torch_stub.Tensor = object
        torch_stub.dtype = object

        # common submodules used in imports
        nn_stub = types.ModuleType('torch.nn')
        cuda_stub = types.ModuleType('torch.cuda')

        sys.modules['torch'] = torch_stub
        sys.modules['torch.nn'] = nn_stub
        sys.modules['torch.cuda'] = cuda_stub

    # Also stub torch.hub if used
    if 'torch' in sys.modules and not hasattr(sys.modules['torch'], 'hub'):
        sys.modules['torch'].hub = types.ModuleType('torch.hub')

    # Provide a minimal 'transformers' shim for environments without the package installed.
    try:
        import transformers  # noqa: F401
    except Exception:
        transformers_mod = types.ModuleType('transformers')

        token_mod = types.ModuleType('transformers.tokenization_utils_base')
        # Minimal placeholders for names used by imports
        token_mod.BatchEncoding = dict
        token_mod.PaddingStrategy = str
        token_mod.PreTokenizedInput = list
        token_mod.TextInput = str
        token_mod.TruncationStrategy = str

        utils_mod = types.ModuleType('transformers.utils')
        utils_mod.TensorType = object
        class _logging:
            @staticmethod
            def get_logger(name=None):
                class Logger:
                    def info(self, *a, **k): pass
                    def warning(self, *a, **k): pass
                    def debug(self, *a, **k): pass
                return Logger()
        utils_mod.logging = _logging()

        feat_mod = types.ModuleType('transformers.feature_extraction_utils')
        class FeatureExtractionMixin:
            pass
        feat_mod.FeatureExtractionMixin = FeatureExtractionMixin

        # expose submodules on the main transformers module and sys.modules
        transformers_mod.tokenization_utils_base = token_mod
        transformers_mod.utils = utils_mod
        transformers_mod.feature_extraction_utils = feat_mod

        sys.modules['transformers'] = transformers_mod
        sys.modules['transformers.tokenization_utils_base'] = token_mod
        sys.modules['transformers.utils'] = utils_mod
        sys.modules['transformers.feature_extraction_utils'] = feat_mod
else:
    # Shim disabled: tests that require these heavy deps should either install them or enable the shims explicitly.
    pass
