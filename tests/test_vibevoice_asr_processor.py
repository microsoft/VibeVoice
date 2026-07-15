import json
import math

from vibevoice.processor.vibevoice_asr_processor import (
    _resolve_speech_tok_compress_ratio,
)


def test_asr_processor_derives_compression_ratio_from_encoder_ratios(tmp_path):
    model_config_path = tmp_path / "config.json"
    processor_config_path = tmp_path / "preprocessor_config.json"

    model_config_path.write_text(
        json.dumps(
            {
                "acoustic_tokenizer_config": {
                    "encoder_ratios": [8, 5, 5, 4, 2, 2],
                },
            }
        ),
        encoding="utf-8",
    )

    processor_config_path.write_text(
        json.dumps({"speech_tok_compress_ratio": 320}),
        encoding="utf-8",
    )

    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    processor_config = json.loads(
        processor_config_path.read_text(encoding="utf-8")
    )

    expected_ratio = math.prod(
        model_config["acoustic_tokenizer_config"]["encoder_ratios"]
    )

    effective_ratio = _resolve_speech_tok_compress_ratio(
        processor_config,
        model_config,
    )

    # The processor should derive the compression ratio from the model
    # configuration rather than using a stale value from
    # preprocessor_config.json.
    assert effective_ratio == expected_ratio

    audio_samples = 34 * 60 * 24000

    expected_tokens = math.ceil(audio_samples / expected_ratio)
    stale_tokens = math.ceil(
        audio_samples / processor_config["speech_tok_compress_ratio"]
    )

    assert math.ceil(audio_samples / effective_ratio) == expected_tokens
    assert stale_tokens > expected_tokens
    assert stale_tokens == expected_tokens * 10