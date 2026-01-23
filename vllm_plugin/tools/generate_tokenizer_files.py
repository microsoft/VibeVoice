#!/usr/bin/env python3
"""
Standalone tool to generate VibeVoice tokenizer files from Qwen2 base.

Downloads base tokenizer from Qwen2 and patches it with VibeVoice-specific
audio tokens and chat template modifications.

Usage:
    python generate_tokenizer_files.py --output /path/to/output [--compare /path/to/reference]
"""

import argparse
import json
import os
import shutil
import tempfile
from typing import Optional, Dict, Any


# Qwen2.5 extended tokens (151646-151664)
# These are NOT in base Qwen2-7B but ARE in Qwen2.5 and Qwen2-VL
# VibeVoice uses some of these for speech: object_ref_start/end, box_start
QWEN25_EXTENDED_TOKENS = {
    "<|object_ref_start|>": 151646,  # Used as speech_start_id
    "<|object_ref_end|>": 151647,    # Used as speech_end_id
    "<|box_start|>": 151648,         # Used as speech_pad_id
    "<|box_end|>": 151649,
    "<|quad_start|>": 151650,
    "<|quad_end|>": 151651,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
    "<tool_call>": 151657,
    "</tool_call>": 151658,
    "<|fim_prefix|>": 151659,
    "<|fim_middle|>": 151660,
    "<|fim_suffix|>": 151661,
    "<|fim_pad|>": 151662,
    "<|repo_name|>": 151663,
    "<|file_sep|>": 151664,
}

# VibeVoice-specific audio tokens (IDs follow Qwen2.5's last token 151664)
VIBEVOICE_AUDIO_TOKENS = {
    "<|AUDIO|>": 151665,
    "<|audio_bos|>": 151666,
    "<|audio_eos|>": 151667,
}

# All extended tokens (Qwen2.5 + VibeVoice)
ALL_EXTENDED_TOKENS = {**QWEN25_EXTENDED_TOKENS, **VIBEVOICE_AUDIO_TOKENS}

# Chat template with audio support
# Key modification: handles part['type'] == 'audio' or 'audio_url' -> '<|AUDIO|>'
VIBEVOICE_CHAT_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {%- if messages[0]['content'] is string %}
            {{- messages[0]['content'] }}
        {%- else %}
            {%- for part in messages[0]['content'] %}
                {%- if part['type'] == 'text' %}
                    {{- part['text'] }}
                {%- elif part['type'] == 'audio' or part['type'] == 'audio_url' %}
                    {{- '<|AUDIO|>' }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' }}
        {%- if messages[0]['content'] is string %}
            {{- messages[0]['content'] }}
        {%- else %}
            {%- for part in messages[0]['content'] %}
                {%- if part['type'] == 'text' %}
                    {{- part['text'] }}
                {%- elif part['type'] == 'audio' or part['type'] == 'audio_url' %}
                    {{- '<|AUDIO|>' }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' }}
        {%- if message['content'] is string %}
            {{- message['content'] }}
        {%- else %}
            {%- for part in message['content'] %}
                {%- if part['type'] == 'text' %}
                    {{- part['text'] }}
                {%- elif part['type'] == 'audio' or part['type'] == 'audio_url' %}
                    {{- '<|AUDIO|>' }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""


# Default to Qwen2.5-7B which has all the extended tokens (151646-151664)
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B"


def download_qwen_tokenizer_files(output_dir: str, qwen_model: str = DEFAULT_QWEN_MODEL) -> None:
    """Download base tokenizer files from Qwen2.5 (which includes extended tokens)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
    
    files_to_download = [
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in files_to_download:
        print(f"Downloading {filename} from {qwen_model}...")
        hf_hub_download(
            repo_id=qwen_model,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )


def patch_tokenizer_config(output_dir: str) -> None:
    """
    Patch tokenizer_config.json with VibeVoice audio tokens and chat template.
    """
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 1. Add ALL extended tokens to added_tokens_decoder (Qwen2.5 + VibeVoice audio)
    if "added_tokens_decoder" not in config:
        config["added_tokens_decoder"] = {}
    
    for token, token_id in ALL_EXTENDED_TOKENS.items():
        if str(token_id) not in config["added_tokens_decoder"]:
            # Determine if token should be marked as "special" 
            # tool_call tokens are NOT special in Qwen2.5
            is_special = token not in ("<tool_call>", "</tool_call>", "<|fim_prefix|>", 
                                       "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
                                       "<|repo_name|>", "<|file_sep|>")
            config["added_tokens_decoder"][str(token_id)] = {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": is_special,
            }
    
    # 2. Add audio tokens to additional_special_tokens
    if "additional_special_tokens" not in config:
        config["additional_special_tokens"] = []
    
    for token in VIBEVOICE_AUDIO_TOKENS.keys():
        if token not in config["additional_special_tokens"]:
            config["additional_special_tokens"].append(token)
    
    # 3. Modify chat_template to support audio
    # Instead of replacing entirely, we patch the existing template to handle audio
    chat_template = config.get("chat_template", "")
    if chat_template and "<|AUDIO|>" not in chat_template:
        # Insert audio handling into the template
        # Find patterns like: {%- if part['type'] == 'text' %}
        # Add after: {%- elif part['type'] == 'audio' or part['type'] == 'audio_url' %}\n                    {{- '<|AUDIO|>' }}
        audio_handler = """{%- elif part['type'] == 'audio' or part['type'] == 'audio_url' %}
                    {{- '<|AUDIO|>' }}"""
        
        # Pattern to find: after handling 'text' type, before endif
        import re
        # Look for the pattern where we handle text type and add audio handling
        pattern = r"(\{\%- if part\['type'\] == 'text' \%\}\s*\n\s*\{\{- part\['text'\] \}\})"
        replacement = r"\1\n                " + audio_handler.replace("\n", r"\n")
        
        modified_template = re.sub(pattern, replacement, chat_template)
        
        if modified_template != chat_template:
            config["chat_template"] = modified_template
            print("  - Added audio support to existing chat_template")
        else:
            # Fallback: use our predefined template
            print("  - Warning: Could not patch existing template, using predefined template")
            config["chat_template"] = VIBEVOICE_CHAT_TEMPLATE
    
    # 4. Update model_max_length for long audio support
    config["model_max_length"] = 131072
    
    # 5. Add add_bos_token if not present
    if "add_bos_token" not in config:
        config["add_bos_token"] = False
    
    # Write back
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Patched {config_path}")


def patch_tokenizer_json(output_dir: str) -> None:
    """
    Patch tokenizer.json with VibeVoice audio tokens.
    """
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
    
    # Find existing token IDs to avoid duplicates
    existing_ids = set()
    if "added_tokens" in tokenizer:
        for token_entry in tokenizer["added_tokens"]:
            existing_ids.add(token_entry.get("id"))
    
    # Add ALL extended tokens (Qwen2.5 + VibeVoice audio)
    for token, token_id in ALL_EXTENDED_TOKENS.items():
        if token_id not in existing_ids:
            # Determine if token should be marked as "special"
            is_special = token not in ("<tool_call>", "</tool_call>", "<|fim_prefix|>", 
                                       "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
                                       "<|repo_name|>", "<|file_sep|>")
            tokenizer["added_tokens"].append({
                "id": token_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": is_special,
            })
    
    # Write back
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, indent=2, ensure_ascii=False)
    
    print(f"Patched {tokenizer_path}")


def generate_added_tokens_json(output_dir: str) -> None:
    """
    Generate added_tokens.json from tokenizer_config.json.
    """
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    added_tokens = {}
    for token_id, token_info in config.get("added_tokens_decoder", {}).items():
        content = token_info.get("content")
        if content:
            added_tokens[content] = int(token_id)
    
    output_path = os.path.join(output_dir, "added_tokens.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(added_tokens, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {output_path}")


def generate_special_tokens_map_json(output_dir: str) -> None:
    """
    Generate special_tokens_map.json with VibeVoice special tokens.
    """
    # Build the special tokens map
    special_tokens_map = {
        "additional_special_tokens": [],
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
    }
    
    # Add audio tokens as additional_special_tokens
    for token in VIBEVOICE_AUDIO_TOKENS.keys():
        special_tokens_map["additional_special_tokens"].append({
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        })
    
    # Add some commonly used special tokens
    common_special = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>"]
    for token in common_special:
        special_tokens_map["additional_special_tokens"].append({
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        })
    
    output_path = os.path.join(output_dir, "special_tokens_map.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {output_path}")


def generate_vibevoice_tokenizer_files(output_dir: str, qwen_model: str = DEFAULT_QWEN_MODEL) -> None:
    """
    Generate all 6 VibeVoice tokenizer files.
    
    Files generated:
    1. vocab.json - from Qwen2.5 (unchanged)
    2. merges.txt - from Qwen2.5 (unchanged)
    3. tokenizer.json - from Qwen2.5 + audio tokens
    4. tokenizer_config.json - from Qwen2.5 + audio tokens + chat_template
    5. added_tokens.json - generated from tokenizer_config.json
    6. special_tokens_map.json - generated with VibeVoice tokens
    """
    print(f"=== Generating VibeVoice tokenizer files to {output_dir} ===\n")
    
    # Step 1: Download base files from Qwen2
    download_qwen_tokenizer_files(output_dir, qwen_model)
    
    # Step 2: Patch tokenizer_config.json
    patch_tokenizer_config(output_dir)
    
    # Step 3: Patch tokenizer.json
    patch_tokenizer_json(output_dir)
    
    # Step 4: Generate added_tokens.json
    generate_added_tokens_json(output_dir)
    
    # Step 5: Generate special_tokens_map.json
    generate_special_tokens_map_json(output_dir)
    
    print(f"\n✅ All 6 tokenizer files generated in {output_dir}")


def compare_json_files(file1: str, file2: str, name: str) -> Dict[str, Any]:
    """Compare two JSON files and return differences."""
    result = {
        "name": name,
        "identical": False,
        "differences": [],
    }
    
    if not os.path.exists(file1):
        result["differences"].append(f"File 1 not found: {file1}")
        return result
    
    if not os.path.exists(file2):
        result["differences"].append(f"File 2 not found: {file2}")
        return result
    
    with open(file1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    
    with open(file2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    
    if data1 == data2:
        result["identical"] = True
        return result
    
    # Find specific differences
    def find_diff(d1, d2, path=""):
        diffs = []
        if isinstance(d1, dict) and isinstance(d2, dict):
            all_keys = set(d1.keys()) | set(d2.keys())
            for k in all_keys:
                new_path = f"{path}.{k}" if path else k
                if k not in d1:
                    diffs.append(f"Missing in generated: {new_path}")
                elif k not in d2:
                    diffs.append(f"Extra in generated: {new_path}")
                else:
                    diffs.extend(find_diff(d1[k], d2[k], new_path))
        elif isinstance(d1, list) and isinstance(d2, list):
            if len(d1) != len(d2):
                diffs.append(f"{path}: list length differs ({len(d1)} vs {len(d2)})")
            # For lists, just check if they're equal (detailed diff is complex)
            if d1 != d2:
                diffs.append(f"{path}: list content differs")
        elif d1 != d2:
            # Truncate long values for readability
            v1 = str(d1)[:100] + "..." if len(str(d1)) > 100 else str(d1)
            v2 = str(d2)[:100] + "..." if len(str(d2)) > 100 else str(d2)
            diffs.append(f"{path}: '{v1}' vs '{v2}'")
        return diffs
    
    result["differences"] = find_diff(data1, data2)
    return result


def compare_text_files(file1: str, file2: str, name: str) -> Dict[str, Any]:
    """Compare two text files."""
    result = {
        "name": name,
        "identical": False,
        "differences": [],
    }
    
    if not os.path.exists(file1):
        result["differences"].append(f"File 1 not found: {file1}")
        return result
    
    if not os.path.exists(file2):
        result["differences"].append(f"File 2 not found: {file2}")
        return result
    
    with open(file1, "r", encoding="utf-8") as f:
        content1 = f.read()
    
    with open(file2, "r", encoding="utf-8") as f:
        content2 = f.read()
    
    if content1 == content2:
        result["identical"] = True
    else:
        lines1 = content1.splitlines()
        lines2 = content2.splitlines()
        result["differences"].append(f"Line count: {len(lines1)} vs {len(lines2)}")
        
        # Find first difference
        for i, (l1, l2) in enumerate(zip(lines1, lines2)):
            if l1 != l2:
                result["differences"].append(f"First diff at line {i+1}")
                break
    
    return result


def compare_with_reference(generated_dir: str, reference_dir: str) -> None:
    """Compare generated files with reference files."""
    print(f"\n=== Comparing generated files with reference ===")
    print(f"Generated: {generated_dir}")
    print(f"Reference: {reference_dir}\n")
    
    files_to_compare = [
        ("vocab.json", "json"),
        ("merges.txt", "text"),
        ("tokenizer.json", "json"),
        ("tokenizer_config.json", "json"),
        ("added_tokens.json", "json"),
        ("special_tokens_map.json", "json"),
    ]
    
    all_identical = True
    
    for filename, file_type in files_to_compare:
        gen_file = os.path.join(generated_dir, filename)
        ref_file = os.path.join(reference_dir, filename)
        
        if file_type == "json":
            result = compare_json_files(gen_file, ref_file, filename)
        else:
            result = compare_text_files(gen_file, ref_file, filename)
        
        if result["identical"]:
            print(f"✅ {filename}: IDENTICAL")
        else:
            print(f"❌ {filename}: DIFFERENT")
            for diff in result["differences"][:5]:  # Show first 5 differences
                print(f"   - {diff}")
            if len(result["differences"]) > 5:
                print(f"   ... and {len(result['differences']) - 5} more differences")
            all_identical = False
    
    print()
    if all_identical:
        print("🎉 All files are identical!")
    else:
        print("⚠️  Some files have differences. See details above.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VibeVoice tokenizer files from Qwen2 base"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for generated files (default: temp directory)"
    )
    parser.add_argument(
        "--compare", "-c",
        type=str,
        default=None,
        help="Reference directory to compare generated files against"
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default=DEFAULT_QWEN_MODEL,
        help=f"Qwen model to download base tokenizer from (default: {DEFAULT_QWEN_MODEL})"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output:
        output_dir = args.output
        cleanup = False
    else:
        output_dir = tempfile.mkdtemp(prefix="vibevoice_tokenizer_")
        cleanup = not args.compare  # Only cleanup if not comparing
    
    try:
        # Generate files
        generate_vibevoice_tokenizer_files(output_dir, args.qwen_model)
        
        # Compare if requested
        if args.compare:
            compare_with_reference(output_dir, args.compare)
        
        if not args.output:
            print(f"\nGenerated files are in: {output_dir}")
            
    finally:
        if cleanup and not args.output:
            print(f"\nCleaning up temporary directory: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
