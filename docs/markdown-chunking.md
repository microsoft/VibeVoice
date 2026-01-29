# Markdown Chunking for VibeVoice-Narrator

This document describes the Markdown chunking functionality used in VibeVoice-Narrator for converting Markdown documents to speech.

## Overview

The Markdown chunking system automatically splits Markdown documents into logical segments based on heading hierarchy. This enables:

- **Natural Speech Segmentation**: Breaks long documents at natural boundaries (headings)
- **Context Preservation**: Each chunk maintains its heading context for better speech synthesis
- **Configurable Depth**: Control how deep to chunk (H1, H2, H3, etc.)
- **Markdown Stripping**: Optional removal of markdown formatting for cleaner TTS input

## Usage

### Python API

```python
from vibevoice.processor.chunking import MarkdownChunker, MarkdownChunk
import re

# Create chunker with depth 1 (split by H1 headings)
chunker = MarkdownChunker(chunk_depth=1)

# Chunk a markdown document
with open("document.md", "r") as f:
    markdown_text = f.read()
    chunks = chunker.chunk(markdown_text)

# Access chunk information
for chunk in chunks:
    print(f"Heading: {chunk.heading}")
    print(f"Content: {chunk.content[:100]}...")
    word_count = len(re.findall(r"\w+", chunk.content))
    print(f"Word count: {word_count}")
```

### Command Line

```bash
# Use with demo script
python demo/chunked_markdown_tts_realtime.py \
    --markdown document.md \
    --depth 3 \
    --speaker Emma
```

## Chunker Parameters

| Parameter | Type | Default | Description |
|-----------|------|-------------|-------------|
| `chunk_depth` | int | `1` | Heading level to split on; a new chunk is created only for headings whose level equals this value. See **chunk_depth behavior** below for details. |
| `strip_markdown` | bool | `True` | Remove markdown formatting from content |
| `min_chunk_size` | int | `None` | Minimum words per chunk (None=disabled) |

### chunk_depth behavior

The `chunk_depth` parameter determines which heading levels start new chunks and how material before the first matching heading is treated:

- A new chunk is emitted only when a heading's level equals `chunk_depth`. Lower-level headings (numeric level greater than `chunk_depth`, such as H2 when `chunk_depth=1`) remain inside the current chunk's content rather than starting new chunks.
- If content appears before the first heading at the configured level, it is emitted as a `Document` preface chunk (heading `Document`, level 0) when `chunk_depth` &gt; 1. For example, with `chunk_depth=2` content before the first H2 becomes a preface chunk; with `chunk_depth=1`, H1 headings start chunks and preface behavior is not applicable.
- This design preserves higher-level context within chunks and gives fine-grained control over chunk granularity for TTS.

## Chunk Structure

Each [`MarkdownChunk`](../vibevoice/processor/chunking/markdown_chunker.py) contains:

- **`heading`**: The heading text (e.g., "Chapter 1", "Introduction")
- **`content`**: The markdown content under the heading
- **`level`**: The heading level (1 for H1, 2 for H2, etc.)

## Example

### Input Markdown

```markdown
# Introduction

Welcome to VibeVoice-Narrator. This tool converts Markdown documents to speech.

## Features

- Markdown chunking
- Multi-speaker support
- Real-time streaming

## Getting Started

Install the package and start converting your documents.
```

### Chunks Generated (chunk_depth=1)

**Note**: These examples show output with `strip_markdown=False` to illustrate chunk boundaries. With the default `strip_markdown=True`, heading markers (e.g., `##`) are removed from chunk content.

When `chunk_depth=1`, the chunker creates chunks only for H1 headings. Lower-level headings (H2/H3) are kept inside the H1 chunk's content. For the example input above, this produces a single H1 chunk that contains the H1 introduction text and the two H2 sections as part of its content:

1. **Heading**: "Introduction"
   - **Content**: "Welcome to VibeVoice-Narrator. This tool converts Markdown documents to speech.\n\n## Features\n\n- Markdown chunking\n- Multi-speaker support\n- Real-time streaming\n\n## Getting Started\n\nInstall the package and start converting your documents." 
   - **Level**: 1

### Chunks Generated (chunk_depth=2)

**Note**: These examples show output with `strip_markdown=False` to illustrate chunk boundaries. With the default `strip_markdown=True`, heading markers (e.g., `#`, `##`) are removed from chunk content.

When `chunk_depth=2`, the chunker creates chunks for H2 headings. Content that appears before the first H2 (including the H1 heading and its paragraph) is emitted as a preface chunk with heading `Document` (level 0). The H2 headings then become separate chunks:

1. **Heading**: "Document"
   - **Content**: "# Introduction\n\nWelcome to VibeVoice-Narrator. This tool converts Markdown documents to speech."
   - **Level**: 0

2. **Heading**: "Features"
   - **Content**: "- Markdown chunking\n- Multi-speaker support\n- Real-time streaming"
   - **Level**: 2

3. **Heading**: "Getting Started"
   - **Content**: "Install the package and start converting your documents."
   - **Level**: 2
```

## Advanced Usage

### Custom Chunk Depth

```python
# Chunk by H2 headings (sections)
chunker = MarkdownChunker(chunk_depth=2)

# Chunk by H3 headings (subsections)
chunker = MarkdownChunker(chunk_depth=3)
```

### Markdown Stripping

```python
# Keep markdown formatting
chunker = MarkdownChunker(strip_markdown=False)

# Remove markdown formatting for cleaner TTS
chunker = MarkdownChunker(strip_markdown=True)
```

### Getting Chunk Summary

```python
summary = chunker.get_chunk_summary(chunks)
print(summary)
# Output:
# Total chunks: 3
# Total words: 150
# Average words per chunk: 50
```

## Implementation Details

The [`MarkdownChunker`](../vibevoice/processor/chunking/markdown_chunker.py) class:

- Parses Markdown using regex to identify headings
- Handles nested heading structures
- Supports code fence detection (ignores headings in code blocks)
- Provides word count and character count for each chunk
- Generates summary statistics for the entire document

## Notes

- Code fences (```) are properly handled - headings inside are ignored
- Empty sections are preserved as valid chunks
- Documents without headings are treated as a single chunk with heading "Document"
