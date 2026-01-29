import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MarkdownChunk:
    """Represents a chunk of markdown content."""

    heading: str
    level: int
    content: str
    start_line: int


class MarkdownChunker:
    """Chunk markdown documents by heading depth."""

    def __init__(self, chunk_depth: int = 1, strip_markdown: bool = True):
        if chunk_depth < 1 or chunk_depth > 6:
            raise ValueError("chunk_depth must be between 1 and 6")
        self.chunk_depth = chunk_depth
        self.strip_markdown = strip_markdown
        self.heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def chunk(self, markdown_text: str) -> List[MarkdownChunk]:
        """Split markdown into chunks based on heading depth."""
        lines = markdown_text.splitlines()
        chunks: List[MarkdownChunk] = []
        current_chunk: Optional[MarkdownChunk] = None
        preface_lines: List[str] = []
        in_code_fence = False

        for line_num, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_fence = not in_code_fence

            match = None
            if not in_code_fence:
                match = self.heading_pattern.match(line.strip())

            if match:
                heading_level = len(match.group(1))
                heading_text = match.group(2).strip()

                if heading_level == self.chunk_depth:
                    prefix_text = None
                    if current_chunk:
                        chunks.append(current_chunk)
                    elif preface_lines:
                        prefix_text = self._preface_prefix(preface_lines)
                        if prefix_text is None:
                            preface_chunk = self._preface_chunk(preface_lines)
                            if preface_chunk:
                                chunks.append(preface_chunk)
                        preface_lines = []

                    current_chunk = MarkdownChunk(
                        heading=heading_text,
                        level=heading_level,
                        content="",
                        start_line=line_num,
                    )
                    if prefix_text:
                        current_chunk.content += prefix_text + "\n"
                else:
                    if current_chunk:
                        current_chunk.content += line + "\n"
                    else:
                        preface_lines.append(line)
            else:
                if current_chunk:
                    current_chunk.content += line + "\n"
                else:
                    if not preface_lines and line.strip() == "":
                        continue
                    preface_lines.append(line)

        if current_chunk:
            chunks.append(current_chunk)
        elif preface_lines:
            preface_chunk = self._preface_chunk(preface_lines)
            if preface_chunk:
                chunks.append(preface_chunk)

        if self.strip_markdown:
            chunks = [self._strip_markdown_formatting(chunk) for chunk in chunks]

        return chunks

    def _preface_chunk(self, preface_lines: List[str]) -> Optional[MarkdownChunk]:
        content = "\n".join(preface_lines).strip()
        if not content:
            return None
        return MarkdownChunk(heading="Document", level=0, content=content, start_line=0)

    def _preface_prefix(self, preface_lines: List[str]) -> Optional[str]:
        meaningful_lines = [line for line in preface_lines if line.strip()]
        if not meaningful_lines:
            return ""

        for line in meaningful_lines:
            match = self.heading_pattern.match(line.strip())
            if not match:
                return None
            level = len(match.group(1))
            if level >= self.chunk_depth:
                return None

        return "\n".join(preface_lines).strip("\n")

    def _strip_markdown_formatting(self, chunk: MarkdownChunk) -> MarkdownChunk:
        """Remove common markdown syntax from content."""
        content = chunk.content

        # Remove fenced code blocks
        content = re.sub(r"```[\s\S]*?```", "", content)
        content = re.sub(r"~~~[\s\S]*?~~~", "", content)

        # Inline code
        content = re.sub(r"`([^`]+)`", r"\1", content)

        # Images: ![alt](url) -> alt
        content = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", content)

        # Links: [text](url) -> text
        content = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", content)

        # Bold/italic
        content = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
        content = re.sub(r"__(.+?)__", r"\1", content)
        content = re.sub(r"\*(.+?)\*", r"\1", content)
        content = re.sub(r"_(.+?)_", r"\1", content)

        # Remaining headings
        content = re.sub(r"^#{1,6}\s+", "", content, flags=re.MULTILINE)

        # Cleanup excessive blank lines
        content = re.sub(r"\n{3,}", "\n\n", content)
        chunk.content = content.strip()
        return chunk

    def get_chunk_summary(self, chunks: List[MarkdownChunk]) -> str:
        """Generate a human-readable summary of chunks."""
        lines = [f"Total chunks: {len(chunks)}"]
        for index, chunk in enumerate(chunks, 1):
            word_count = len(chunk.content.split()) if chunk.content else 0
            heading = chunk.heading if chunk.heading else "(no heading)"
            lines.append(f"{index}. {heading} ({word_count} words)")
        return "\n".join(lines)
