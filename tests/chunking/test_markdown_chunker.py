import unittest

from vibevoice.processor.chunking import MarkdownChunker


class TestMarkdownChunker(unittest.TestCase):
    def test_single_level_heading(self):
        md = """
# Chapter 1
Content for chapter 1.

# Chapter 2
Content for chapter 2.
"""
        chunker = MarkdownChunker(chunk_depth=1)
        chunks = chunker.chunk(md)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading, "Chapter 1")
        self.assertEqual(chunks[1].heading, "Chapter 2")

    def test_nested_headings(self):
        md = """
# Chapter 1
Introduction

## Section 1.1
Details

## Section 1.2
More details

# Chapter 2
New chapter
"""
        chunker = MarkdownChunker(chunk_depth=1)
        chunks = chunker.chunk(md)

        self.assertEqual(len(chunks), 2)
        self.assertIn("Section 1.1", chunks[0].content)
        self.assertIn("Section 1.2", chunks[0].content)

    def test_markdown_stripping(self):
        md = """
# Test
This is **bold** and *italic* text.
Here's a [link](http://example.com).
And `inline code`.
"""
        chunker = MarkdownChunker(strip_markdown=True)
        chunks = chunker.chunk(md)

        content = chunks[0].content
        self.assertNotIn("**", content)
        self.assertNotIn("*", content)
        self.assertNotIn("[", content)
        self.assertIn("bold", content)
        self.assertIn("italic", content)
        self.assertIn("inline code", content)

    def test_empty_sections(self):
        md = """
# Chapter 1

# Chapter 2
Some content
"""
        chunker = MarkdownChunker(chunk_depth=1)
        chunks = chunker.chunk(md)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].content.strip(), "")
        self.assertIn("Some content", chunks[1].content)

    def test_chunk_depth_2(self):
        md = """
# Part 1
## Chapter 1
Content 1
## Chapter 2
Content 2
# Part 2
## Chapter 3
Content 3
"""
        chunker = MarkdownChunker(chunk_depth=2)
        chunks = chunker.chunk(md)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].heading, "Chapter 1")
        self.assertEqual(chunks[1].heading, "Chapter 2")
        self.assertEqual(chunks[2].heading, "Chapter 3")

    def test_no_headings(self):
        md = """
This is a document
with no headings.
"""
        chunker = MarkdownChunker(chunk_depth=1)
        chunks = chunker.chunk(md)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading, "Document")
        self.assertIn("no headings", chunks[0].content)

    def test_ignore_code_fence_headings(self):
        md = """
# Title

```
# Not a heading
```

# Next
"""
        chunker = MarkdownChunker(chunk_depth=1)
        chunks = chunker.chunk(md)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading, "Title")
        self.assertEqual(chunks[1].heading, "Next")


if __name__ == "__main__":
    unittest.main()
