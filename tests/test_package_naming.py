#!/usr/bin/env python3
"""
Test package naming consistency.

This test verifies that the vibevoice package follows Python packaging standards
by using consistent lowercase naming throughout.

Issue: https://github.com/microsoft/VibeVoice/issues/184
"""

import os
import sys
import unittest
from pathlib import Path


class TestPackageNaming(unittest.TestCase):
    """Test that package naming follows Python standards."""

    def test_package_importable(self):
        """Test that vibevoice package can be imported."""
        import vibevoice

        self.assertIsNotNone(vibevoice)

    def test_package_name_lowercase(self):
        """Test that package directory is lowercase."""
        # Get the repository root (parent of this test file's location)
        test_dir = Path(__file__).parent
        repo_root = test_dir.parent.parent  # tests/ -> vllm_plugin/ -> repo_root/

        # Check that vibevoice directory exists and is lowercase
        vibevoice_dir = repo_root / "vibevoice"
        self.assertTrue(
            vibevoice_dir.exists(),
            f"vibevoice directory should exist at {vibevoice_dir}",
        )
        self.assertTrue(vibevoice_dir.is_dir(), "vibevoice should be a directory")

        # Verify the directory name is lowercase
        self.assertEqual(
            vibevoice_dir.name,
            "vibevoice",
            "Package directory must be lowercase 'vibevoice'",
        )

    def test_imports_use_lowercase(self):
        """Test that all imports use lowercase vibevoice."""
        import vibevoice.modular
        import vibevoice.processor
        import vibevoice.schedule

        # Verify key modules are importable
        self.assertIsNotNone(vibevoice.modular)
        self.assertIsNotNone(vibevoice.processor)
        self.assertIsNotNone(vibevoice.schedule)


if __name__ == "__main__":
    unittest.main()
