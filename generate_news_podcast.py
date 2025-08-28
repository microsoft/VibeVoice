#!/usr/bin/env python3
"""
Quick start script for generating news podcasts
"""
import sys
import os

# Add the parent directory to path so we can import news_podcast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_podcast.main import main

if __name__ == "__main__":
    main()