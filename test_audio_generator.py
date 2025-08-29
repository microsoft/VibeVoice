#!/usr/bin/env python3
"""
测试修复后的 audio_generator.py 功能
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from news_podcast.audio_generator import PodcastAudioGenerator

def test_audio_generator():
    """测试音频生成器"""
    print("Testing PodcastAudioGenerator...")
    
    try:
        # 使用默认模型路径
        generator = PodcastAudioGenerator(model_path="WestZhang/VibeVoice-Large-pt")
        
        # 测试对话文本（使用 VibeVoice 格式）
        sample_dialogue = """Speaker 1: Welcome to today's tech news podcast! We have some exciting developments to discuss.
Speaker 2: Absolutely! There's been quite a lot happening in the AI world recently.
Speaker 1: Let's dive into the main stories. What caught your attention today?
Speaker 2: The latest breakthroughs in voice synthesis are particularly impressive.
Speaker 1: Great insights! Thanks for joining us today."""
        
        print("Available voices:", generator.get_available_voices())
        
        # 设置输出文件
        output_file = "/tmp/test_podcast_fixed.wav"
        
        # 生成播客音频
        print("Generating podcast audio...")
        success = generator.generate_podcast_audio(sample_dialogue, output_file)
        
        if success:
            print(f"✅ Test podcast generated successfully: {output_file}")
            # 检查文件是否存在
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"   File size: {file_size} bytes")
            else:
                print("❌ Output file not found")
        else:
            print("❌ Failed to generate test podcast")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_generator()