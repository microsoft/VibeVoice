#!/usr/bin/env python3
"""
Test script for news podcast generation (text-only, no audio)
"""
import sys
import os
from datetime import datetime

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_podcast.news_fetcher import NewsFetcher
from news_podcast.ollama_processor import OllamaProcessor

def test_news_pipeline():
    """Test the news processing pipeline without audio generation"""
    
    print("🗞️ Testing News Podcast Pipeline (Text Only)")
    print("=" * 50)
    
    try:
        # Step 1: Fetch news
        print("1. Fetching today's hot news...")
        fetcher = NewsFetcher()
        news = fetcher.get_today_hot_news(8)
        
        if not news:
            print("❌ No news found!")
            return False
        
        print(f"✅ Found {len(news)} news items:")
        for i, item in enumerate(news, 1):
            print(f"   {i}. {item['title'][:60]}... ({item['source']}, Score: {item['score']})")
        
        # Step 2: Process with LLM
        print("\n2. Processing news with Ollama LLM...")
        processor = OllamaProcessor()
        
        if not processor.check_ollama_connection():
            print("❌ Cannot connect to Ollama!")
            return False
        
        print("   - Generating news summary...")
        summary = processor.summarize_news(news, max_items=6)
        
        if not summary:
            print("❌ Summary generation failed!")
            return False
        
        print(f"   ✅ Summary generated ({len(summary)} chars)")
        
        # Step 3: Create dialogue
        print("   - Creating podcast dialogue...")
        dialogue = processor.create_podcast_dialogue(summary, num_speakers=2)
        
        if not dialogue:
            print("❌ Dialogue generation failed!")
            return False
        
        lines = [line for line in dialogue.split('\n') if ':' in line and line.strip()]
        print(f"   ✅ Dialogue created ({len(lines)} dialogue lines)")
        
        # Step 4: Enhance dialogue
        print("   - Enhancing dialogue for audio...")
        enhanced_dialogue = processor.enhance_for_audio(dialogue)
        
        if enhanced_dialogue:
            print("   ✅ Dialogue enhanced")
        else:
            enhanced_dialogue = dialogue
            print("   ⚠️ Using original dialogue")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dialogue
        dialogue_file = f"{output_dir}/test_dialogue_{timestamp}.txt"
        with open(dialogue_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_dialogue)
        
        # Save news data
        import json
        news_file = f"{output_dir}/test_news_{timestamp}.json"
        with open(news_file, 'w', encoding='utf-8') as f:
            json.dump(news, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Files saved:")
        print(f"   - Dialogue: {dialogue_file}")
        print(f"   - News data: {news_file}")
        
        # Show dialogue preview
        print(f"\n🎙️ Dialogue Preview:")
        print("-" * 30)
        preview_lines = enhanced_dialogue.split('\n')[:10]
        for line in preview_lines:
            if line.strip():
                print(line)
        if len(enhanced_dialogue.split('\n')) > 10:
            print("...")
        
        print("\n✅ Test completed successfully!")
        print("\nTo generate audio, run:")
        print("python generate_news_podcast.py --model-path microsoft/VibeVoice-1.5B")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_news_pipeline()
    sys.exit(0 if success else 1)