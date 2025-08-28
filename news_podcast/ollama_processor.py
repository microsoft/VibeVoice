import requests
import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class OllamaProcessor:
    """Processes news content using local Ollama LLM"""
    
    def __init__(self, base_url: str = "http://172.36.237.245:11434", model: str = "qwen2.5-coder:1.5b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
    
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Make a call to Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return None
    
    def summarize_news(self, news_items: List[Dict[str, Any]], max_items: int = 10) -> str:
        """Summarize news items into a coherent summary"""
        # Select top news items
        selected_news = news_items[:max_items]
        
        # Prepare news content for summarization
        news_content = []
        for i, item in enumerate(selected_news, 1):
            title = item.get('title', '')
            text = item.get('text', '')
            source = item.get('source', '')
            
            news_entry = f"{i}. {title} (Source: {source})"
            if text and len(text) > 20:
                news_entry += f"\\n   Summary: {text[:200]}..."
            news_content.append(news_entry)
        
        news_text = "\\n\\n".join(news_content)
        
        system_prompt = """You are a professional news analyst and podcast content creator. Your task is to analyze today's hot news and create a comprehensive summary that will be used for podcast generation."""
        
        prompt = f"""Please analyze the following hot news items and create a comprehensive summary in English:

{news_text}

Please provide:
1. A brief overview of the main themes and topics
2. Key highlights and important developments
3. Any connections or patterns between different stories
4. The potential impact or significance of these developments

Keep the summary informative, engaging, and suitable for audio content. Focus on the most important and interesting aspects."""

        return self._call_ollama(prompt, system_prompt) or "Unable to generate summary."
    
    def create_podcast_dialogue(self, news_summary: str, num_speakers: int = 2) -> str:
        """Convert news summary into a multi-speaker podcast dialogue"""
        
        system_prompt = f"""You are a professional podcast script writer. Create an engaging, natural conversation between {num_speakers} speakers discussing today's hot news.

CRITICAL FORMAT REQUIREMENTS - FOLLOW EXACTLY:
- Use EXACTLY this format: "Speaker 1:", "Speaker 2:", "Speaker 3:", "Speaker 4:" 
- NO markdown formatting (###), NO quotes around dialogue
- NO names like "Host:", "Expert:", "Sarah:", etc.
- Each speaker's dialogue should be on its own line
- Use natural conversational language with contractions, fillers, and emotional expressions
- Include natural speech patterns like "Well...", "You know", "Yeah", "Hmm", "Right?"

Example correct format:
Speaker 1: Hey there, welcome to today's news podcast!
Speaker 2: Thanks for having me. There's been quite a lot happening.
Speaker 1: Absolutely! Let's dive right in.

Guidelines:
- Make the conversation natural and engaging with real human speech patterns
- Include different perspectives and insights
- Use conversational English with natural transitions and reactions
- Each speaker should have distinct personality and viewpoints
- Include questions, agreements, disagreements, and natural dialogue flow
- Keep the content informative but accessible
- Aim for about 15-20 exchanges total
- Add natural pauses and emotional responses"""
        
        prompt = f"""Based on the following news summary, create an engaging podcast dialogue between {num_speakers} speakers:

News Summary:
{news_summary}

Create a natural conversation where the speakers discuss these topics, share insights, ask questions, and provide different perspectives. Make sure each speaker contributes meaningfully to the discussion.

REMEMBER: Use "Speaker 1:", "Speaker 2:", etc. format ONLY. Make it sound like real people talking naturally."""

        dialogue = self._call_ollama(prompt, system_prompt)
        
        if not dialogue:
            # Fallback dialogue if API fails
            dialogue = self._create_fallback_dialogue(news_summary, num_speakers)
        
        # Post-process to fix format issues
        dialogue = self._fix_dialogue_format(dialogue)
        
        return dialogue
    
    def _create_fallback_dialogue(self, news_summary: str, num_speakers: int) -> str:
        """Create a simple fallback dialogue if LLM fails"""
        return f"""Speaker 1: Welcome to today's hot news podcast! We've got some fascinating developments to discuss.

Speaker 2: Absolutely! There's been quite a lot happening in the tech and news world today.

Speaker 1: Right? Let's dive into the main stories. From what I'm seeing, we have several interesting developments across different sectors.

Speaker 2: That's right. The technology sector seems particularly active, and there are some significant global news items worth discussing.

Speaker 1: What are your thoughts on the overall trends we're seeing?

Speaker 2: I think these developments show how rapidly the digital landscape is evolving. It's fascinating to see the interconnections between different stories.

Speaker 1: Excellent point. These stories really highlight the dynamic nature of today's news cycle.

Speaker 2: Thanks for joining us today, everyone. We'll be back tomorrow with more hot news analysis!"""
    
    def _fix_dialogue_format(self, dialogue: str) -> str:
        """Fix common formatting issues in generated dialogue"""
        import re
        
        lines = dialogue.strip().split('\n')
        fixed_lines = []
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a speaker line (various formats)
            speaker_match = re.match(r'^[*#]*\s*Speaker\s*(\d+)[*]*\s*:', line)
            if speaker_match:
                # Save previous speaker if exists
                if current_speaker and current_text:
                    fixed_lines.append(f"Speaker {current_speaker}: {current_text}")
                
                # Start new speaker
                current_speaker = speaker_match.group(1)
                remaining_text = line[speaker_match.end():].strip()
                
                # Remove markdown formatting
                remaining_text = re.sub(r'^[*#\s]*', '', remaining_text)
                remaining_text = re.sub(r'[*#]*$', '', remaining_text)
                
                # Remove quotes and clean up text
                remaining_text = re.sub(r'^"(.+)"$', r'\1', remaining_text)
                
                # Remove self-introductions and names  
                remaining_text = re.sub(r"^(Hi there!?\s+)?I'm \w+[,.]?\s*", 'Hi there! ', remaining_text, flags=re.IGNORECASE)
                remaining_text = re.sub(r"^(Well,?\s+)?(as an? \w+[,.]?\s*)", r'\1', remaining_text, flags=re.IGNORECASE)
                
                current_text = remaining_text.strip()
                
            elif current_speaker and line.startswith('"') and line.endswith('"'):
                # This is dialogue text with quotes
                text = line[1:-1]  # Remove quotes
                if current_text:
                    current_text += " " + text
                else:
                    current_text = text
            elif current_speaker and not re.match(r'^[*#]*[A-Z][a-z]+[*]*:', line):
                # This is continuation text (not another speaker)
                clean_line = re.sub(r"^I'm \w+[,.]?\s*", '', line, flags=re.IGNORECASE)
                if current_text:
                    current_text += " " + clean_line
                else:
                    current_text = clean_line
        
        # Don't forget the last speaker
        if current_speaker and current_text:
            fixed_lines.append(f"Speaker {current_speaker}: {current_text}")
        
        return '\n\n'.join(fixed_lines)
    
    def enhance_for_audio(self, dialogue: str) -> str:
        """Enhance dialogue for better audio generation"""
        
        system_prompt = """You are an audio content specialist. Enhance the given dialogue to make it more suitable for text-to-speech generation by adding natural pauses, emphasis markers, and improving flow.

CRITICAL: Keep the exact "Speaker 1:", "Speaker 2:" format. Do NOT change it to any other format."""
        
        prompt = f"""Please enhance the following podcast dialogue for better audio generation:

{dialogue}

Improvements to make:
1. Add natural pauses with periods and commas
2. Ensure smooth transitions between speakers  
3. Make language more conversational and natural for speech
4. Fix any awkward phrasing that might sound odd when spoken
5. Keep the same content but improve readability for TTS
6. MAINTAIN the "Speaker 1:", "Speaker 2:" format exactly as is
7. Add natural speech fillers and expressions where appropriate

Enhanced dialogue:"""

        enhanced = self._call_ollama(prompt, system_prompt)
        return enhanced or dialogue
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_available_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model.get('name', '') for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        return []

if __name__ == "__main__":
    # Test the processor
    processor = OllamaProcessor()
    
    if processor.check_ollama_connection():
        print("✓ Ollama connection successful")
        models = processor.list_available_models()
        print(f"Available models: {models}")
        
        # Test with sample news
        sample_news = [
            {
                'title': 'AI Breakthrough in Language Models',
                'text': 'Researchers have achieved a significant breakthrough in language model efficiency',
                'source': 'Tech News',
                'score': 100
            },
            {
                'title': 'New Programming Framework Released',
                'text': 'A new framework promises to revolutionize web development',
                'source': 'Developer News',
                'score': 85
            }
        ]
        
        summary = processor.summarize_news(sample_news)
        print(f"\\nNews Summary:\\n{summary}\\n")
        
        dialogue = processor.create_podcast_dialogue(summary, num_speakers=2)
        print(f"Podcast Dialogue:\\n{dialogue}")
    else:
        print("✗ Cannot connect to Ollama. Please check the connection and URL.")