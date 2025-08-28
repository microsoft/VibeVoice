import requests
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class NewsFetcher:
    """Fetches hot news from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_hacker_news_top(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch top stories from Hacker News"""
        try:
            # Get top story IDs
            response = self.session.get('https://hacker-news.firebaseio.com/v0/topstories.json')
            story_ids = response.json()[:limit]
            
            stories = []
            for story_id in story_ids:
                story_response = self.session.get(f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json')
                story_data = story_response.json()
                
                if story_data and story_data.get('title'):
                    stories.append({
                        'title': story_data.get('title', ''),
                        'url': story_data.get('url', ''),
                        'score': story_data.get('score', 0),
                        'text': story_data.get('text', ''),
                        'source': 'Hacker News'
                    })
            
            return stories
            
        except Exception as e:
            logger.error(f"Error fetching Hacker News: {e}")
            return []
    
    def fetch_reddit_hot(self, subreddit: str = 'worldnews', limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch hot posts from Reddit"""
        try:
            url = f'https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}'
            response = self.session.get(url)
            data = response.json()
            
            stories = []
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                stories.append({
                    'title': post_data.get('title', ''),
                    'url': post_data.get('url', ''),
                    'score': post_data.get('score', 0),
                    'text': post_data.get('selftext', ''),
                    'source': f'Reddit r/{subreddit}'
                })
            
            return stories
            
        except Exception as e:
            logger.error(f"Error fetching Reddit: {e}")
            return []
    
    def fetch_github_trending(self, language: str = '', limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch trending repositories from GitHub"""
        try:
            # Use GitHub's search API for trending repos
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            date_filter = week_ago.strftime('%Y-%m-%d')
            
            query = f"created:>{date_filter}"
            if language:
                query += f" language:{language}"
            
            url = f'https://api.github.com/search/repositories'
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': limit
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            stories = []
            for repo in data.get('items', []):
                stories.append({
                    'title': f"{repo.get('name', '')} - {repo.get('description', '')}",
                    'url': repo.get('html_url', ''),
                    'score': repo.get('stargazers_count', 0),
                    'text': repo.get('description', ''),
                    'source': 'GitHub Trending'
                })
            
            return stories
            
        except Exception as e:
            logger.error(f"Error fetching GitHub trending: {e}")
            return []
    
    def fetch_all_news(self, limit_per_source: int = 5) -> List[Dict[str, Any]]:
        """Fetch news from all sources"""
        all_news = []
        
        # Fetch from different sources
        hacker_news = self.fetch_hacker_news_top(limit_per_source)
        reddit_tech = self.fetch_reddit_hot('technology', limit_per_source)
        reddit_world = self.fetch_reddit_hot('worldnews', limit_per_source)
        github_trending = self.fetch_github_trending(limit=limit_per_source)
        
        all_news.extend(hacker_news)
        all_news.extend(reddit_tech)
        all_news.extend(reddit_world)
        all_news.extend(github_trending)
        
        # Sort by score/popularity
        all_news.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return all_news
    
    def get_today_hot_news(self, total_limit: int = 20) -> List[Dict[str, Any]]:
        """Get today's hot news with deduplication"""
        news = self.fetch_all_news()
        
        # Simple deduplication based on title similarity
        unique_news = []
        seen_titles = set()
        
        for item in news:
            title_lower = item.get('title', '').lower()
            # Simple check for similar titles
            is_duplicate = any(
                self._similar_titles(title_lower, seen_title) 
                for seen_title in seen_titles
            )
            
            if not is_duplicate and len(unique_news) < total_limit:
                unique_news.append(item)
                seen_titles.add(title_lower)
        
        return unique_news
    
    def _similar_titles(self, title1: str, title2: str, threshold: float = 0.7) -> bool:
        """Check if two titles are similar"""
        # Simple word-based similarity check
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

if __name__ == "__main__":
    # Test the news fetcher
    fetcher = NewsFetcher()
    news = fetcher.get_today_hot_news(10)
    
    for i, item in enumerate(news, 1):
        print(f"{i}. {item['title']} (Source: {item['source']}, Score: {item['score']})")
        if item.get('text'):
            print(f"   {item['text'][:100]}...")
        print()