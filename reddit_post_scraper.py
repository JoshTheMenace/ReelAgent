import praw
import pandas as pd
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initialize Reddit API client
        
        Args:
            client_id: Your Reddit app client ID
            client_secret: Your Reddit app client secret  
            user_agent: A unique identifier for your app
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
    def scrape_subreddit(self, subreddit_name, sort_type='hot', limit=100):
        """
        Scrape posts from a subreddit
        
        Args:
            subreddit_name: Name of the subreddit (without r/)
            sort_type: 'hot', 'new', 'rising', 'top'
            limit: Number of posts to scrape
            
        Returns:
            List of dictionaries containing post data
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_data = []
        
        # Get posts based on sort type
        if sort_type == 'hot':
            posts = subreddit.hot(limit=limit)
        elif sort_type == 'new':
            posts = subreddit.new(limit=limit)
        elif sort_type == 'rising':
            posts = subreddit.rising(limit=limit)
        elif sort_type == 'top':
            posts = subreddit.top(limit=limit)
        else:
            raise ValueError("sort_type must be 'hot', 'new', 'rising', or 'top'")
        
        for post in posts:
            print(post)
            post_data = {
                'title': post.title,
                'author': str(post.author) if post.author else '[deleted]',
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'url': post.url,
                'selftext': post.selftext,
                'is_self': post.is_self,
                'permalink': f"https://reddit.com{post.permalink}",
                'flair': post.link_flair_text,
                'gilded': post.gilded,
                'distinguished': post.distinguished,
                'stickied': post.stickied,
                'over_18': post.over_18,
                'spoiler': post.spoiler,
                'locked': post.locked
            }
            posts_data.append(post_data)
            
        return posts_data
    
    def save_to_csv(self, posts_data, filename):
        """Save posts data to CSV file"""
        df = pd.DataFrame(posts_data)
        df.to_csv(filename, index=False)
        print(f"Saved {len(posts_data)} posts to {filename}")
    
    def save_to_json(self, posts_data, filename):
        """Save posts data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, indent=2, default=str)
        print(f"Saved {len(posts_data)} posts to {filename}")
    
    def get_post_comments(self, post_id, limit=100):
        """
        Get comments from a specific post
        
        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to fetch
            
        Returns:
            List of comment data
        """
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)  # Remove "load more comments"
        
        comments_data = []
        for comment in submission.comments.list()[:limit]:
            if hasattr(comment, 'body'):  # Skip deleted comments
                comment_data = {
                    'body': comment.body,
                    'author': str(comment.author) if comment.author else '[deleted]',
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc),
                    'parent_id': comment.parent_id,
                    'is_submitter': comment.is_submitter,
                    'gilded': comment.gilded,
                    'distinguished': comment.distinguished,
                    'stickied': comment.stickied
                }
                comments_data.append(comment_data)
        
        return comments_data

# Example usage
def main():
    # You need to create a Reddit app to get these credentials
    # Go to https://www.reddit.com/prefs/apps/ and create a "script" app
    CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
    CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
    USER_AGENT = os.environ.get("REDDIT_USER_AGENT")
    
    # Initialize scraper
    scraper = RedditScraper(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    
    # Scrape posts from a subreddit
    try:
        posts = scraper.scrape_subreddit('python', sort_type='hot', limit=50)
        
        # Save data
        scraper.save_to_csv(posts, 'reddit_posts.csv')
        scraper.save_to_json(posts, 'reddit_posts.json')
        
        # Print some basic stats
        print(f"\nScraped {len(posts)} posts")
        print(f"Average score: {sum(post['score'] for post in posts) / len(posts):.2f}")
        print(f"Total comments: {sum(post['num_comments'] for post in posts)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()