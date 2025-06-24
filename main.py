import praw
import pandas as pd
from datetime import datetime
import json
import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your video creation classes - MoviePy 2.x compatible
try:
    from video_creator import VideoCreator, VideoCreatorConfig, RedditVideoHandler
    print("✅ Video creator modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import video creator: {e}")
    print("Make sure video_creator.py is in the same directory")
    raise

class RedditToVideoProcessor:
    """Integrate Reddit scraping with video creation"""
    
    def __init__(self, reddit_credentials: Dict, tts_config: Dict = None):
        """
        Initialize the processor
        
        Args:
            reddit_credentials: Dict with client_id, client_secret, user_agent
            tts_config: Dict with TTS configuration (optional)
        """
        # Initialize Reddit scraper
        self.reddit = praw.Reddit(
            client_id=reddit_credentials["client_id"],
            client_secret=reddit_credentials["client_secret"],
            user_agent=reddit_credentials["user_agent"]
        )
        
        # Initialize video creator
        self.video_config = VideoCreatorConfig()
        self.tts_config = tts_config
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_post_with_comments(self, subreddit_name: str, post_index: int = 0, 
                             sort_type: str = 'hot', max_comments: int = 10) -> tuple:
        """
        Get a specific Reddit post with its comments
        
        Args:
            subreddit_name: Name of subreddit
            post_index: Which post to get (0 = first, 1 = second, etc.)
            sort_type: 'hot', 'new', 'rising', 'top'
            max_comments: Maximum number of comments to fetch
            
        Returns:
            Tuple of (post_data, comments_list)
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get posts
            if sort_type == 'hot':
                posts = subreddit.hot(limit=post_index + 1)
            elif sort_type == 'new':
                posts = subreddit.new(limit=post_index + 1)
            elif sort_type == 'rising':
                posts = subreddit.rising(limit=post_index + 1)
            elif sort_type == 'top':
                posts = subreddit.top(limit=post_index + 1)
            else:
                raise ValueError("sort_type must be 'hot', 'new', 'rising', or 'top'")
            
            # Get the specific post
            posts_list = list(posts)
            if post_index >= len(posts_list):
                raise IndexError(f"Post index {post_index} not found. Only {len(posts_list)} posts available.")
            
            target_post = posts_list[post_index]
            
            # Extract post data
            post_data = {
                'id': target_post.id,
                'title': target_post.title,
                'author': str(target_post.author) if target_post.author else '[deleted]',
                'score': target_post.score,
                'upvote_ratio': target_post.upvote_ratio,
                'num_comments': target_post.num_comments,
                'created_utc': datetime.fromtimestamp(target_post.created_utc),
                'url': target_post.url,
                'selftext': target_post.selftext,
                'is_self': target_post.is_self,
                'permalink': f"https://reddit.com{target_post.permalink}",
                'flair': target_post.link_flair_text,
                'subreddit': str(target_post.subreddit),
                'gilded': target_post.gilded,
                'distinguished': target_post.distinguished,
                'stickied': target_post.stickied,
                'over_18': target_post.over_18,
                'spoiler': target_post.spoiler,
                'locked': target_post.locked
            }
            
            # Get comments
            comments_list = self._get_post_comments(target_post.id, max_comments)
            
            self.logger.info(f"Retrieved post: {post_data['title'][:50]}... with {len(comments_list)} comments")
            
            return post_data, comments_list
            
        except Exception as e:
            self.logger.error(f"Error getting post with comments: {e}")
            raise
    
    def _get_post_comments(self, post_id: str, limit: int = 10) -> List[Dict]:
        """Get comments from a specific post"""
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "load more comments"
            
            comments_data = []
            comment_count = 0
            
            for comment in submission.comments.list():
                if comment_count >= limit:
                    break
                    
                if hasattr(comment, 'body') and comment.body not in ['[deleted]', '[removed]']:
                    comment_data = {
                        'id': comment.id,
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
                    comment_count += 1
            
            # Sort comments by score (most upvoted first)
            comments_data.sort(key=lambda x: x['score'], reverse=True)
            
            return comments_data
            
        except Exception as e:
            self.logger.error(f"Error getting comments for post {post_id}: {e}")
            return []
    
    def create_video_from_post(self, subreddit_name: str, post_index: int = 0,
                             platform: str = "youtube_shorts", max_comments: int = 5,
                             output_filename: str = None, quality: str = "medium",
                             use_tts_for_comments: bool = True, background_video_path: str = None) -> str:
        """
        Create a video from a Reddit post with comments using subtitles
        
        Args:
            subreddit_name: Subreddit to get post from
            post_index: Which post to use (0 = first post)
            platform: Target platform for video
            max_comments: Maximum comments to include
            output_filename: Output video filename
            quality: Video quality ('low', 'medium', 'high')
            use_tts_for_comments: Whether to add TTS narration for comments
            background_video_path: Path to background video file
            
        Returns:
            Path to created video file
        """
        try:
            # Get post and comments
            self.logger.info(f"Fetching post {post_index} from r/{subreddit_name}...")
            post_data, comments_list = self.get_post_with_comments(
                subreddit_name, post_index, max_comments=max_comments
            )
            
            # Create output filename if not provided
            if not output_filename:
                safe_title = "".join(c for c in post_data['title'][:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"reddit_video_{safe_title}_{timestamp}.mp4"
            
            # Initialize video creator
            self.logger.info("Creating video with subtitles...")
            reddit_video = RedditVideoHandler(self.video_config, self.tts_config)
            
            # Set platform
            reddit_video.set_platform(platform)
            
            # Add main post TTS narration first to calculate timing
            main_tts_duration = 0
            if self.tts_config:
                narration_text = f"Here's an interesting post from r/{post_data['subreddit']}: {post_data['title']}"
                self.logger.info("Generating main post TTS with subtitles...")
                main_tts_duration, word_timings = reddit_video.add_tts_narration_with_subtitles(
                    narration_text, start_time=0
                )
            
            # Calculate total video duration dynamically
            base_duration = max(6, main_tts_duration + 1)  # At least 6 seconds for title display
            
            # Temporarily add background to get timing
            if background_video_path and os.path.exists(background_video_path):
                reddit_video.add_background_video(background_video_path, duration=120)  # Temporary duration
                self.logger.info(f"Using background video: {background_video_path}")
            else:
                reddit_video.add_background_color((20, 20, 30), duration=120)  # Temporary duration
                if background_video_path:
                    self.logger.warning(f"Background video not found: {background_video_path}, using solid color")
            
            # Add the Reddit post and comments - this returns the actual end time
            self.logger.info("Adding Reddit post and comments with subtitle timing...")
            actual_end_time = reddit_video.add_reddit_post_with_subtitles(
                post_data, 
                comments_list, 
                use_tts_for_comments=use_tts_for_comments and self.tts_config is not None
            )
            
            # Calculate the real total duration needed
            total_duration = max(actual_end_time, base_duration) + 3  # Add 3 seconds buffer
            
            # Reset and recreate with correct duration
            reddit_video.reset()
            reddit_video.set_platform(platform)
            
            # Add background with correct duration
            if background_video_path and os.path.exists(background_video_path):
                reddit_video.add_background_video(background_video_path, duration=total_duration)
            else:
                reddit_video.add_background_color((20, 20, 30), duration=total_duration)
            
            # Re-add main TTS narration with subtitles
            if self.tts_config:
                reddit_video.add_tts_narration_with_subtitles(narration_text, start_time=0)
            
            # Re-add the Reddit post and comments with subtitles
            reddit_video.add_reddit_post_with_subtitles(
                post_data, 
                comments_list, 
                use_tts_for_comments=use_tts_for_comments and self.tts_config is not None
            )
            
            # Render video
            self.logger.info(f"Rendering video to {output_filename}... (duration: {total_duration:.1f}s)")
            final_path = reddit_video.render_video(output_filename, quality=quality)
            
            # Cleanup
            reddit_video.reset()
            
            self.logger.info(f"Video created successfully: {final_path}")
            
            # Print video info
            self._print_video_info(post_data, comments_list, final_path)
            
            return final_path
            
        except Exception as e:
            self.logger.error(f"Error creating video: {e}")
            raise
    
    def _print_video_info(self, post_data: Dict, comments_list: List[Dict], video_path: str):
        """Print information about the created video"""
        print("\n" + "="*60)
        print("📹 VIDEO CREATED SUCCESSFULLY!")
        print("="*60)
        print(f"📂 File: {video_path}")
        print(f"📝 Post: {post_data['title']}")
        print(f"👤 Author: u/{post_data['author']}")
        print(f"📊 Score: {post_data['score']} ({post_data['upvote_ratio']*100:.1f}% upvoted)")
        print(f"💬 Comments included: {len(comments_list)}")
        print(f"🏷️  Subreddit: r/{post_data['subreddit']}")
        print(f"🔗 Original: {post_data['permalink']}")
        
        if comments_list:
            print(f"\n📋 Top comments included:")
            for i, comment in enumerate(comments_list[:3]):
                print(f"  {i+1}. u/{comment['author']} ({comment['score']} pts): {comment['body'][:60]}...")
        
        print("="*60)

def main():
    """Example usage with subtitle-based format and background video"""
    
    # Reddit API credentials
    reddit_creds = {
        "client_id": os.environ.get("REDDIT_CLIENT_ID"),
        "client_secret": os.environ.get("REDDIT_CLIENT_SECRET"),
        "user_agent": os.environ.get("REDDIT_USER_AGENT")
    }
    
    # TTS configuration
    tts_config = {
        "provider": "azure",
        "speech_key": os.environ.get("AZURE_SPEECH_KEY"),
        "speech_region": os.environ.get("AZURE_SPEECH_REGION"),
        "voice": "en-US-RyanMultilingualNeural"
    }
    
    # Validate that TTS config is available
    if not tts_config["speech_key"] or not tts_config["speech_region"]:
        print("⚠️  Warning: Azure Speech Services credentials not found in environment variables")
        print("   Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in your .env file")
        print("   TTS features will be disabled")
        tts_config = None
    
    # Background video path - replace with your video file
    background_video = "parkour.mp4"  # Change this to your background video path
    
    # Check if background video exists
    if not os.path.exists(background_video):
        print(f"⚠️  Background video not found: {background_video}")
        print("   Using solid color background instead")
        print("   Place your background video file in the same directory and update the path")
        background_video = None
    else:
        print(f"✅ Found background video: {background_video}")
    
    # Create processor
    processor = RedditToVideoProcessor(reddit_creds, tts_config)
    
    # Create video from post with subtitle-based comments
    try:
        video_path = processor.create_video_from_post(
            subreddit_name="askreddit",  # Good subreddit for interesting comments
            post_index=0,  # First post
            platform="youtube_shorts",
            max_comments=2,  # Reduced to 2 comments for testing and shorter videos
            quality="medium",
            use_tts_for_comments=True,  # Enable TTS with subtitles for comments
            background_video_path=background_video  # Use background video
        )
        
        print(f"\n✅ Success! Subtitle-based video saved as: {video_path}")
        print("\n🎥 Video Features:")
        print("   • Background video (or solid color if video not found)")
        print("   • Clean, readable subtitles")
        print("   • TTS narration for all comments")
        print("   • No comment bubbles - modern subtitle-only format")
        print("   • Proper audio timing with no overlaps")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()