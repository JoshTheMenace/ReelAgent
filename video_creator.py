import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
from dotenv import load_dotenv
load_dotenv()

# Video processing - MoviePy 2.x compatible imports
try:
    # Try MoviePy 2.x imports first
    from moviepy import VideoFileClip, AudioFileClip, TextClip, ColorClip, CompositeVideoClip, CompositeAudioClip, ImageClip
    from moviepy import concatenate_videoclips, concatenate_audioclips
    # from moviepy.config import check_dependencies
    print("✅ MoviePy 2.x imported successfully")
    MOVIEPY_VERSION = 2
except ImportError as e1:
    try:
        # Fallback to MoviePy 1.x imports
        from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, ColorClip, CompositeVideoClip, CompositeAudioClip, ImageClip
        from moviepy.editor import concatenate_videoclips, concatenate_audioclips
        from moviepy.config import check_dependencies
        print("✅ MoviePy 1.x imported successfully")
        MOVIEPY_VERSION = 1
    except ImportError as e2:
        print(f"❌ Failed to import MoviePy 2.x: {e1}")
        print(f"❌ Failed to import MoviePy 1.x: {e2}")
        raise ImportError("MoviePy not properly installed or incompatible version")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

# Text-to-speech
import azure.cognitiveservices.speech as speechsdk
from io import BytesIO

class VideoCreatorConfig:
    """Configuration management for video creation"""
    
    DEFAULT_CONFIG = {
        "video": {
            "default_fps": 30,
            "default_duration": 60,
            "fade_duration": 0.5,
            "background_color": (0, 0, 0)
        },
        "audio": {
            "sample_rate": 44100,
            "background_volume": 0.3,
            "tts_volume": 1.0,
            "fade_in_duration": 1.0,
            "fade_out_duration": 1.0
        },
        "text": {
            "default_font": "arial.ttf",
            "default_size": 48,
            "default_color": (255, 255, 255),
            "margin": 50,
            "line_height": 1.2
        },
        "platforms": {
            "youtube": {"width": 1920, "height": 1080, "fps": 30},
            "youtube_shorts": {"width": 1080, "height": 1920, "fps": 30},
            "tiktok": {"width": 1080, "height": 1920, "fps": 30},
            "instagram": {"width": 1080, "height": 1080, "fps": 30}
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self._merge_config(custom_config)
    
    def _merge_config(self, custom_config: Dict):
        """Recursively merge custom config with defaults"""
        def merge_dict(base, custom):
            for key, value in custom.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        merge_dict(self.config, custom_config)
    
    def get(self, path: str, default=None):
        """Get config value using dot notation (e.g., 'video.default_fps')"""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

class TextToSpeechEngine:
    """Text-to-speech engine supporting multiple providers"""
    
    def __init__(self, provider: str = "azure", **kwargs):
        self.provider = provider
        self.config = kwargs
        
        if provider == "azure":
            self.speech_key = kwargs.get("speech_key")
            self.speech_region = kwargs.get("speech_region")
            if self.speech_key and self.speech_region:
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=self.speech_key, 
                    region=self.speech_region
                )
                self.speech_config.speech_synthesis_voice_name = kwargs.get("voice", "en-US-AriaNeural")
    
    def text_to_audio(self, text: str, output_path: str = None) -> str:
        """Convert text to audio file"""
        if not output_path:
            output_path = tempfile.mktemp(suffix=".wav")
        
        if self.provider == "azure" and hasattr(self, 'speech_config'):
            try:
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=self.speech_config,
                    audio_config=speechsdk.audio.AudioOutputConfig(filename=output_path)
                )
                result = synthesizer.speak_text_async(text).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    return output_path
                else:
                    raise Exception(f"TTS failed: {result.reason}")
            except Exception as e:
                logging.warning(f"Azure TTS failed: {e}, falling back to silent audio")
                return self._create_silent_audio(len(text) * 0.1, output_path)
        else:
            # Fallback: create silent audio as placeholder
            return self._create_silent_audio(len(text) * 0.1, output_path)
    
    def _create_silent_audio(self, duration: float, output_path: str) -> str:
        """Create silent audio as fallback"""
        try:
            # Create silent audio clip
            from moviepy import AudioClip
            silent_clip = AudioClip(lambda t: [0, 0], duration=duration)
            silent_clip.write_audiofile(output_path, verbose=False, logger=None)
            silent_clip.close()
        except ImportError:
            # Fallback method using numpy
            import numpy as np
            import wave
            
            sample_rate = 44100
            samples = int(duration * sample_rate)
            audio_data = np.zeros((samples, 2), dtype=np.int16)
            
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
        
        return output_path

class VideoCreator:
    """Core video creation engine"""
    
    def __init__(self, config: VideoCreatorConfig = None, tts_config: Dict = None):
        self.config = config or VideoCreatorConfig()
        self.logger = self._setup_logging()
        
        # Initialize TTS
        self.tts_engine = None
        if tts_config:
            self.tts_engine = TextToSpeechEngine(**tts_config)
        
        # Track temporary files for cleanup
        self.temp_files = []
        
        # Video composition elements
        self.video_clips = []
        self.audio_clips = []
        self.text_clips = []
        
        # Platform settings
        self.platform = "youtube"
        self.resolution = (1920, 1080)
        self.fps = 30
    
    def _setup_logging(self):
        """Setup logging"""
        logger = logging.getLogger("VideoCreator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - VideoCreator - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def set_platform(self, platform: str):
        """Set target platform and adjust settings"""
        if platform in self.config.get("platforms", {}):
            platform_config = self.config.get(f"platforms.{platform}")
            self.platform = platform
            self.resolution = (platform_config["width"], platform_config["height"])
            self.fps = platform_config["fps"]
            self.logger.info(f"Set platform to {platform}: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
        else:
            self.logger.warning(f"Unknown platform: {platform}")
    
    def add_background_video(self, video_path: str, duration: Optional[float] = None):
        """Add background video"""
        try:
            video = VideoFileClip(video_path)
            
            if duration and video.duration > duration:
                try:
                    video = video.subclipped(0, duration)  # MoviePy 2.x
                except AttributeError:
                    video = video.subclip(0, duration)     # MoviePy 1.x
            elif duration and video.duration < duration:
                # Loop video to match duration
                loops_needed = int(duration / video.duration) + 1
                video = concatenate_videoclips([video] * loops_needed)
                try:
                    video = video.subclipped(0, duration)  # MoviePy 2.x
                except AttributeError:
                    video = video.subclip(0, duration)     # MoviePy 1.x
            
            # Resize to match target resolution
            try:
                video = video.resized(self.resolution)     # MoviePy 2.x
            except AttributeError:
                video = video.resize(self.resolution)      # MoviePy 1.x
            
            self.video_clips.append(video)
            self.logger.info(f"Added background video: {video_path}")
            
        except Exception as e:
            self.logger.error(f"Error adding background video: {e}")
    
    def add_background_color(self, color: Tuple[int, int, int] = None, duration: float = 60):
        """Add solid color background"""
        if not color:
            color = self.config.get("video.background_color", (0, 0, 0))
        
        # Create colored background
        color_clip = ColorClip(size=self.resolution, color=color, duration=duration)
        self.video_clips.append(color_clip)
        self.logger.info(f"Added color background: {color}")
    
    def _create_text_clip_safe(self, text: str, font_size: int, color: str = 'white'):
        """Safely create a TextClip with version compatibility"""
        try:
            # Try MoviePy 2.x first - use only keyword arguments to avoid conflicts
            if MOVIEPY_VERSION == 2:
                text_clip = TextClip(
                    text=text,
                    font_size=font_size,
                    color=color
                )
            else:
                # MoviePy 1.x
                text_clip = TextClip(
                    txt=text,
                    fontsize=font_size,
                    color=color
                )
            return text_clip
        except Exception as e:
            self.logger.warning(f"Failed to create text clip with preferred method: {e}")
            try:
                # Fallback - try minimal parameters
                if MOVIEPY_VERSION == 2:
                    text_clip = TextClip(text, font_size=font_size)
                else:
                    text_clip = TextClip(text, fontsize=font_size)
                return text_clip
            except Exception as e2:
                self.logger.error(f"Failed to create text clip with fallback: {e2}")
                raise e2
    
    def add_text_overlay(self, text: str, start_time: float = 0, duration: float = None,
                        position: Union[str, Tuple] = 'center', font_size: int = None,
                        color: Tuple[int, int, int] = None, animation: str = None):
        """Add text overlay to video"""
        
        font_size = font_size or self.config.get("text.default_size", 48)
        color = color or self.config.get("text.default_color", (255, 255, 255))
        
        try:
            # Create text clip using safe method
            text_clip = self._create_text_clip_safe(text, font_size, 'white')
            
            # Set duration
            if duration:
                try:
                    text_clip = text_clip.with_duration(duration)  # MoviePy 2.x
                except AttributeError:
                    text_clip = text_clip.set_duration(duration)   # MoviePy 1.x
            
            # Set position
            try:
                text_clip = text_clip.with_position(position)  # MoviePy 2.x
            except AttributeError:
                text_clip = text_clip.set_position(position)   # MoviePy 1.x
            
            # Set timing
            try:
                text_clip = text_clip.with_start(start_time)  # MoviePy 2.x
            except AttributeError:
                text_clip = text_clip.set_start(start_time)   # MoviePy 1.x
            
            # Apply animation (simplified for compatibility)
            if animation == "fade_in":
                try:
                    text_clip = text_clip.fadein(0.5)
                except:
                    pass  # Skip animation if not supported
            elif animation == "fade_out":
                try:
                    text_clip = text_clip.fadeout(0.5)
                except:
                    pass
            elif animation == "fade_in_out":
                try:
                    text_clip = text_clip.fadein(0.5).fadeout(0.5)
                except:
                    pass
            
            self.text_clips.append(text_clip)
            self.logger.info(f"Added text overlay: {text[:30]}...")
            
        except Exception as e:
            self.logger.error(f"Error adding text overlay: {e}")
            raise
    
    def add_tts_narration(self, text: str, voice: str = None, start_time: float = 0):
        """Add text-to-speech narration"""
        if not self.tts_engine:
            self.logger.warning("No TTS engine configured")
            return
        
        try:
            # Generate audio
            temp_audio = tempfile.mktemp(suffix=".wav")
            self.temp_files.append(temp_audio)
            
            if voice:
                original_voice = self.tts_engine.config.get("voice")
                self.tts_engine.speech_config.speech_synthesis_voice_name = voice
            
            audio_path = self.tts_engine.text_to_audio(text, temp_audio)
            
            # Restore original voice
            if voice and 'original_voice' in locals():
                self.tts_engine.speech_config.speech_synthesis_voice_name = original_voice
            
            # Add to audio clips - compatible with both MoviePy versions
            try:
                audio_clip = AudioFileClip(audio_path).with_start(start_time)  # MoviePy 2.x
            except AttributeError:
                audio_clip = AudioFileClip(audio_path).set_start(start_time)   # MoviePy 1.x
                
            self.audio_clips.append(audio_clip)
            
            self.logger.info(f"Added TTS narration: {text[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Error adding TTS narration: {e}")
    
    def add_background_music(self, music_path: str, volume: float = None, loop: bool = True):
        """Add background music"""
        try:
            volume = volume or self.config.get("audio.background_volume", 0.3)
            
            music = AudioFileClip(music_path)
            
            # Apply volume
            try:
                music = music.with_volume_scaled(volume)  # MoviePy 2.x
            except AttributeError:
                music = music.volumex(volume)             # MoviePy 1.x
            
            # Loop music if needed
            if loop and len(self.video_clips) > 0:
                video_duration = max([clip.duration for clip in self.video_clips])
                if music.duration < video_duration:
                    loops_needed = int(video_duration / music.duration) + 1
                    music = concatenate_audioclips([music] * loops_needed)
                    try:
                        music = music.subclipped(0, video_duration)  # MoviePy 2.x
                    except AttributeError:
                        music = music.subclip(0, video_duration)     # MoviePy 1.x
            
            self.audio_clips.append(music)
            self.logger.info(f"Added background music: {music_path}")
            
        except Exception as e:
            self.logger.error(f"Error adding background music: {e}")
    
    def create_subtitle_clip(self, text: str, start_time: float, duration: float,
                           style: str = "default") -> TextClip:
        """Create a subtitle clip with specific styling"""
        
        styles = {
            "default": {
                "font_size": 36,
                "color": "white",
                "stroke_color": "black",
                "stroke_width": 2
            },
            "youtube": {
                "font_size": 42,
                "color": "white",
                "stroke_color": "black", 
                "stroke_width": 3
            },
            "tiktok": {
                "font_size": 48,
                "color": "white",
                "stroke_color": "black",
                "stroke_width": 3
            }
        }
        
        style_config = styles.get(style, styles["default"])
        
        try:
            # Create subtitle using safe method
            subtitle = self._create_text_clip_safe(text, style_config["font_size"], style_config["color"])
            
            # Set position, start time, and duration
            try:
                subtitle = (subtitle
                           .with_position(('center', 'bottom'))
                           .with_start(start_time)
                           .with_duration(duration))  # MoviePy 2.x
            except AttributeError:
                subtitle = (subtitle
                           .set_position(('center', 'bottom'))
                           .set_start(start_time)
                           .set_duration(duration))   # MoviePy 1.x
            
            return subtitle
            
        except Exception as e:
            self.logger.error(f"Error creating subtitle clip: {e}")
            raise
    
    def add_subtitles(self, subtitle_data: List[Dict], style: str = "default"):
        """Add subtitles from subtitle data"""
        for subtitle in subtitle_data:
            text = subtitle.get("text", "")
            start = subtitle.get("start", 0)
            duration = subtitle.get("duration", 2)
            
            subtitle_clip = self.create_subtitle_clip(text, start, duration, style)
            self.text_clips.append(subtitle_clip)
        
        self.logger.info(f"Added {len(subtitle_data)} subtitle clips")
    
    def render_video(self, output_path: str, quality: str = "medium") -> str:
        """Render the final video"""
        try:
            self.logger.info("Starting video render...")
            
            # Quality presets
            quality_settings = {
                "low": {"bitrate": "1000k", "audio_bitrate": "128k"},
                "medium": {"bitrate": "2500k", "audio_bitrate": "192k"},
                "high": {"bitrate": "5000k", "audio_bitrate": "320k"}
            }
            
            settings = quality_settings.get(quality, quality_settings["medium"])
            
            # Combine video clips
            if not self.video_clips:
                # Create default background if no video
                self.add_background_color(duration=60)
            
            final_video = CompositeVideoClip(self.video_clips + self.text_clips)
            
            # Combine audio clips
            if self.audio_clips:
                final_audio = CompositeAudioClip(self.audio_clips)
                try:
                    final_video = final_video.with_audio(final_audio)  # MoviePy 2.x
                except AttributeError:
                    final_video = final_video.set_audio(final_audio)   # MoviePy 1.x
            
            # Render with version-specific parameters
            if MOVIEPY_VERSION == 2:
                # MoviePy 2.x - simplified parameters
                final_video.write_videofile(
                    output_path,
                    fps=self.fps
                )
            else:
                # MoviePy 1.x - use original parameters
                temp_audiofile = tempfile.mktemp(suffix=".mp3")
                final_video.write_videofile(
                    output_path,
                    fps=self.fps,
                    bitrate=settings["bitrate"],
                    audio_bitrate=settings["audio_bitrate"],
                    temp_audiofile=temp_audiofile,
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
            
            # Cleanup
            try:
                final_video.close()
            except:
                pass
            self._cleanup_temp_files()
            
            self.logger.info(f"Video rendered successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error rendering video: {e}")
            self._cleanup_temp_files()
            raise
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"Could not remove temp file {temp_file}: {e}")
        self.temp_files.clear()
    
    def reset(self):
        """Reset the video creator for new video"""
        # Close all clips to free resources
        for clip in self.video_clips + self.audio_clips + self.text_clips:
            try:
                clip.close()
            except:
                pass
        
        self.video_clips.clear()
        self.audio_clips.clear()
        self.text_clips.clear()
        self._cleanup_temp_files()
        
        self.logger.info("Video creator reset")

class RedditVideoHandler(VideoCreator):
    """Specialized handler for Reddit-style videos"""
    
    def __init__(self, config: VideoCreatorConfig = None, tts_config: Dict = None):
        super().__init__(config, tts_config)
        self.comment_style = {
            "background_color": (30, 30, 30),
            "text_color": (255, 255, 255),
            "username_color": (255, 165, 0),
            "padding": 20,
            "corner_radius": 10
        }
    
    def create_comment_bubble(self, comment_data: Dict, position: Tuple[int, int],
                            start_time: float, duration: float) -> ImageClip:
        """Create a Reddit-style comment bubble"""
        
        username = comment_data.get("author", "Anonymous")
        text = comment_data.get("body", "")
        score = comment_data.get("score", 0)
        
        # Calculate bubble dimensions
        max_width = int(self.resolution[0] * 0.8)
        
        # Create bubble image
        bubble_img = self._create_bubble_image(username, text, score, max_width)
        
        # Convert to video clip - compatible with both MoviePy versions
        try:
            # MoviePy 2.x syntax
            bubble_clip = ImageClip(np.array(bubble_img)).with_duration(duration).with_start(start_time)
            bubble_clip = bubble_clip.with_position(position)
            
            # Add fade in/out animation
            try:
                bubble_clip = bubble_clip.fadein(0.3).fadeout(0.3)
            except:
                pass  # Skip animation if not supported
        except (AttributeError, TypeError):
            # MoviePy 1.x fallback
            bubble_clip = ImageClip(np.array(bubble_img)).set_duration(duration).set_start(start_time)
            bubble_clip = bubble_clip.set_position(position)
            
            # Add fade in/out animation
            try:
                bubble_clip = bubble_clip.fadein(0.3).fadeout(0.3)
            except:
                pass
        
        return bubble_clip
    
    def _create_bubble_image(self, username: str, text: str, score: int, max_width: int) -> Image.Image:
        """Create comment bubble image using PIL"""
        
        # Fonts (using default system fonts)
        try:
            username_font = ImageFont.truetype("arial.ttf", 16)
            text_font = ImageFont.truetype("arial.ttf", 14) 
            score_font = ImageFont.truetype("arial.ttf", 12)
        except:
            username_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            score_font = ImageFont.load_default()
        
        # Calculate text dimensions
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        
        # Wrap text
        wrapped_text = self._wrap_text(text, text_font, max_width - 40, draw)
        
        # Calculate bubble size
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=text_font)
        username_bbox = draw.textbbox((0, 0), f"u/{username}", font=username_font)
        score_bbox = draw.textbbox((0, 0), f"▲ {score}", font=score_font)
        
        bubble_width = max(text_bbox[2], username_bbox[2], score_bbox[2]) + 40
        bubble_height = text_bbox[3] + username_bbox[3] + score_bbox[3] + 60
        
        # Create bubble
        bubble = Image.new('RGBA', (bubble_width, bubble_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bubble)
        
        # Draw background
        draw.rounded_rectangle(
            [(0, 0), (bubble_width, bubble_height)],
            radius=self.comment_style["corner_radius"],
            fill=(*self.comment_style["background_color"], 230)
        )
        
        # Draw content
        y_pos = 15
        
        # Username
        draw.text((20, y_pos), f"u/{username}", 
                 fill=self.comment_style["username_color"], font=username_font)
        y_pos += username_bbox[3] + 10
        
        # Comment text
        draw.multiline_text((20, y_pos), wrapped_text,
                          fill=self.comment_style["text_color"], font=text_font)
        y_pos += text_bbox[3] + 10
        
        # Score
        draw.text((20, y_pos), f"▲ {score}",
                 fill=(255, 100, 100), font=score_font)
        
        return bubble
    
    def _wrap_text(self, text: str, font, max_width: int, draw) -> str:
        """Wrap text to fit within specified width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            
            if bbox[2] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, break it
                    lines.append(word[:20] + "...")
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def add_reddit_post(self, post_data: Dict, comments: List[Dict] = None):
        """Add Reddit post with optional comments"""
        
        title = post_data.get("title", "")
        author = post_data.get("author", "")
        score = post_data.get("score", 0)
        subreddit = post_data.get("subreddit", "")
        
        # Add title as main text overlay
        self.add_text_overlay(
            f"r/{subreddit}",
            start_time=0,
            duration=3,
            position=('center', 50),
            font_size=24,
            animation="fade_in"
        )
        
        self.add_text_overlay(
            title,
            start_time=0.5,
            duration=5,
            position='center',
            font_size=32,
            animation="fade_in_out"
        )
        
        # Add comments as bubbles
        if comments:
            comment_start_time = 3
            for i, comment in enumerate(comments[:5]):  # Limit to 5 comments
                position = (50, 200 + i * 150)
                duration = 4
                
                bubble_clip = self.create_comment_bubble(
                    comment, position, comment_start_time, duration
                )
                self.text_clips.append(bubble_clip)
                
                comment_start_time += 2  # Stagger comment appearances
        
        self.logger.info(f"Added Reddit post: {title[:30]}...")

# Example usage and testing
def main():
    """Example usage of the video creation system"""
    
    # Configuration
    config = VideoCreatorConfig()
    
    # TTS configuration (optional - requires Azure Cognitive Services)
    tts_config = {
        "provider": "azure",
        "speech_key": os.environ.get("AZURE_SPEECH_KEY"),  # Replace with actual key
        "speech_region": os.environ.get("AZURE_SPEECH_REGION"),   # Replace with actual region
        "voice": "en-US-AriaNeural"
    }
    
    # Create Reddit video handler
    reddit_video = RedditVideoHandler(config, tts_config)
    
    # Set platform
    reddit_video.set_platform("youtube_shorts")
    
    # Add background
    reddit_video.add_background_color((20, 20, 30), duration=20)
    
    # Sample Reddit post data
    post_data = {
        "title": "Scientists discover new quantum computing breakthrough",
        "author": "tech_researcher",
        "score": 1500,
        "subreddit": "technology"
    }
    
    # Sample comments
    comments = [
        {"author": "user1", "body": "This is incredible! The implications for cryptography are huge.", "score": 245},
        {"author": "user2", "body": "Finally! I've been waiting for this breakthrough for years.", "score": 156},
        {"author": "user3", "body": "Can someone explain this in simple terms?", "score": 89}
    ]
    
    # Create video
    reddit_video.add_reddit_post(post_data, comments)
    
    # Add narration (if TTS is configured)
    # reddit_video.add_tts_narration("Check out this amazing breakthrough in quantum computing!")
    
    # Render video
    output_path = "reddit_video_test.mp4"
    reddit_video.render_video(output_path, quality="medium")
    
    print(f"Video created: {output_path}")
    
    # Reset for next video
    reddit_video.reset()

if __name__ == "__main__":
    main()