import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

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
    
    def text_to_audio_with_timing(self, text: str, output_path: str = None) -> tuple:
        """Convert text to audio file and return path, duration, and word timings"""
        if not output_path:
            output_path = tempfile.mktemp(suffix=".wav")
        
        if self.provider == "azure" and hasattr(self, 'speech_config'):
            try:
                # Configure for word boundary events
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=self.speech_config,
                    audio_config=speechsdk.audio.AudioOutputConfig(filename=output_path)
                )
                
                # Track word timings
                word_timings = []
                
                def word_boundary_cb(evt):
                    try:
                        # Handle different data types from Azure SDK
                        if hasattr(evt.audio_offset, 'total_seconds'):
                            # It's a timedelta object
                            start_time = float(evt.audio_offset.total_seconds())
                        else:
                            # It's already a number (in 100-nanosecond units)
                            start_time = float(evt.audio_offset) / 10000000.0
                        
                        if hasattr(evt, 'duration'):
                            if hasattr(evt.duration, 'total_seconds'):
                                # It's a timedelta object
                                duration = float(evt.duration.total_seconds())
                            else:
                                # It's already a number (in 100-nanosecond units)
                                duration = float(evt.duration) / 10000000.0
                        else:
                            duration = 0.3  # Default duration
                        
                        word_timings.append({
                            'word': str(evt.text),
                            'start_time': start_time,
                            'duration': duration
                        })
                    except Exception as e:
                        # Skip problematic word boundaries rather than logging spam
                        pass
                
                # Connect the callback
                synthesizer.synthesis_word_boundary.connect(word_boundary_cb)
                
                result = synthesizer.speak_text_async(text).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    # Get actual duration from the audio file
                    duration = self._get_audio_duration(output_path)
                    
                    # If we didn't get word timings, estimate them
                    if not word_timings:
                        word_timings = self._estimate_word_timings(text, duration)
                    
                    return output_path, duration, word_timings
                else:
                    raise Exception(f"TTS failed: {result.reason}")
            except Exception as e:
                logging.warning(f"Azure TTS with timing failed: {e}, falling back to estimated timing")
                duration = len(text) * 0.08  # Estimate: 0.08 seconds per character
                word_timings = self._estimate_word_timings(text, duration)
                return self._create_silent_audio(duration, output_path), duration, word_timings
        else:
            # Fallback: create silent audio with estimated timing
            duration = len(text) * 0.08  # Estimate: 0.08 seconds per character
            word_timings = self._estimate_word_timings(text, duration)
            return self._create_silent_audio(duration, output_path), duration, word_timings
    
    def _estimate_word_timings(self, text: str, total_duration: float) -> List[Dict]:
        """Estimate word timings when real timing isn't available"""
        words = text.split()
        if not words:
            return []
        
        # Estimate based on word length and average speech rate
        word_timings = []
        current_time = 0.0
        
        # Calculate average time per character
        total_chars = sum(len(word) for word in words)
        time_per_char = total_duration / max(total_chars, 1)
        
        for word in words:
            word_duration = len(word) * time_per_char + 0.1  # Add small pause
            word_timings.append({
                'word': word,
                'start_time': current_time,
                'duration': word_duration
            })
            current_time += word_duration
        
        return word_timings
    
    def text_to_audio(self, text: str, output_path: str = None) -> tuple:
        """Convert text to audio file and return path and duration (backward compatibility)"""
        audio_path, duration, _ = self.text_to_audio_with_timing(text, output_path)
        return audio_path, duration
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file"""
        try:
            from moviepy import AudioFileClip
            audio = AudioFileClip(audio_path)
            duration = float(audio.duration)  # Ensure it's a float
            audio.close()
            return duration
        except Exception as e:
            self.logger.warning(f"Could not get audio duration: {e}")
            # Fallback: estimate based on file size
            try:
                import os
                file_size = os.path.getsize(audio_path)
                # Rough estimate: WAV files at 44.1kHz, 16-bit, stereo ~ 176KB per second
                estimated_duration = file_size / (44100 * 2 * 2)
                return max(1.0, float(estimated_duration))  # Minimum 1 second, ensure float
            except:
                return 3.0  # Default fallback
    
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
                # Fallback - try with minimal parameters and positional arguments
                if MOVIEPY_VERSION == 2:
                    text_clip = TextClip(text)
                    if hasattr(text_clip, 'with_font_size'):
                        text_clip = text_clip.with_font_size(font_size)
                else:
                    text_clip = TextClip(text, fontsize=font_size)
                return text_clip
            except Exception as e2:
                self.logger.error(f"Failed to create text clip with fallback: {e2}")
                # Last resort - create with just text
                try:
                    if MOVIEPY_VERSION == 2:
                        return TextClip(text)
                    else:
                        return TextClip(text)
                except:
                    raise Exception("Could not create TextClip with any method")
    
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
    
    def add_tts_narration_with_subtitles(self, text: str, voice: str = None, start_time: float = 0) -> tuple:
        """Add text-to-speech narration with synchronized subtitles and return duration and word timings"""
        if not self.tts_engine:
            self.logger.warning("No TTS engine configured")
            return 0, []
        
        try:
            # Generate audio with word timings
            temp_audio = tempfile.mktemp(suffix=".wav")
            self.temp_files.append(temp_audio)
            
            if voice:
                original_voice = self.tts_engine.config.get("voice")
                self.tts_engine.speech_config.speech_synthesis_voice_name = voice
            
            audio_path, duration, word_timings = self.tts_engine.text_to_audio_with_timing(text, temp_audio)
            
            # Restore original voice
            if voice and 'original_voice' in locals():
                self.tts_engine.speech_config.speech_synthesis_voice_name = original_voice
            
            # Add to audio clips
            try:
                audio_clip = AudioFileClip(audio_path).with_start(start_time)  # MoviePy 2.x
            except AttributeError:
                audio_clip = AudioFileClip(audio_path).set_start(start_time)   # MoviePy 1.x
                
            self.audio_clips.append(audio_clip)
            
            # Create simplified subtitles (no complex word highlighting for now)
            self._create_simple_subtitles(text, duration, start_time)
            
            self.logger.info(f"Added TTS with subtitles: {text[:50]}... (duration: {duration:.1f}s)")
            return duration, word_timings
            
        except Exception as e:
            self.logger.error(f"Error adding TTS narration with subtitles: {e}")
            return 0, []
    
    def _create_simple_subtitles(self, text: str, duration: float, start_time: float):
        """Create simple subtitles without complex highlighting"""
        try:
            # Split text into manageable chunks (about 40 characters per line)
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > 40 and current_chunk:  # +1 for space
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Create subtitle clips for each chunk
            chunk_duration = duration / len(chunks) if chunks else duration
            
            for i, chunk_text in enumerate(chunks):
                chunk_start = start_time + (i * chunk_duration)
                
                # Create subtitle clip
                subtitle_clip = self._create_text_clip_safe(chunk_text, 44, 'white')
                
                # Position at bottom of screen
                subtitle_position = ('center', self.resolution[1] - 150)
                
                # Use MoviePy 2.x compatible methods
                try:
                    subtitle_clip = (subtitle_clip
                                   .with_position(subtitle_position)
                                   .with_start(chunk_start)
                                   .with_duration(chunk_duration))
                    
                    # Add subtle fade for smooth transitions
                    try:
                        subtitle_clip = subtitle_clip.fadein(0.2).fadeout(0.2)
                    except:
                        pass  # Skip fade if not supported
                    
                except AttributeError:
                    # Fallback for MoviePy 1.x
                    subtitle_clip = (subtitle_clip
                                   .set_position(subtitle_position)
                                   .set_start(chunk_start)
                                   .set_duration(chunk_duration))
                    
                    # Add subtle fade for smooth transitions
                    try:
                        subtitle_clip = subtitle_clip.fadein(0.2).fadeout(0.2)
                    except:
                        pass
                
                self.text_clips.append(subtitle_clip)
                
        except Exception as e:
            self.logger.warning(f"Failed to create subtitles: {e}")
            # Fallback: single subtitle for entire text
            try:
                simple_text = text[:60] + "..." if len(text) > 60 else text
                simple_subtitle = self._create_text_clip_safe(simple_text, 40, 'white')
                
                # Use MoviePy 2.x compatible methods with fallback
                try:
                    simple_subtitle = (simple_subtitle
                                     .with_position(('center', self.resolution[1] - 150))
                                     .with_start(start_time)
                                     .with_duration(duration))
                except AttributeError:
                    # Fallback for MoviePy 1.x
                    simple_subtitle = (simple_subtitle
                                     .set_position(('center', self.resolution[1] - 150))
                                     .set_start(start_time)
                                     .set_duration(duration))
                
                self.text_clips.append(simple_subtitle)
            except Exception as e2:
                self.logger.error(f"Failed to create fallback subtitle: {e2}")
                pass
    

    
    def _create_word_highlights(self, word_chunk: List[Dict], chunk_text: str, chunk_start: float, start_offset: float):
        """Create individual word highlight effects"""
        for i, word_info in enumerate(word_chunk):
            word = word_info['word']
            word_start = start_offset + word_info['start_time']
            word_duration = word_info['duration']
            
            # Create highlighted version of the word
            # We'll create a yellow highlight that appears over the word
            try:
                highlight_clip = self._create_text_clip_safe(word, 42, 'yellow')
                
                # Position it to align with the word in the subtitle
                # This is a simplified approach - in a real implementation, you'd calculate exact word positions
                word_offset = i * 100  # Approximate horizontal offset per word
                highlight_position = ('center', self.resolution[1] - 150)  # Bottom area
                
                highlight_clip = (highlight_clip
                                .with_position(highlight_position)
                                .with_start(word_start)
                                .with_duration(word_duration))
                
                # Add fade in/out for smooth highlighting
                try:
                    highlight_clip = highlight_clip.fadein(0.1).fadeout(0.1)
                except:
                    pass
                
                self.text_clips.append(highlight_clip)
                
            except Exception as e:
                self.logger.warning(f"Failed to create word highlight for '{word}': {e}")
                continue
    
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
    
    def add_reddit_post_with_subtitles(self, post_data: Dict, comments: List[Dict] = None, use_tts_for_comments: bool = True):
        """Add Reddit post with subtitle-based comments (no visual bubbles)"""
        
        title = post_data.get("title", "")
        subreddit = post_data.get("subreddit", "")
        
        # Add title as main text overlay (brief display)
        self.add_text_overlay(
            f"r/{subreddit}",
            start_time=0,
            duration=2,
            position=('center', 100),
            font_size=28,
            animation="fade_in"
        )
        
        self.add_text_overlay(
            title,
            start_time=1,
            duration=4,
            position='center',
            font_size=36,
            animation="fade_in_out"
        )
        
        # Process comments with TTS and subtitles only
        current_time = 6.0  # Start after title display
        
        if comments and use_tts_for_comments and self.tts_engine:
            for i, comment in enumerate(comments[:3]):  # Limit to 3 comments for better timing
                comment_text = comment.get("body", "")
                comment_author = comment.get("author", "Anonymous")
                
                # Skip very long comments to keep video manageable
                if len(comment_text) > 250:
                    comment_text = comment_text[:247] + "..."
                
                # Clean up comment text
                comment_text = comment_text.replace('\n', ' ').strip()
                
                # Create TTS narration with subtitles for the comment
                narration_text = f"Comment by {comment_author}: {comment_text}"
                
                try:
                    tts_duration, word_timings = self.add_tts_narration_with_subtitles(
                        narration_text, 
                        start_time=current_time
                    )
                    
                    # Use actual TTS duration or fallback to estimated duration
                    if tts_duration > 0:
                        comment_duration = tts_duration
                    else:
                        # Fallback: estimate duration based on text length
                        comment_duration = len(narration_text) * 0.08  # ~0.08 seconds per character
                        
                        # Create simple subtitle as fallback
                        subtitle_clip = self._create_text_clip_safe(narration_text, 42, 'white')
                        try:
                            subtitle_clip = (subtitle_clip
                                           .with_position(('center', self.resolution[1] - 200))
                                           .with_start(current_time)
                                           .with_duration(comment_duration))
                        except AttributeError:
                            subtitle_clip = (subtitle_clip
                                           .set_position(('center', self.resolution[1] - 200))
                                           .set_start(current_time)
                                           .set_duration(comment_duration))
                        
                        self.text_clips.append(subtitle_clip)
                    
                    # Move to next comment after this one finishes
                    current_time += comment_duration + 1.5  # 1.5 second gap between comments
                    
                    self.logger.info(f"Added comment {i+1} with subtitles: {comment_text[:30]}... (duration: {comment_duration:.1f}s)")
                    
                except Exception as e:
                    self.logger.error(f"Error processing comment {i+1}: {e}")
                    # Skip this comment and continue
                    current_time += 3.0  # Add minimal time for failed comment
                    continue
        
        self.logger.info(f"Added Reddit post with subtitles: {title[:30]}...")
        return current_time  # Return total duration for video length calculation

# Example usage and testing
def main():
    """Example usage of the video creation system"""
    
    # Configuration
    config = VideoCreatorConfig()
    
    # TTS configuration (optional - requires Azure Cognitive Services)
    tts_config = {
        "provider": "azure",
        "speech_key": "your_speech_key",  # Replace with actual key
        "speech_region": "your_region",   # Replace with actual region
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