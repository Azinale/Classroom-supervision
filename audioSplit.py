from moviepy import VideoFileClip

# Load video file and extract audio
video = VideoFileClip("Download.mp4")
audio = video.audio

# Save audio to a file
audio.write_audiofile("extracted_audio.wav")
