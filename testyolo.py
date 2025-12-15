import torch
import librosa
from moviepy import VideoFileClip
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30

# =============================
# 1. Tách audio từ video
# =============================
def extract_audio_from_video(video_file: str, audio_file: str) -> str:
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file, fps=SAMPLE_RATE)
    video.close()
    return audio_file

# =============================
# 2. Cắt audio thành các đoạn 30s
# =============================
def split_audio(audio, sr, chunk_seconds=30):
    chunk_size = chunk_seconds * sr
    return [
        audio[i:i + chunk_size]
        for i in range(0, len(audio), chunk_size)
    ]

# =============================
# 3. Transcribe toàn bộ audio
# =============================
def transcribe_audio_full(audio_file: str, model_name="openai/whisper-small") -> str:
    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.eval()

    forced_ids = processor.get_decoder_prompt_ids(
        language="vi",
        task="transcribe"
    )

    chunks = split_audio(audio, sr, CHUNK_SECONDS)
    results = []

    for idx, chunk in enumerate(chunks):
        inputs = processor(
            chunk,
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE
        )

        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"],
                forced_decoder_ids=forced_ids
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        print(f"✔ Chunk {idx+1}/{len(chunks)} done")
        results.append(text.strip())

    return " ".join(results)

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    video_file = "baigiang.mp4"
    audio_file = "extracted_audio.wav"

    extract_audio_from_video(video_file, audio_file)

    transcription = transcribe_audio_full(
        audio_file,
        model_name="openai/whisper-small"
    )

    print("\n===== FULL TRANSCRIPTION =====")
    print(transcription)
