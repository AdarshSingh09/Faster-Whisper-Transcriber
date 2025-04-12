from faster_whisper import WhisperModel
import os

def transcribe_audio_to_srt(audio_file_path, model_size="medium", device="cuda", compute_type="int8", beam_size=5, captions_dir="captions"):
    
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        
        segments, info = model.transcribe(audio_file_path, beam_size=beam_size)

        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

        if not os.path.exists(captions_dir):
            os.makedirs(captions_dir)

        audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        srt_file_path = os.path.join(captions_dir, f"{audio_file_name}.srt")

        with open(srt_file_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(segments, start=1):
                start_time = segment.start
                end_time = segment.end
                text = segment.text.strip() 

                # SRT format (HH:MM:SS,ms)
                def format_timestamp(seconds):
                    milliseconds = int(seconds * 1000)
                    hours, milliseconds = divmod(milliseconds, 3600000)
                    minutes, milliseconds = divmod(milliseconds, 60000)
                    seconds, milliseconds = divmod(milliseconds, 1000)
                    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

                srt_file.write(f"{i}\n")
                srt_file.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                srt_file.write(f"{text}\n\n")

        print(f"SRT file created at: {srt_file_path}")
        return srt_file_path

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

if __name__ == "__main__":
    audio_file = r"D:\2025\project-stl\src"
    srt_file = transcribe_audio_to_srt(audio_file)

    if srt_file:
        print(f"Transcription and SRT creation successful!  SRT file saved to {srt_file}")
    else:
        print("Transcription or SRT creation failed.")
