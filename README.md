# Faster Whisper Audio Transcriber

This Python script uses the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library to transcribe audio files and generate SRT (SubRip Subtitle) files.

This script is based on the faster-whisper library, which is a re-implementation of OpenAI's Whisper model with the following optimizations:

* Up to 4x faster than the original Whisper implementation.
* Lower memory usage.

##  Model Selection and Performance

This script is configured to use the `medium` Whisper model with `int8` quantization. This provides a good balance of speed and accuracy, and it worked well in my testing.

* **`medium` model:** Offers a good compromise between speed and accuracy.
* **`int8` quantization:** Improves the speed of transcription, especially on CPUs and some GPUs.

##   Other Model Options

The faster-whisper library supports various Whisper models. You can modify the `model_size` and `compute_type` parameters in the `transcribe_audio_to_srt` function to use different models:

* **Larger models (e.g., `large-v3`):** Provide higher accuracy but are slower and require more memory. To use the large model, change `model_size="large-v3"`
* **`float16` compute type:** For maximum accuracy on GPUs, you can use `compute_type="float16"`. This is the default.

    Example of using the large model and float16:

    ```python
    srt_file = transcribe_audio_to_srt(audio_file, model_size="large-v3", compute_type="float16")
    ```

    Note that this will be significantly slower and require a GPU with sufficient memory.

##  What This Code Does

The `main.py` script performs the following actions:

1.  Takes an audio file as input.
2.  Uses the faster-whisper library to transcribe the audio into text.
3.  Generates an SRT file containing the transcribed text with corresponding timestamps.
4.  Saves the SRT file to the `captions` directory.
5. **CPU and GPU Support**: The script can run on both CPUs and GPUs.

##  Usage

1.  **Prerequisites:**

    * Python 3.7 or higher
    * faster-whisper library (install with `pip install faster-whisper`)
    * (Optional) CUDA 12.6 toolkit and cuDNN 9 for GPU acceleration
    * soundfile (install with `pip install soundfile`)

2.  **Installation Details:**
    * For the latest installation instructions, including details on CUDA and cuDNN, please refer to the official faster-whisper GitHub repository: [faster-whisper](https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#requirements)
    * A `requirements.txt` file is included, which lists the Python packages needed to run this script.  You can install them using pip:

        ```bash
        pip install -r requirements.txt
        ```
3.  **Run the script:**

    ```bash
    python main.py
    ```

    * Ensure that the `audio_file` variable in `main.py` is set to the correct path of your audio file.
    * The script automatically uses the GPU if available.  You can specify the device in the code if needed.
