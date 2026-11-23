import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from pathlib import Path
from typing import Tuple, Optional, Union

def setup_logging(log_file: str = "audio_trimmer.log") -> logging.Logger:
    """Configure console and file logging for the application."""
    logger = logging.getLogger("audio_trimmer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


def read_audio_file(file_path: str) -> Tuple[int, np.ndarray]:
    """Read an audio file using soundfile and return sample rate and audio data."""
    import soundfile as sf

    logger.debug("Attempting to read audio file from path: %s", file_path)
    try:
        data, samplerate = sf.read(file_path)
        logger.debug(
            "Successfully read audio file '%s' with shape %s and sample rate %d Hz",
            file_path,
            data.shape,
            samplerate,
        )
        return samplerate, data
    except Exception:
        logger.exception("Failed to read audio file: %s", file_path)
        sys.exit(1)


def print_audio_info(sample_rate: int, audio_data: np.ndarray) -> float:
    """Log and return summary information about the audio signal."""
    duration = len(audio_data) / sample_rate
    logger.info("Sample rate: %d Hz", sample_rate)
    logger.info("Audio length: %d samples", len(audio_data))
    logger.info("Duration: %.2f seconds", duration)

    if audio_data.ndim == 1:
        logger.info("Channel mode: Mono")
        logger.info(
            "Amplitude range: [%.3f, %.3f]",
            float(audio_data.min()),
            float(audio_data.max()),
        )
    else:
        logger.info("Channel mode: Stereo (%d channels)", audio_data.shape[1])
        logger.info(
            "Left channel amplitude range: [%.3f, %.3f]",
            float(audio_data[:, 0].min()),
            float(audio_data[:, 0].max()),
        )
        logger.info(
            "Right channel amplitude range: [%.3f, %.3f]",
            float(audio_data[:, 1].min()),
            float(audio_data[:, 1].max()),
        )

    return duration


def trim_audio(audio_data: np.ndarray, sample_rate: int, start_time: float, end_time: float)-> np.ndarray:
    """Trim audio data to the specified [start_time, end_time] interval in seconds."""
    start_sample = max(0, int(start_time * sample_rate))
    end_sample = min(len(audio_data), int(end_time * sample_rate))
    logger.debug(
        "Calculated sample range: [%d, %d] for interval [%.2fs, %.2fs]",
        start_sample,
        end_sample,
        start_time,
        end_time,
    )

    if start_sample >= end_sample:
        logger.error(
            "Invalid trim range: start_sample (%d) >= end_sample (%d)",
            start_sample,
            end_sample,
        )
        sys.exit(1)

    if audio_data.ndim == 1:
        return audio_data[start_sample:end_sample]
    return audio_data[start_sample:end_sample, :]


def plot_audio_comparison(original_audio: np.ndarray, trimmed_audio: np.ndarray, original_duration: float, start_time: float, end_time: float) -> None:
    """Plot waveform comparison between original and trimmed audio."""
    logger.debug("Generating waveform comparison plot")
    time_original = np.linspace(0, original_duration, len(original_audio))
    time_trimmed = np.linspace(start_time, end_time, len(trimmed_audio))

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    if original_audio.ndim == 1:
        ax1.plot(time_original, original_audio, "b-", alpha=0.7, linewidth=0.5)
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Original Audio (Mono)")
    else:
        ax1.plot(
            time_original,
            original_audio[:, 0],
            "b-",
            alpha=0.7,
            linewidth=0.5,
            label="Left channel",
        )
        ax1.plot(
            time_original,
            original_audio[:, 1],
            "r-",
            alpha=0.7,
            linewidth=0.5,
            label="Right channel",
        )
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Original Audio (Stereo)")

    ax1.axvspan(start_time, end_time, color="red", alpha=0.3, label="Trim region")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if trimmed_audio.ndim == 1:
        ax2.plot(time_trimmed, trimmed_audio, "g-", alpha=0.7, linewidth=0.5)
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Trimmed Audio (Mono)")
    else:
        ax2.plot(
            time_trimmed,
            trimmed_audio[:, 0],
            "b-",
            alpha=0.7,
            linewidth=0.5,
            label="Left channel",
        )
        ax2.plot(
            time_trimmed,
            trimmed_audio[:, 1],
            "r-",
            alpha=0.7,
            linewidth=0.5,
            label="Right channel",
        )
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Trimmed Audio (Stereo)")

    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_audio_file(file_path: str, sample_rate: int, audio_data: np.ndarray)-> None:
    """Save trimmed audio data to disk using soundfile."""
    import soundfile as sf

    logger.debug("Writing trimmed audio to '%s' at %d Hz", file_path, sample_rate)
    try:
        sf.write(file_path, audio_data, sample_rate)
        logger.info("Trimmed audio saved successfully to: %s", file_path)
    except Exception:
        logger.exception("Failed to save trimmed audio to: %s", file_path)
        sys.exit(1)


def main() -> None:
    """Parse CLI arguments, trim audio, display comparison, and save output."""
    parser = argparse.ArgumentParser(
        description="Trim audio within a specified time interval"
    )
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("output_file", help="Path to save trimmed audio file")
    parser.add_argument("start_time", type=float, help="Start time in seconds")
    parser.add_argument("end_time", type=float, help="End time in seconds")

    args = parser.parse_args()
    logger.info(
        "Starting audio trimmer | input='%s' | output='%s' | start=%.2fs | end=%.2fs",
        args.input_file,
        args.output_file,
        args.start_time,
        args.end_time,
    )

    if not os.path.exists(args.input_file):
        logger.error("Input file does not exist: %s", args.input_file)
        sys.exit(1)

    if args.start_time < 0 or args.end_time < 0:
        logger.error(
            "Time values cannot be negative: start=%.2f, end=%.2f",
            args.start_time,
            args.end_time,
        )
        sys.exit(1)

    if args.start_time >= args.end_time:
        logger.error(
            "Start time (%.2fs) must be less than end time (%.2fs)",
            args.start_time,
            args.end_time,
        )
        sys.exit(1)

    sample_rate, original_audio = read_audio_file(args.input_file)

    logger.info("=== Original Audio Info ===")
    original_duration = print_audio_info(sample_rate, original_audio)

    end_time = args.end_time
    if end_time > original_duration:
        logger.warning(
            "Requested end time (%.2fs) exceeds audio duration (%.2fs); clamping to %.2fs",
            end_time,
            original_duration,
            original_duration,
        )
        end_time = original_duration

    logger.info("Trimming audio from %.2fs to %.2fs...", args.start_time, end_time)
    trimmed_audio = trim_audio(original_audio, sample_rate, args.start_time, end_time)

    logger.info("=== Trimmed Audio Info ===")
    trimmed_duration = len(trimmed_audio) / sample_rate
    print_audio_info(sample_rate, trimmed_audio)

    plot_audio_comparison(
        original_audio,
        trimmed_audio,
        original_duration,
        args.start_time,
        end_time,
    )

    save_audio_file(args.output_file, sample_rate, trimmed_audio)
    logger.info(
        "Finished! New duration: %.2f seconds | Log saved to audio_trimmer.log",
        trimmed_duration,
    )


if __name__ == "__main__":
    main()