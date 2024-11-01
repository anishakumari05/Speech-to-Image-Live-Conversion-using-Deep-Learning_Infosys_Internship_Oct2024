# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1V8H6jMTNqhyUM7weYACq3gpAC6tjW3BX
"""

import whisper
import sounddevice as sd
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import streamlit as st

# Load the Whisper model for audio transcription
model = whisper.load_model("base")

def record_audio(duration=5, samplerate=16000):
    """
    Record audio for a specified duration and sample rate,
    and return the flattened audio data.
    """
    st.write("Recording...")
    # Record audio for the given duration
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished.")
    return audio.flatten()

def transcribe_audio(audio):
    """
    Transcribe the given audio using Whisper.
    """
    # Pad or trim the audio to fit Whisper's requirements
    audio = whisper.pad_or_trim(audio)
    # Convert audio to log-mel spectrogram and move to model's device
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Set decoding options
    options = whisper.DecodingOptions(language="en")
    # Decode the spectrogram to get the transcription
    result = model.decode(mel, options)
    return result.text

def generate_image_from_text(prompt, guidance_scale=8.0, steps=50):
    """
    Generate an image based on the given text prompt using Stable Diffusion.
    """
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
    pipe.scheduler.num_inference_steps = steps

    # Generate image using the provided prompt
    with torch.autocast("cuda"):
        generated_image = pipe(prompt, guidance_scale=guidance_scale).images[0]
    return generated_image

# Streamlit UI elements
st.title("Speech-to-Image Conversion")
st.write("Record audio, transcribe it to text, and generate an image based on the transcription.")

# Set recording duration (in seconds)
duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

if st.button("Start Recording"):
    # Step 1: Record and transcribe audio
    audio = record_audio(duration)
    transcription = transcribe_audio(audio)
    st.write("Transcription:", transcription)

    # Step 2: Generate an image based on the transcription
    st.write("Generating image based on transcription...")
    generated_image = generate_image_from_text(transcription)

    # Display the generated image
    st.image(generated_image, caption="Generated Image", use_column_width=True)