#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install git+https://github.com/openai/whisper.git --use-deprecated=legacy-resolver')


# In[5]:


get_ipython().system('pip install -U openai-whisper')


# In[1]:


import whisper
model = whisper.load_model("base")
result = model.transcribe(r"C:\Users\sarve\Downloads\Recording.mp3")
print(result["text"])


# In[2]:


import sys
print(sys.executable)
get_ipython().system('{sys.executable} -m pip install sounddevice')
get_ipython().system('pip show sounddevice')


# In[4]:


import sounddevice as sd
import numpy as np
import queue
import threading
import torch
import whisper
import time

model = whisper.load_model("base")
audio_queue = queue.Queue()

def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    audio_queue.put(audio.flatten())  
    print("Recording finished.")

def transcribe_audio():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            result = model.transcribe(audio_tensor.numpy(), language='en')
            print("Transcription:")
            print(result['text'])

transcription_thread = threading.Thread(target=transcribe_audio)
transcription_thread.daemon = True
transcription_thread.start()

def start_recording(total_duration=30, interval=5):
    start_time = time.time()
    try:
        while (time.time() - start_time) < total_duration:
            record_audio(duration=interval)
    except KeyboardInterrupt:
        print("Recording interrupted. Exiting...")

start_recording(total_duration=30, interval=5)



# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install lyricsgenius')
import whisper
import lyricsgenius
model = whisper.load_model("base")
result = model.transcribe(r"C:\Users\sarve\Downloads\Shape of You.mp3")
lyrics = result["text"]
print("Transcribed Lyrics:", lyrics)
genius = lyricsgenius.Genius("xLe0xGDFWC73A2zvQlLbbeeaASGNYTD8aU2uilUuxYg9q_dxUttzuEEAatQlw7eZ")
first_line = lyrics.split("\n")[0]
song = genius.search_song(first_line)
if song:
    print(f"Song Name: {song.title}")
    print(f"Artist: {song.artist}")
else:
    print("Song could not be identified.")


# In[ ]:


#Side project
import sys
get_ipython().system('{sys.executable} -m pip install lyricsgenius')
get_ipython().system('{sys.executable} -m pip install lyricsgenius requests')
import whisper
import lyricsgenius
model = whisper.load_model("base")
result = model.transcribe(r"C:\Users\sarve\Downloads\Shape of You.mp3")
lyrics = result["text"]
print("Transcribed Lyrics:", lyrics)
genius = lyricsgenius.Genius("xLe0xGDFWC73A2zvQlLbbeeaASGNYTD8aU2uilUuxYg9q_dxUttzuEEAatQlw7eZ")
first_line = lyrics.split("\n")[0]
song = genius.search_song(first_line)
if song:
    print(f"Song Name: {song.title}")
    print(f"Artist: {song.artist}")
else:
    print("Song could not be identified.")


# In[2]:


#Side project
import sys
import requests
from bs4 import BeautifulSoup
import whisper
get_ipython().system('{sys.executable} -m pip install beautifulsoup4 requests')
model = whisper.load_model("base")
result = model.transcribe(r"C:\Users\sarve\Downloads\Shape of You.mp3")
lyrics = result["text"]
print("Transcribed Lyrics:", lyrics)
def search_google_for_song(lyrics):
    query = f"{lyrics.strip()} song"
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error during requests to Google: {e}")
        return ""
html = search_google_for_song(lyrics)
soup = BeautifulSoup(html, "html.parser")
result_block = soup.find_all('div', class_='BNeawe iBp4i AP7Wnd')

if result_block:
    for result in result_block:
        print(result.get_text())
else:
    print("Song not found.")


# In[5]:


import sys
get_ipython().system('{sys.executable} -m pip install jiwer')
import whisper
from difflib import SequenceMatcher
from jiwer import wer
model = whisper.load_model("medium")
transcribed_text = model.transcribe(r"C:\Users\sarve\Downloads\Combined.mp3")["text"]
print("Transcribed Text:", transcribed_text)
reference_texts = [
    "Bonjour et bienvenue parmi nous. Hello everyone, welcome to the",
    "Bonjour et bienvenue parmi nous. Hello everyone, welcome to the project.",
]
total_wer = 0
total_count = len(reference_texts)
for ref_text in reference_texts:
    similarity = SequenceMatcher(None, ref_text, transcribed_text).ratio()
    error_rate = wer(ref_text, transcribed_text)
    total_wer += error_rate   
    print(f"Reference Text: {ref_text}")
    print(f"Similarity Ratio: {similarity:.2f}, WER: {error_rate:.2f}")
    print("Match!" if similarity > 0.8 else "No Match!")
    print()
average_wer = total_wer / total_count
accuracy = 1 - average_wer
print(f"Average WER: {average_wer:.2f}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# In[4]:


#extra work
import sounddevice as sd
import numpy as np
import queue
import threading
import torch
import whisper
from jiwer import wer 
model = whisper.load_model("base")
audio_queue = queue.Queue()
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    audio_queue.put(audio.flatten())  
    print("Recording finished.")
def transcribe_audio():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            result = model.transcribe(audio_tensor.numpy(), language='en')
            transcription_text = result['text']
            print("Transcription:")
            print(transcription_text)
            reference_text = "Hello everyone, welcome to the project."
            error_rate = wer(reference_text, transcription_text)
            print(f"Word Error Rate: {error_rate:.2f}")
transcription_thread = threading.Thread(target=transcribe_audio)
transcription_thread.daemon = True
transcription_thread.start()
def start_recording():
    try:
        while True:
            record_audio(duration=5)
    except KeyboardInterrupt:
        print("Recording interrupted. Exiting...")


# In[ ]:


start_recording()


# In[ ]:


#model-fine tuning
import os
import pandas as pd
csv_path = r'C:\Users\sarve\Downloads\sample_dataset\filtered_csv_file.csv'
audio_folder_path = r'C:\Users\sarve\Downloads\sample_dataset\train'
df = pd.read_csv(csv_path)
df['file_name'] = df['file_name'].apply(lambda x: os.path.abspath(os.path.join(audio_folder_path, os.path.basename(x))))
df.to_csv(csv_path, index=False)


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install datasets')
get_ipython().system('{sys.executable} -m pip install transformers')
get_ipython().system('{sys.executable} -m pip install librosa')
get_ipython().system('{sys.executable} -m pip install soundfile')
get_ipython().system('{sys.executable} -m pip install transformers[torch]')
from datasets import load_dataset, Audio
dataset = load_dataset('csv', data_files=r'C:\Users\sarve\Downloads\sample_dataset\filtered_csv_file.csv')
dataset = dataset.cast_column('file_name', Audio(sampling_rate=16000))
dataset = dataset.rename_column('file_name', 'audio')
dataset = dataset.rename_column('phrase', 'sentence')
print(dataset)
from transformers import WhisperProcessor
from datasets import load_dataset
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
def prepare_dataset(batch):
    batch["input_features"] = processor(batch["audio"]["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch
columns_to_remove = [
    'audio_clipping', 
    'audio_clipping:confidence', 
    'background_noise_audible',
    'background_noise_audible:confidence', 
    'overall_quality_of_the_audio',
    'quiet_speaker', 
    'quiet_speaker:confidence', 
    'speaker_id',
    'file_download', 
    'prompt', 
    'writer_id'
]
dataset = dataset.map(prepare_dataset, remove_columns=columns_to_remove)
print(dataset)


# In[3]:


import sys
get_ipython().system('{sys.executable} -m pip install "accelerate>=0.26.0"')
get_ipython().system('{sys.executable} -m pip install "transformers[torch]"')
import accelerate
import transformers
import torch
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.nn.utils.rnn import pad_sequence
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    fp16=False,
    save_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
)

class DataCollatorForWhisper:
    def __call__(self, features):
        
        input_features = [torch.tensor(feature["input_features"]) for feature in features]
        labels = [torch.tensor(feature["labels"]) for feature in features]

        
        input_features_padded = pad_sequence(input_features, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_features": input_features_padded,
            "labels": labels_padded
        }
data_collator = DataCollatorForWhisper()

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    tokenizer=processor.tokenizer,  
    data_collator=data_collator,
)
trainer.train()
model.save_pretrained("whisper-finetuned")
processor.save_pretrained("whisper-finetuned")


# In[1]:


#VQ model
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        return x

# Define Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.deconv1(x))
        x = nn.ReLU()(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

# Vector Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, z):
        z_flattened = z.view(-1, z.size(1))
        distances = torch.cdist(z_flattened.unsqueeze(0), self.embedding.weight.unsqueeze(0))
        encoding_indices = torch.argmin(distances, dim=2).view(z.size(0), -1)
        quantized = self.embedding(encoding_indices).view(z.size())
        return quantized, encoding_indices

# Define VQ-GAN
class VQGAN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VQGAN, self).__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        quantized, _ = self.quantizer(z)
        reconstructed = self.decoder(quantized)
        return reconstructed

# Instantiate models and define loss and optimizer
num_embeddings = 512
embedding_dim = 256
vqgan = VQGAN(num_embeddings, embedding_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(vqgan.parameters()), lr=0.0002)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_epochs = 5

for epoch in range(num_epochs):
    for images, _ in dataloader:
        optimizer.zero_grad()
        
        reconstructed_images = vqgan(images)
        loss = criterion(reconstructed_images, images)
        
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Visualization
with torch.no_grad():
    sample_images = next(iter(dataloader))[0]
    reconstructed_images = vqgan(sample_images)

def show_images(original, reconstructed):
    fig, axes = plt.subplots(2, len(original), figsize=(12, 4))
    for i in range(len(original)):
        axes[0][i].imshow(original[i].permute(1, 2, 0).numpy())
        axes[0][i].axis('off')
        
        axes[1][i].imshow(reconstructed[i].permute(1, 2, 0).numpy())
        axes[1][i].axis('off')
    
    plt.show()

show_images(sample_images[:5], reconstructed_images[:5])


# In[ ]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.deconv1(x))
        x = nn.ReLU()(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Sigmoid to ensure output is in [0, 1]
        return x
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, z):
        z_flattened = z.view(-1, z.size(1))
        distances = torch.cdist(z_flattened.unsqueeze(0), self.embedding.weight.unsqueeze(0))
        encoding_indices = torch.argmin(distances, dim=2).view(z.size(0), -1)
        quantized = self.embedding(encoding_indices).view(z.size())
        return quantized, encoding_indices

class VQGAN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VQGAN, self).__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        quantized, _ = self.quantizer(z)
        reconstructed = self.decoder(quantized)
        return reconstructed


num_embeddings = 512
embedding_dim = 256
vqgan = VQGAN(num_embeddings, embedding_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(vqgan.parameters(), lr=0.0002)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_epochs = 5

for epoch in range(num_epochs):
    for images, _ in dataloader:
        optimizer.zero_grad()
        
        reconstructed_images = vqgan(images)
        
        # Calculate loss between original and reconstructed images
        loss = criterion(reconstructed_images.view(-1), images.view(-1))  # Flatten for MSELoss
        
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    sample_images = next(iter(dataloader))[0]
    reconstructed_images = vqgan(sample_images)

def show_images(original, reconstructed):
    fig, axes = plt.subplots(2, len(original), figsize=(12, 4))
    for i in range(len(original)):
        # Ensure values are clamped between [0, 1] for display
        axes[0][i].imshow(original[i].permute(1, 2, 0).clamp(0.0, 1.0).numpy())
        axes[0][i].axis('off')
        
        axes[1][i].imshow(reconstructed[i].permute(1, 2, 0).clamp(0.0, 1.0).numpy())
        axes[1][i].axis('off')
    
    plt.show()

show_images(sample_images[:5], reconstructed_images[:5])


# In[5]:


import torch
from IPython.display import display, Image as IPImage
import sys

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)  # Use float32 to avoid black images
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")  
prompt = "a photo of an astronaut riding a horse on mars"

with torch.no_grad():  
    image = pipe(prompt, num_inference_steps=50).images[0]  

image.save("astronaut_rides_horse.png")
display(IPImage("astronaut_rides_horse.png"))


# In[ ]:


##Integrating recorded speech along with image generation
import whisper
import torch
from IPython.display import display, Image as IPImage
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
whisper_model = whisper.load_model("base")
audio_file_path = r"C:\Users\sarve\Downloads\Recording1.mp3"
result = whisper_model.transcribe(audio_file_path)
prompt = result["text"]
print("Transcribed Prompt:", prompt)
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
with torch.no_grad():
    image = pipe(prompt, num_inference_steps=50).images[0]
image.save("generated_image.png")
display(IPImage("generated_image.png"))



# In[1]:


#Integrating in real-time speech along with image generation with sdm
import sounddevice as sd
import numpy as np
import queue
import threading
import torch
import whisper  
import time
from IPython.display import display, Image as IPImage
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

whisper_model = whisper.load_model("base")
audio_queue = queue.Queue()

def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    audio_queue.put(audio.flatten())  
    print("Recording finished.")

def transcribe_and_generate_image():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            result = whisper_model.transcribe(audio_tensor.numpy(), language='en')
            transcription = result['text']
            print("Transcription:")
            print(transcription)
            
            generate_image(transcription)

def generate_image(prompt):
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    image = pipe(prompt).images[0]
    image_path = f"{prompt.replace(' ', '_')}.png"
    image.save(image_path)
    display(IPImage(image_path))

transcription_thread = threading.Thread(target=transcribe_and_generate_image)
transcription_thread.daemon = True
transcription_thread.start()

def start_recording(total_duration=5, interval=5):
    start_time = time.time()
    try:
        while (time.time() - start_time) < total_duration:
            record_audio(duration=interval)
    except KeyboardInterrupt:
        print("Recording interrupted. Exiting...")

start_recording(total_duration=5, interval=5)


# In[3]:


pip install streamlit_jupyter


# In[2]:


# Integrating real-time speech along with image generation with ldm 
import sounddevice as sd
import numpy as np
import queue
import threading
import torch
import whisper  
import time
from IPython.display import display, Image as IPImage
from diffusers import DiffusionPipeline

whisper_model = whisper.load_model("base")
audio_queue = queue.Queue()

def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    audio_queue.put(audio.flatten())  
    print("Recording finished.")

def transcribe_and_generate_image():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            result = whisper_model.transcribe(audio_tensor.numpy(), language='en')
            transcription = result['text']
            print("Transcription:")
            print(transcription)
            
            generate_image(transcription)

def generate_image(prompt):
    ldm = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

    images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=7).images

    for idx, image in enumerate(images):
        image_path = f"{prompt.replace(' ', '_')}-{idx}.png"
        image.save(image_path)
        display(IPImage(image_path))

transcription_thread = threading.Thread(target=transcribe_and_generate_image)
transcription_thread.daemon = True
transcription_thread.start()

def start_recording(total_duration=5, interval=5):
    start_time = time.time()
    try:
        while (time.time() - start_time) < total_duration:
            record_audio(duration=interval)
    except KeyboardInterrupt:
        print("Recording interrupted. Exiting...")

start_recording(total_duration=5, interval=5)


# In[1]:


get_ipython().system('pip install ipywidgets sounddevice torch whisper diffusers')
get_ipython().system('jupyter nbextension enable --py widgetsnbextension')


# In[1]:


#UI using ipwidgets
import sounddevice as sd
import numpy as np
import queue
import threading
import torch
import whisper  
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from IPython.display import display, Image as IPImage, clear_output
import ipywidgets as widgets

whisper_model = whisper.load_model("base")
audio_queue = queue.Queue()
latest_transcription = ""  
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    audio_queue.put(audio.flatten())  
    print("Recording finished.")

def transcribe_and_generate_image():
    global latest_transcription 
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            result = whisper_model.transcribe(audio_tensor.numpy(), language='en')
            latest_transcription = result['text'] 
            print("Transcription:")
            print(latest_transcription)

def generate_image(prompt):
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    image = pipe(prompt).images[0]
    image_path = f"{prompt.replace(' ', '_')}.png"
    image.save(image_path)
    return image_path

def display_image(image_path):
    clear_output(wait=True)
    display(IPImage(image_path))

def on_record_button_clicked(b):
    with output_area:
        record_audio(duration=5)

def on_transcribe_button_clicked(b):
    with output_area:
        transcription_thread = threading.Thread(target=transcribe_and_generate_image)
        transcription_thread.daemon = True
        transcription_thread.start()

def on_generate_image_button_clicked(b):
    if latest_transcription:
        with output_area:
            image_path = generate_image(latest_transcription)  
            display_image(image_path)
    else:
        print("No transcription available. Please record audio first.")

def on_clear_button_clicked(b):
    with output_area:
        clear_output(wait=True)

record_button = widgets.Button(description="Record Audio")
transcribe_button = widgets.Button(description="Transcribe Audio")
generate_image_button = widgets.Button(description="Generate Image")
clear_button = widgets.Button(description="Clear Output")
output_area = widgets.Output()

record_button.on_click(on_record_button_clicked)
transcribe_button.on_click(on_transcribe_button_clicked)
generate_image_button.on_click(on_generate_image_button_clicked)
clear_button.on_click(on_clear_button_clicked)

button_box = widgets.HBox([record_button, transcribe_button, generate_image_button, clear_button])
display(button_box, output_area)


# In[ ]:


#ui using streamlit
get_ipython().system('pip install --upgrade transformers')
from diffusers import StableDiffusionPipeline
get_ipython().system('pip show diffusersimport streamlit as st')
import sounddevice as sd
import numpy as np
import queue
import threading
import torch
import whisper  
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import time
import subprocess

whisper_model = whisper.load_model("base")
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

audio_queue = queue.Queue()
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []
if 'cached_images' not in st.session_state:
    st.session_state.cached_images = {}

def record_audio(duration=5, sample_rate=16000):
    st.write("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    audio_queue.put(audio.flatten())  
    st.write("Recording finished.")

def transcribe_and_generate_image():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor /= torch.max(torch.abs(audio_tensor))
            result = whisper_model.transcribe(audio_tensor.numpy(), language='en')
            transcription = result['text']
            
            if transcription not in st.session_state.transcriptions:
                st.session_state.transcriptions.append(transcription)
                generate_image(transcription)

def generate_image(prompt):
    if prompt in st.session_state.cached_images:
        image_path = st.session_state.cached_images[prompt]
    else:
        image = pipe(prompt).images[0]
        image_path = f"{prompt.replace(' ', '_')}.png"
        image.save(image_path)
        
        st.session_state.cached_images[prompt] = image_path
    
    st.image(image_path, caption="Generated Image", use_column_width=True)

transcription_thread = threading.Thread(target=transcribe_and_generate_image)
transcription_thread.daemon = True
transcription_thread.start()

st.title("Real-Time Speech to Image Generator")

total_duration = st.slider("Total Recording Duration (seconds)", 5, 60, 5)
interval = st.slider("Recording Interval (seconds)", 1, 10, 5)

if st.button("Start Recording"):
    start_time = time.time()
    try:
        while (time.time() - start_time) < total_duration:
            record_audio(duration=interval)
    except KeyboardInterrupt:
        st.write("Recording interrupted. Exiting...")

if st.button("Show Transcription and Generate Image"):
    if st.session_state.transcriptions:
        latest_transcription = st.session_state.transcriptions[-1]
        st.write("Latest Transcription:")
        st.write(latest_transcription)
        generate_image(latest_transcription)
    else:
        st.write("No transcriptions available. Please record some audio first.")

streamlit_code = """
import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
import torch
import whisper  
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# Load models
whisper_model = whisper.load_model("base")
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Initialize audio queue and session state
audio_queue = queue.Queue()
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []
if 'cached_images' not in st.session_state:
    st.session_state.cached_images = {}

# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    st.write("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    audio_queue.put(audio.flatten())  
    st.write("Recording finished.")

# Function to transcribe audio and generate images
def transcribe_and_generate_image():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor /= torch.max(torch.abs(audio_tensor))
            result = whisper_model.transcribe(audio_tensor.numpy(), language='en')
            transcription = result['text']
            
            # Store transcription in session state
            if transcription not in st.session_state.transcriptions:
                st.session_state.transcriptions.append(transcription)
                generate_image(transcription)

# Function to generate images based on prompt
def generate_image(prompt):
    # Check if image is already cached
    if prompt in st.session_state.cached_images:
        image_path = st.session_state.cached_images[prompt]
    else:
        image = pipe(prompt).images[0]
        image_path = f"{prompt.replace(' ', '_')}.png"
        image.save(image_path)
        
        # Cache the generated image path
        st.session_state.cached_images[prompt] = image_path
    
    st.image(image_path, caption="Generated Image", use_column_width=True)

# Start transcription thread
transcription_thread = threading.Thread(target=transcribe_and_generate_image)
transcription_thread.daemon = True
transcription_thread.start()

# Streamlit UI setup
st.title("Real-Time Speech to Image Generator")

total_duration = st.slider("Total Recording Duration (seconds)", 5, 60, 5)
interval = st.slider("Recording Interval (seconds)", 1, 10, 5)

if st.button("Start Recording"):
    start_time = time.time()
    try:
        while (time.time() - start_time) < total_duration:
            record_audio(duration=interval)
    except KeyboardInterrupt:
        st.write("Recording interrupted. Exiting...")

if st.button("Show Transcription and Generate Image"):
    if st.session_state.transcriptions:
        latest_transcription = st.session_state.transcriptions[-1]
        st.write("Latest Transcription:")
        st.write(latest_transcription)
        generate_image(latest_transcription)
    else:
        st.write("No transcriptions available. Please record some audio first.")
"""

with open('app.py', 'w') as f:
    f.write(streamlit_code)

def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py"])

run_streamlit()

