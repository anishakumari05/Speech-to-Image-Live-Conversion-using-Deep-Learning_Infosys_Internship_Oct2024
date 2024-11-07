import os
import pandas as pd
import streamlit as st
import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.nn.utils.rnn import pad_sequence

# Streamlit UI
st.title("Whisper Fine-Tuning Interface")

st.write("This interface allows you to fine-tune the Whisper model on a custom audio dataset.")

# File paths
csv_path = st.text_input("Enter path to the CSV file with audio metadata:", 'C:/Users/vishw/Downloads/Dataset/Dataset/Recordings/audio__details.csv')
audio_folder_path = st.text_input("Enter path to the audio folder:", 'C:/Users/vishw/Downloads/Dataset/Dataset/Recordings/Train')

# Load dataset and prepare it for training
st.write("Loading and preparing the dataset...")
df = pd.read_csv(csv_path)
df['File_name'] = df['File_name'].apply(lambda x: os.path.abspath(os.path.join(audio_folder_path, os.path.basename(x))))
df.to_csv(csv_path, index=False)  # Save the updated CSV

dataset = load_dataset('csv', data_files=csv_path)
dataset = dataset.cast_column('File_name', Audio(sampling_rate=16000))
dataset = dataset.rename_column('File_name', 'audio')
dataset = dataset.rename_column('phrase', 'sentence')

# Load processor
st.write("Loading Whisper Processor...")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Function to preprocess audio and text
def prepare_dataset(batch):
    batch["input_features"] = processor(batch["audio"]["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# Filter columns and preprocess dataset
columns_to_remove = [
    'audio_clipping', 'audio_clipping:confidence', 'background_noise_audible', 
    'background_noise_audible:confidence', 'overall_quality_of_the_audio', 
    'quiet_speaker', 'quiet_speaker:confidence', 'speaker_id', 'file_download', 
    'prompt', 'writer_id'
]
columns_to_remove = [col for col in columns_to_remove if col in dataset.column_names]
dataset = dataset.map(prepare_dataset, remove_columns=columns_to_remove)

# Load model
st.write("Loading Whisper Model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Training parameters
st.subheader("Training Parameters")
output_dir = st.text_input("Output directory for fine-tuned model:", "whisper-finetuned")
batch_size = st.slider("Per-device train batch size:", 1, 16, 8)
gradient_accumulation_steps = st.slider("Gradient accumulation steps:", 1, 8, 4)
learning_rate = st.number_input("Learning rate:", min_value=1e-6, max_value=1e-3, value=1e-5, step=1e-6, format="%.6f")
num_train_epochs = st.slider("Number of training epochs:", 1, 10, 3)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    fp16=False,
    save_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
)

# Data Collator
class DataCollatorForWhisper:
    def __call__(self, features):
        input_features = [torch.tensor(feature["input_features"]) for feature in features]
        labels = [torch.tensor(feature["labels"]) for feature in features]
        input_features_padded = pad_sequence(input_features, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_features": input_features_padded, "labels": labels_padded}

data_collator = DataCollatorForWhisper()

# Fine-tuning
if st.button("Start Training"):
    st.write("Initializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
    )
    
    st.write("Starting training...")
    trainer.train()
    
    st.write("Saving fine-tuned model...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    st.success("Model fine-tuning completed and saved!")

