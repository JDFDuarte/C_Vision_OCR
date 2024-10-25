import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, load_dataset

# Function to parse .inkml files and convert them to images and labels
def parse_inkml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract strokes and text
    strokes = []
    for trace in root.findall('.//{http://www.w3.org/2003/InkML}trace'):
        points = [tuple(map(float, p.split())) for p in trace.text.strip().split(',')]
        strokes.append(points)
    
    # Extract label (this part may vary depending on your dataset structure)
    label = root.find('.//annotation').text
    
    return strokes, label

# Function to render strokes into an image
def strokes_to_image(strokes):
    img = Image.new('L', (500, 500), 'white')
    draw = ImageDraw.Draw(img)
    
    for stroke in strokes:
        draw.line(stroke, fill='black', width=3)
    
    return img

# Convert .inkml files to dataset format
def create_dataset_from_inkml(folder_path):
    data = {'image': [], 'text': []}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.inkml'):
            file_path = os.path.join(folder_path, filename)
            strokes, label = parse_inkml(file_path)
            image = strokes_to_image(strokes)
            data['image'].append(image)
            data['text'].append(label)
    
    return Dataset.from_dict(data)

# Load datasets
train_dataset = create_dataset_from_inkml('data/mathwriting-2024-excerpt/train')
valid_dataset = create_dataset_from_inkml('data/mathwriting-2024-excerpt/valid')

# Preprocess function for the dataset
def preprocess_data(examples):
    images = [processor(image, return_tensors="pt").pixel_values.squeeze() for image in examples['image']]
    texts = examples['text']
    
    encoding = processor(images=images, text=texts, padding="max_length", truncation=True)
    
    return encoding

# Initialize processor and model
model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

# Apply preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_data, batched=True)
valid_dataset = valid_dataset.map(preprocess_data, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
output_dir = "./trocr_finetuned"
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")

# Function for inference with the fine-tuned model
def recognize_text(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Example usage of the inference function
test_image_path = "path/to/test/image.jpg"
recognized_text = recognize_text(test_image_path)
print(f"Recognized text: {recognized_text}")