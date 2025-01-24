import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from langchain.prompts import PromptTemplate
import json
import os

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from the dataset.")
    return Dataset.from_dict({
        "instruction": [x["instruction"] for x in data],
        "output": [x["output"] for x in data]
    })

file_path = "/Users/kapilgund/Code/persona/this/prompt.json"  # Replace with your actual path
dataset = load_dataset(file_path)

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

# Preprocessing function
def preprocess(examples):
    combined = [
        f"Instruction: {instruction}\nResponse: {output}" 
        for instruction, output in zip(examples["instruction"], examples["output"])
    ]
    tokenized = tokenizer(combined, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True)
print("Tokenized Dataset:\n", tokenized_dataset)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train the model
try:
    print("Starting training...")
    trainer.train()
    print("Training completed successfully.")
except Exception as e:
    print(f"Error during training: {e}")

# Save the fine-tuned model
model_dir = "./fine_tuned_model"
try:
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}.")
except Exception as e:
    print(f"Error saving the model: {e}")

# Export to ONNX
onnx_export_path = "./fine_tuned_model/model.onnx"
dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128)).to(device)

# try:
#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_export_path,
#         input_names=["input_ids"],
#         output_names=["logits"],
#         opset_version=14,
#         dynamic_axes={
#             "input_ids": {0: "batch_size", 1: "sequence_length"},
#             "logits": {0: "batch_size", 1: "sequence_length"},
#         },
#     )
#     print(f"Model exported to ONNX format at {onnx_export_path}")
# except Exception as e:
#     print(f"Error exporting model to ONNX: {e}")

    
try:
    model.eval()  # Set model to evaluation mode
    model.to("cpu")  # Move model to CPU
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128)).to("cpu")  

    torch.onnx.export(
        model,
        dummy_input,
        onnx_export_path,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=14,  # Use an appropriate ONNX opset version
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )
    print(f"Model exported to ONNX format at {onnx_export_path}")
except Exception as e:
    print(f"Error exporting model to ONNX: {e}")

