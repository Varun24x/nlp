import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import wandb
import logging
from pathlib import Path

class SQLDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Dataset for NL to SQL fine-tuning
        :param data_path: Path to the CSV file containing 'nl_query' and 'sql_query'
        :param tokenizer: Pretrained T5 tokenizer
        :param max_length: Maximum sequence length for encoding
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = f"translate to sql: {row['nl_query']}"
        target_text = row['sql_query']

        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def train_model(train_data_paths, val_data_path, model_save_path, batch_size=8, num_epochs=10, learning_rate=5e-5):
    """
    Fine-tune a T5 model on NL-to-SQL tasks.
    :param train_data_paths: List of paths to training data CSVs
    :param val_data_path: Path to validation data CSV
    :param model_save_path: Path to save the fine-tuned model
    :param batch_size: Batch size for training
    :param num_epochs: Number of epochs to train
    :param learning_rate: Learning rate for optimizer
    """
    # Initialize wandb
    wandb.init(project="nl-to-sql")

    # Initialize model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # Load and combine datasets
    train_dfs = [pd.read_csv(path) for path in train_data_paths]
    combined_train_data = pd.concat(train_dfs, ignore_index=True)
    combined_train_data.to_csv("data/combined_train_data.csv", index=False)  # Save for record

    train_dataset = SQLDataset("data/combined_train_data.csv", tokenizer)
    val_dataset = SQLDataset(val_data_path, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

        # Log metrics
        wandb.log({
            'train_loss': total_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader)
        })

    # Save model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create necessary directories
    Path("models/t5_spider_finetuned").mkdir(parents=True, exist_ok=True)
    
    train_model(
        train_data_paths=[
            'data/train_data.csv',  # Spider training data
            'data/custom_data.csv'  # Custom data
        ],
        val_data_path='data/dev_data.csv',
        model_save_path='models/t5_spider_finetuned'
    )
