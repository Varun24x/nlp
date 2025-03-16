import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
from typing import Dict
from tqdm import tqdm


class FeedbackDataset(Dataset):
    def __init__(self, feedback_path: str, tokenizer, max_length: int = 512):
        """
        Dataset class for feedback-based training data.

        :param feedback_path: Path to the feedback CSV file.
        :param tokenizer: Tokenizer for T5 model.
        :param max_length: Maximum token length for input and output.
        """
        # Load and clean the dataset
        self.data = pd.read_csv(feedback_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Sanitize the input data
        self.data['nl_query'] = self.data['nl_query'].apply(lambda x: str(x) if isinstance(x, str) else '')
        self.data['sql_query'] = self.data['sql_query'].apply(lambda x: str(x) if isinstance(x, str) else '')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary with input IDs, attention mask, and labels.
        """
        row = self.data.iloc[idx]
        input_text = row['nl_query']
        target_text = row['sql_query']

        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }


def retrain_model(
    model_path: str,
    feedback_path: str,
    output_path: str,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-5
):
    """
    Retrains a fine-tuned T5 model using feedback data.

    :param model_path: Path to the pre-trained model directory.
    :param feedback_path: Path to the feedback data CSV file.
    :param output_path: Path to save the retrained model.
    :param num_epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.
    """
    try:
        # Load model and tokenizer
        logging.info(f"Loading model from {model_path}")
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)

        # Load dataset and create DataLoader
        logging.info(f"Loading feedback dataset from {feedback_path}")
        dataset = FeedbackDataset(feedback_path, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup device and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Training loop
        logging.info("Starting retraining process")
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            logging.info(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Save retrained model
        logging.info(f"Saving retrained model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logging.info("Model and tokenizer saved successfully.")

    except Exception as e:
        logging.error(f"Error during retraining: {e}")
        raise


def main():
    logging.basicConfig(level=logging.INFO)

    # Ensure the output directory exists
    output_path = Path("models/t5_spider_finetuned_retrained")
    output_path.mkdir(parents=True, exist_ok=True)

    # Path to the feedback data
    feedback_path = "data/custom_data.csv"  # Ensure this file exists and contains the required columns

    if not Path(feedback_path).is_file():
        logging.error(f"Feedback file not found: {feedback_path}")
        return

    # Retrain model
    retrain_model(
        model_path="models/t5_spider_finetuned",
        feedback_path=feedback_path,
        output_path=str(output_path),
        num_epochs=5,
        batch_size=4,
        learning_rate=1e-5
    )


if __name__ == "__main__":
    main()
