import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import time

from config import ReMoDAConfig
from model import ReMoDAForSequenceClassification

# Parameters
BATCH_SIZE = 8
SEQ_LEN = 128
MAX_TRAIN_SAMPLES = 500 # Small subset for sandbox CPU
MAX_EVAL_SAMPLES = 100
EPOCHS = 3

class ClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

import numpy as np

def get_dataloaders(seed=42):
    print(f"Loading IMDb dataset (seed={seed})...")
    # Using small subset of IMDb for text classification
    dataset = load_dataset("imdb")

    train_dataset = dataset["train"].shuffle(seed=seed).select(range(MAX_TRAIN_SAMPLES))
    eval_dataset = dataset["test"].shuffle(seed=seed).select(range(MAX_EVAL_SAMPLES))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=SEQ_LEN)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    train_encodings = {
        'input_ids': train_dataset['input_ids']
    }
    eval_encodings = {
        'input_ids': eval_dataset['input_ids']
    }

    train_dataset_obj = ClassificationDataset(
        {'input_ids': train_encodings['input_ids']},
        train_dataset['label']
    )
    eval_dataset_obj = ClassificationDataset(
        {'input_ids': eval_encodings['input_ids']},
        eval_dataset['label']
    )

    train_loader = DataLoader(train_dataset_obj, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset_obj, batch_size=BATCH_SIZE)

    return train_loader, eval_loader, tokenizer.vocab_size

def get_config(vocab_size):
    return ReMoDAConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=SEQ_LEN,
        use_rt_kv=True,
        use_moda=True,
        depth_slots=2
    )

def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        input_ids = batch['input_ids']
        labels = batch['labels']

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    acc = correct / total
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, eval_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in eval_loader:
        input_ids = batch['input_ids']
        labels = batch['labels']

        loss, logits = model(input_ids, labels=labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(eval_loader)
    acc = correct / total
    return avg_loss, acc

def main():
    seeds = [42, 100, 1234]
    print(f"Running NLP Evaluation over seeds: {seeds}")

    all_eval_accs = []
    all_eval_losses = []

    for seed in seeds:
        print(f"\n--- Running Seed: {seed} ---")

        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_loader, eval_loader, vocab_size = get_dataloaders(seed)

        config = get_config(vocab_size)
        print("Initializing ReMoDAForSequenceClassification...")
        model = ReMoDAForSequenceClassification(config, num_labels=2)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        print("Training NLP Task (IMDb Classification)...")
        start_time = time.time()
        for epoch in range(EPOCHS):
            train_loss, train_acc = train(model, train_loader, optimizer)
            eval_loss, eval_acc = evaluate(model, eval_loader)

            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Eval Loss:  {eval_loss:.4f} | Eval Acc:  {eval_acc:.4f}")

        all_eval_accs.append(eval_acc)
        all_eval_losses.append(eval_loss)
        print(f"Seed {seed} Completed in {time.time() - start_time:.2f}s")

    print("\n==================================================")
    print(f"Final NLP Task Evaluation Results (Across {len(seeds)} Seeds):")
    print(f"Mean Final Accuracy: {np.mean(all_eval_accs):.4f} | Std Final Accuracy: {np.std(all_eval_accs):.4f}")
    print(f"Mean Final Loss:     {np.mean(all_eval_losses):.4f} | Std Final Loss:     {np.std(all_eval_losses):.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
