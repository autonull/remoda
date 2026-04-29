import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class TinyStoriesDataset(Dataset):
    def __init__(self, encodings, seq_len):
        self.input_ids = encodings['input_ids']
        self.seq_len = seq_len

        # Calculate how many full sequences we can extract
        self.num_sequences = len(self.input_ids) // self.seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        chunk = self.input_ids[start_idx:end_idx]

        # input_ids and labels are the same (shifted internally by the model)
        return torch.tensor(chunk, dtype=torch.long), torch.tensor(chunk, dtype=torch.long)

def get_dataloaders(batch_size=16, seq_len=256, max_samples=100000):
    """
    Loads TinyStories, tokenizes, and returns train/val DataLoaders.
    Using a subset for rapid convergence demonstration.
    """
    print("Loading TinyStories dataset...")
    # Use streaming or a small slice to avoid memory issues
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{max_samples}]")

    # We use GPT-2 tokenizer as a standard fast tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)

    # Flatten the dataset into a single continuous stream of tokens
    print("Flattening tokens...")
    all_input_ids = []
    for item in tokenized_datasets:
        all_input_ids.extend(item['input_ids'])

    # Split into train/val (90/10)
    split_idx = int(len(all_input_ids) * 0.9)
    train_ids = all_input_ids[:split_idx]
    val_ids = all_input_ids[split_idx:]

    print(f"Total tokens: Train={len(train_ids)}, Val={len(val_ids)}")

    train_dataset = TinyStoriesDataset({'input_ids': train_ids}, seq_len)
    val_dataset = TinyStoriesDataset({'input_ids': val_ids}, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer.vocab_size

if __name__ == "__main__":
    train_loader, val_loader, vocab_size = get_dataloaders(batch_size=4, seq_len=128, max_samples=1000)
    for x, y in train_loader:
        print(f"Batch shape: {x.shape}")
        break
    print(f"Vocab size: {vocab_size}")
