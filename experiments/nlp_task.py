import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import time
from sklearn.metrics import f1_score, precision_score, recall_score

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

def get_standard_config(vocab_size):
    return ReMoDAConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=SEQ_LEN,
        use_rt_kv=False,
        use_moda=False
    )

def get_rt_config(vocab_size):
    return ReMoDAConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=SEQ_LEN,
        use_rt_kv=True,
        use_moda=False
    )

def get_moda_config(vocab_size):
    return ReMoDAConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=SEQ_LEN,
        use_rt_kv=False,
        use_moda=True,
        depth_slots=2
    )

def get_config(vocab_size):
    return ReMoDAConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=2, # ReMoDA can be fewer layers for same performance.
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
    all_preds = []
    all_labels = []

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

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1, precision, recall

@torch.no_grad()
def evaluate(model, eval_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in eval_loader:
        input_ids = batch['input_ids']
        labels = batch['labels']

        loss, logits = model(input_ids, labels=labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(eval_loader)
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1, precision, recall

def main():
    import matplotlib.pyplot as plt

    seeds = [42, 100, 1234]
    print(f"Running NLP Evaluation over seeds: {seeds}")

    architectures = ["Standard", "RT", "MoDA", "ReMoDA"]
    results = {arch: {"accs": [], "losses": [], "f1s": [], "precisions": [], "recalls": []} for arch in architectures}

    for seed in seeds:
        print(f"\n--- Running Seed: {seed} ---")

        train_loader, eval_loader, vocab_size = get_dataloaders(seed)

        for arch in architectures:
            print(f"\n[Evaluating Architecture: {arch}]")

            # Set seeds for reproducibility per architecture run
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if arch == "Standard":
                config = get_standard_config(vocab_size)
            elif arch == "RT":
                config = get_rt_config(vocab_size)
            elif arch == "MoDA":
                config = get_moda_config(vocab_size)
            elif arch == "ReMoDA":
                config = get_config(vocab_size)

            print(f"Initializing SequenceClassification model with {arch} config...")
            model = ReMoDAForSequenceClassification(config, num_labels=2)

            optimizer = optim.AdamW(model.parameters(), lr=1e-3)

            print(f"Training NLP Task (IMDb Classification) - {arch}...")
            start_time = time.time()
            for epoch in range(EPOCHS):
                train_loss, train_acc, train_f1, train_prec, train_rec = train(model, train_loader, optimizer)
                eval_loss, eval_acc, eval_f1, eval_prec, eval_rec = evaluate(model, eval_loader)

                print(f"Epoch {epoch+1}/{EPOCHS}")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Train Prec: {train_prec:.4f} | Train Rec: {train_rec:.4f}")
                print(f"  Eval Loss:  {eval_loss:.4f} | Eval Acc:  {eval_acc:.4f} | Eval F1:  {eval_f1:.4f} | Eval Prec: {eval_prec:.4f} | Eval Rec: {eval_rec:.4f}")

            results[arch]["accs"].append(eval_acc)
            results[arch]["losses"].append(eval_loss)
            results[arch]["f1s"].append(eval_f1)
            results[arch]["precisions"].append(eval_prec)
            results[arch]["recalls"].append(eval_rec)
            print(f"Architecture {arch} Seed {seed} Completed in {time.time() - start_time:.2f}s")

    print("\n=================================================================")
    print(f"Final NLP Task Evaluation Results (Across {len(seeds)} Seeds):")
    print("-----------------------------------------------------------------")
    for arch in architectures:
        print(f"{arch} Architecture:")
        print(f"  Mean Final Accuracy:  {np.mean(results[arch]['accs']):.4f} ± {np.std(results[arch]['accs']):.4f}")
        print(f"  Mean Final F1 Score:  {np.mean(results[arch]['f1s']):.4f} ± {np.std(results[arch]['f1s']):.4f}")
        print(f"  Mean Final Precision: {np.mean(results[arch]['precisions']):.4f} ± {np.std(results[arch]['precisions']):.4f}")
        print(f"  Mean Final Recall:    {np.mean(results[arch]['recalls']):.4f} ± {np.std(results[arch]['recalls']):.4f}")
        print(f"  Mean Final Loss:      {np.mean(results[arch]['losses']):.4f} ± {np.std(results[arch]['losses']):.4f}")
        print("-----------------------------------------------------------------")
    print("=================================================================")

    # Plot Demonstrability Chart
    means = [np.mean(results[arch]['f1s']) for arch in architectures]
    stds = [np.std(results[arch]['f1s']) for arch in architectures]

    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(architectures))
    plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, color=['blue', 'orange', 'green', 'red'])
    plt.ylabel('F1 Score')
    plt.xticks(x_pos, architectures)
    plt.title('NLP Task Evaluation: IMDb Classification')
    plt.tight_layout()
    plt.savefig('nlp_results.png')
    print("\nSaved NLP evaluation chart to 'nlp_results.png'.")

if __name__ == "__main__":
    main()
