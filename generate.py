import torch
from transformers import AutoTokenizer
from model import ReMoDAConfig, ReMoDAModel

def generate_text(model, tokenizer, prompt, max_new_tokens=20):
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"Prompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...\n")

    print(prompt, end="", flush=True)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # In a full production implementation, we would pass `past_key_values`
            # and only compute the latest token.
            # For demonstration of causal ReMoDA generation, we can recompute the
            # sequence at each step. ReMoDA's internal kv_cache will be built
            # appropriately up to the current sequence length.

            # ReMoDA manages its own internal `kv_cache` per layer across sequence depth.
            # We don't need to manually pass it in from the outside for the simple
            # forward recompute method.
            _, logits = model(input_ids)

            # Get the predictions for the next token
            next_token_logits = logits[0, -1, :]

            # Greedy decoding
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)

            # Decode and print
            next_token_str = tokenizer.decode(next_token_id[0])
            print(next_token_str, end="", flush=True)

            # Append to input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    print("\n\nGeneration complete.")

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    import os
    print("Initializing ReMoDA model...")

    # Needs to match the config used in train.py for ReMoDA
    config = ReMoDAConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        use_rt_kv=True,
        use_moda=True,
        depth_slots=2
    )
    model = ReMoDAModel(config)

    if os.path.exists("remoda_model.pt"):
        print("Loading trained weights from remoda_model.pt...")
        model.load_state_dict(torch.load("remoda_model.pt", map_location="cpu"))
    else:
        print("Warning: remoda_model.pt not found. Using untrained weights.")

    prompt = "Once upon a time"
    generate_text(model, tokenizer, prompt, max_new_tokens=30)

if __name__ == "__main__":
    main()
