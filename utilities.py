
import matplotlib.pyplot as plt
import torch

from main import load_texts
from tokenizer import SimpleTokenizer
from transformer import TransformerEncoder, TransformerLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.eval

    def sanity_check(self, sentence, block_size, model_type="encoder"):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        # attn_maps: list of length num_layers, each (batch, num_heads, seq, seq)
        if model_type == "encoder":
            _, attn_maps = self.model(input_tensor, mask=torch.ones(1, block_size).bool().to(device))
        elif model_type == "decoder":
            _, attn_maps = self.model(input_tensor)
        num_layers = len(attn_maps)
        num_heads = attn_maps[0].shape[1]
        print("Number of attention maps:", num_layers * num_heads, f"({num_layers} layers x {num_heads} heads)")

        # Build one figure: rows = Head j, columns = Layer i
        fig, axes = plt.subplots(num_heads, num_layers, figsize=(3 * num_layers, 3 * num_heads))
        if num_heads == 1 and num_layers == 1:
            axes = [[axes]]
        elif num_heads == 1:
            axes = [axes]
        elif num_layers == 1:
            axes = [[ax] for ax in axes]

        for head_idx in range(num_heads):
            for layer_idx in range(num_layers):
                ax = axes[head_idx][layer_idx]
                att = attn_maps[layer_idx][0, head_idx].detach().cpu().numpy()
                im = ax.imshow(att, cmap="hot", interpolation="nearest")
                ax.xaxis.tick_top()
                if head_idx == 0:
                    ax.set_title(f"Layer {layer_idx}")
                if layer_idx == 0:
                    ax.set_ylabel(f"Head {head_idx}")
                # Optional: normalization check
                total_prob = torch.sum(attn_maps[layer_idx][0, head_idx], dim=1)
                if torch.any(total_prob < 0.99) or torch.any(total_prob > 1.01):
                    ax.set_title(ax.get_title() + " (!)")

        plt.tight_layout()
        plt.savefig(f"attention_maps_all_{model_type}.png")
        plt.show()
            

if __name__ == "__main__":
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a 
    
    # ----------------------------  Encoder  ----------------------------
    encoder = TransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        num_layers=4,
        d_model=64,
        num_heads=2,
        pe_type="absolute",
        max_seq_len=32,
        theta=10000.0,
        device=device
    )
    state_dict = torch.load("checkpoints/classifier.pt", map_location=device)
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    encoder.load_state_dict(state_dict)

    utilities = Utilities(tokenizer, encoder)
    utilities.sanity_check("Today we stand united in purpose and conviction, determined to strengthen our democracy, protect our shared freedoms, and build an economy that rewards hard work, restores opportunity, and renews faith in our common future.", 32, model_type="encoder")


    # ----------------------------  Decoder  ----------------------------
    decoder = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        num_layers=4,
        d_model=64,
        num_heads=2,
        pe_type="absolute",
        max_seq_len=32,
        theta=10000.0,
        device=device
    )
    state_dict = torch.load("checkpoints/lm.pt", map_location=device)

    decoder.load_state_dict(state_dict)

    utilities = Utilities(tokenizer, decoder)
    utilities.sanity_check("Today we stand united in purpose and conviction, determined to strengthen our democracy, protect our shared freedoms, and build an economy that rewards hard work, restores opportunity, and renews faith in our common future.", 32, model_type="decoder")