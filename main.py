import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from torch import nn
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerClassifier
import argparse
import wandb

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch, block_size):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    attention_mask = padded_sequences.ne(0)
    labels = torch.stack(labels)
    return {
        "input_ids": padded_sequences,
        "labels": labels,
        "attention_mask": attention_mask
    }

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
            log_probs = classifier(input_ids, mask=attention_mask)
            _, predicted = torch.max(log_probs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--pe_type", type=str, default="absolute")
    parser.add_argument("--theta", type=float, default=10000.0)
    # LM / transformer hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Number of independent sequences to process in parallel")
    parser.add_argument("--block_size", type=int, default=32, help="Maximum context length for predictions")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--n_embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    # LM evaluation
    parser.add_argument("--eval_interval", type=int, default=100, help="How often to evaluate train and test perplexity during training")
    parser.add_argument("--max_iters", type=int, default=500, help="Max training iterations for language modeling")
    parser.add_argument("--eval_iters", type=int, default=200, help="Number of iterations to evaluate perplexity on the test set")
    # Classifier hyperparameters
    parser.add_argument("--n_input", type=int, default=64, help="Input size for the classifier, should match n_embd")
    parser.add_argument("--n_hidden", type=int, default=100, help="Hidden size for the classifier")
    parser.add_argument("--n_output", type=int, default=3, help="Output size for the classifier (number of classes)")
    parser.add_argument("--epochs_CLS", type=int, default=15, help="Epochs for classifier training")
    parser.add_argument("--cls_mode", type=str, default="mean", help="Mode for the classifier")
    parser.add_argument("--save_cls", type=str, default=None, help="Path to save classifier state_dict after training (e.g. classifier.pt)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="CSE256_PA2", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional)")
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        wandb.define_metric("train/step")
        wandb.define_metric("train/loss", step_metric="train/step")

        wandb.define_metric("val/epoch")
        wandb.define_metric("val/accuracy", step_metric="val/epoch")

    # ----------------------------  Loading data and creating tokenizer  ----------------------------
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    # ----------------------------  Classification Task  --------------------------------
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    train_CLS_loader = DataLoader(
        train_CLS_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_batch(b, args.block_size),
        shuffle=True,
    )
    test_CLS_loader = DataLoader(
        test_CLS_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_batch(b, args.block_size),
        shuffle=False,
    )
    classifier = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        num_layers=args.n_layer,
        d_model=args.n_embd,
        num_heads=args.n_head,
        pe_type=args.pe_type,
        mode=args.cls_mode,
        max_seq_len=args.block_size,
        theta=args.theta,
        device=device
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()
    global_step = 0
    for epoch in range(args.epochs_CLS):
        for step, batch in enumerate(train_CLS_loader):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            log_probs = classifier(input_ids, mask=attention_mask)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": global_step
                }, step=global_step)
            global_step += 1
        accuracy = compute_classifier_accuracy(classifier, test_CLS_loader)
        if args.wandb:
            wandb.log({
                "val/accuracy": accuracy,
                "val/epoch": epoch + 1
            })
        print(f"Epoch {epoch + 1}, Step {step + 1}, Classifier accuracy: {accuracy:.2f}%")

    if args.save_cls:
        torch.save(classifier.state_dict(), args.save_cls)
        print(f"Saved classifier to {args.save_cls}")

    # ----------------------------  Language Modeling Task  ----------------------------
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, args.block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=args.batch_size, shuffle=True)

     # for the classification  task, you will train for a fixed number of epochs like this:





    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= args.max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here

    



if __name__ == "__main__":
    main()
