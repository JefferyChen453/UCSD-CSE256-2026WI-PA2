import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from torch import nn
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerClassifier, TransformerLM
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


def compute_perplexity(decoderLMmodel, data_loader, criterion, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        logits, _ = decoderLMmodel(X)
        loss = criterion(logits.view(-1, logits.shape[-1]), Y.view(-1))
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
    parser.add_argument("--mode", type=str, default="both", choices=["cls", "lm", "both"])
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
    parser.add_argument("--eval_iters", type=int, default=100, help="Number of iterations to evaluate perplexity on the test set")
    # Classifier hyperparameters
    parser.add_argument("--n_input", type=int, default=64, help="Input size for the classifier, should match n_embd")
    parser.add_argument("--n_hidden", type=int, default=100, help="Hidden size for the classifier")
    parser.add_argument("--n_output", type=int, default=3, help="Output size for the classifier (number of classes)")
    parser.add_argument("--epochs_CLS", type=int, default=15, help="Epochs for classifier training")
    parser.add_argument("--cls_mode", type=str, default="mean", help="Mode for the classifier")
    parser.add_argument("--save_cls", type=str, default=None, help="Path to save classifier state_dict after training (e.g. classifier.pt)")
    parser.add_argument("--save_lm", type=str, default=None, help="Path to save language model state_dict after training (e.g. decoderLMmodel.pt)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="CSE256_PA2", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional)")
    args = parser.parse_args()

    # ----------------------------  Loading data and creating tokenizer  ----------------------------
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    # ----------------------------  Classification Task  --------------------------------
    if args.mode == "cls" or args.mode == "both":
        if args.wandb:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            wandb.define_metric("train/step")
            wandb.define_metric("train/loss", step_metric="train/step")

            wandb.define_metric("val/epoch")
            wandb.define_metric("val/accuracy", step_metric="val/epoch")
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
        wandb.finish()

    # ----------------------------  Language Modeling Task  ----------------------------
    if args.mode == "lm" or args.mode == "both":
        if args.wandb:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            wandb.define_metric("train/iter")
            wandb.define_metric("train/loss", step_metric="train/iter")
            wandb.define_metric("val/perplexity_hbush", step_metric="train/iter")
            wandb.define_metric("val/perplexity_obama", step_metric="train/iter")
            wandb.define_metric("val/perplexity_wbush", step_metric="train/iter")
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        
        with open("speechesdataset/test_LM_hbush.txt", 'r', encoding='utf-8') as f:
            lmtestText_hbush = f.read()
        with open("speechesdataset/test_LM_obama.txt", 'r', encoding='utf-8') as f:
            lmtestText_obama = f.read()
        with open("speechesdataset/test_LM_wbush.txt", 'r', encoding='utf-8') as f:
            lmtestText_wbush = f.read()
        
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, args.block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=args.batch_size, shuffle=True)
        test_LM_dataset_hbush = LanguageModelingDataset(tokenizer, lmtestText_hbush, args.block_size)
        test_LM_dataset_obama = LanguageModelingDataset(tokenizer, lmtestText_obama, args.block_size)
        test_LM_dataset_wbush = LanguageModelingDataset(tokenizer, lmtestText_wbush, args.block_size)
        test_LM_loader_hbush = DataLoader(test_LM_dataset_hbush, batch_size=args.batch_size, shuffle=False)
        test_LM_loader_obama = DataLoader(test_LM_dataset_obama, batch_size=args.batch_size, shuffle=False)
        test_LM_loader_wbush = DataLoader(test_LM_dataset_wbush, batch_size=args.batch_size, shuffle=False)

        decoderLMmodel = TransformerLM(
            vocab_size=tokenizer.vocab_size,
            num_layers=args.n_layer,
            d_model=args.n_embd,
            num_heads=args.n_head,
            pe_type=args.pe_type,
            max_seq_len=args.block_size,
            theta=args.theta,
            device=device
        )
        optimizer = torch.optim.Adam(decoderLMmodel.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        global_iter = 0
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= args.max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = decoderLMmodel(xb)
            loss = criterion(logits.view(-1, logits.shape[-1]), yb.view(-1))
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/iter": global_iter + 1,
                })
            if (global_iter + 1) % args.eval_interval == 0:
                train_perplexity = compute_perplexity(decoderLMmodel, train_LM_loader, criterion)
                perplexity_hbush = compute_perplexity(decoderLMmodel, test_LM_loader_hbush, criterion)
                perplexity_obama = compute_perplexity(decoderLMmodel, test_LM_loader_obama, criterion)
                perplexity_wbush = compute_perplexity(decoderLMmodel, test_LM_loader_wbush, criterion)
                if args.wandb:
                    wandb.log({
                        "train/perplexity": train_perplexity,
                        "val/perplexity_hbush": perplexity_hbush,
                        "val/perplexity_obama": perplexity_obama,
                        "val/perplexity_wbush": perplexity_wbush,
                    })
                print(f"Iteration {global_iter + 1}: Train perplexity: {train_perplexity:.2f}, Perplexity for HBush: {perplexity_hbush:.2f}, Obama: {perplexity_obama:.2f}, WBush: {perplexity_wbush:.2f}")
            global_iter += 1
        if args.save_lm:
            torch.save(decoderLMmodel.state_dict(), args.save_lm)
            print(f"Saved language model to {args.save_lm}")

        wandb.finish()



if __name__ == "__main__":
    main()
