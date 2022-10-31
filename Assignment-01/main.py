import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets
from model import BiLSTMPOSTagger

import numpy as np

import time
import random
import os, sys
import argparse
import json

import warnings

warnings.filterwarnings("ignore")

# set command line options
parser = argparse.ArgumentParser(description="main.py")
parser.add_argument(
    "--mode",
    type=str,
    choices=["train", "eval"],
    default="train",
    help="Run mode",
)
parser.add_argument(
    "--lang",
    type=str,
    choices=["en", "cs", "es", "ar", "hy", "lt", "af", "ta"],
    default="en",
    help="Language code",
)
parser.add_argument(
    "--model-name",
    type=str,
    default=None,
    help="name of the saved model",
)
args = parser.parse_args()

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

if args.model_name is None:
    args.model_name = "{}-model".format(args.lang)

# set a fixed seed for reproducibility
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

params = json.load(open("config.json"))


def main():
    print("Running main.py in {} mode with lang: {}".format(args.mode, args.lang))
    # define fields for your data, we have two: the text itself and the POS tag
    TEXT = data.NestedField(lower=True)
    UD_TAGS = data.Field()

    fields = (("text", TEXT), ("udtags", UD_TAGS))

    # load the data from the specific path
    train_data, valid_data, test_data = datasets.UDPOS.splits(
        fields=fields,
        path=os.path.join("data", args.lang),
        train="{}-ud-train.conll".format(args.lang),
        validation="{}-ud-dev.conll".format(args.lang),
        test="{}-ud-test.conll".format(args.lang),
    )  # modify this to include our own dataset
    # building the vocabulary for both text and the labels
    MIN_FREQ = 2

    TEXT.build_vocab(
        train_data, min_freq=MIN_FREQ,
        # vectors="glove.6B.300d",
        # unk_init=torch.Tensor.normal_
    )
    UD_TAGS.build_vocab(train_data)

    if args.mode == "train":
        print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
        print(f"Unique tokens in UD_TAG vocabulary: {len(UD_TAGS.vocab)}")
        print()
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of validation examples: {len(valid_data)}")

        print(f"Number of tokens in the training set: {sum(TEXT.vocab.freqs.values())}")

    print(f"Number of testing examples: {len(test_data)}")

    if args.mode == "train":
        print("Tag\t\tCount\t\tPercentage\n")
        for tag, count, percent in tag_percentage(UD_TAGS.vocab.freqs.most_common()):
            print(f"{tag}\t\t{count}\t\t{percent*100:4.1f}%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=params["batch_size"],
        device=device,
    )

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = BiLSTMPOSTagger(
        input_dim=len(TEXT.vocab),
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        output_dim=len(UD_TAGS.vocab),
        n_layers=params["n_layers"],
        bidirectional=params["bidirectional"],
        dropout=params["dropout"],
        pad_idx=PAD_IDX,
    )
    # pretrained_embeddings = TEXT.vocab.vectors
    # model.embedding.weight.data.copy_(pretrained_embeddings)
    # fixing the pretrained embeddings
    # model.embedding.weight.requires_grad = False

    if args.mode == "train":

        def init_weights(m):
            for name, param in m.named_parameters():
                # nn.init.normal_(param.data, mean=0, std=0.1)
                nn.init.xavier_uniform_(param.data, gain=1.0)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.apply(init_weights)
        print(f"The model has {count_parameters(model):,} trainable parameters")
        model.embedding.weight.data[PAD_IDX] = torch.zeros(params["embedding_dim"])
        optimizer = optim.Adam(model.parameters())

    TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
    TAG_UNK_IDX = UD_TAGS.vocab.unk_index
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)

    if args.mode == "train":
        N_EPOCHS = 15
        best_valid_loss = float("inf")
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss, train_acc = train(
                model,
                train_iterator,
                optimizer,
                criterion,
                TAG_PAD_IDX,
                TAG_UNK_IDX,
            )
            valid_loss, valid_acc = evaluate(
                model, valid_iterator, criterion, TAG_PAD_IDX, TAG_UNK_IDX
            )
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    model.state_dict(), "saved_models/{}.pt".format(args.model_name)
                )

            print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

    try:
        model.load_state_dict(torch.load("saved_models/{}.pt".format(args.model_name)))
    except Exception as e:
        print(
            "Model file `{}` doesn't exist. You need to train the model by running this code in train mode. Run python main.py --help for more instructions".format(
                "saved_models/{}.pt".format(args.model_name)
            )
        )
        return

    test_loss, test_acc = evaluate(
        model, test_iterator, criterion, TAG_PAD_IDX, TAG_UNK_IDX
    )
    print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%")


def tag_percentage(tag_counts):
    total_count = sum([count for tag, count in tag_counts])
    tag_counts_percentages = [
        (tag, count, count / total_count) for tag, count in tag_counts
    ]
    return tag_counts_percentages


def categorical_accuracy(preds, y, tag_pad_idx, tag_unk_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True
    )  # get the index of the max probability
    non_pad_elements = torch.nonzero((y != tag_pad_idx) & (y != tag_unk_idx))
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    # print(correct.float().sum(), y[non_pad_elements].shape[0])
    return correct.float().sum(), y[non_pad_elements].shape[0]



def train(model, iterator, optimizer, criterion, tag_pad_idx, tag_unk_idx):

    epoch_loss = 0
    epoch_correct = 0
    epoch_n_label = 0

    model.train()

    for batch in iterator:

        text = batch.text
        tags = batch.udtags

        optimizer.zero_grad()

        # text = [sent len, batch size]

        predictions = model(text)
        loss = -model.crf(predictions, tags)

        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]

        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)

        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]

        # loss = criterion(predictions, tags)

        correct, n_labels = categorical_accuracy(
            predictions, tags, tag_pad_idx, tag_unk_idx
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_correct += correct.item()
        epoch_n_label += n_labels

    return epoch_loss / len(iterator), epoch_correct / epoch_n_label


def evaluate(model, iterator, criterion, tag_pad_idx, tag_unk_idx):

    epoch_loss = 0
    epoch_correct = 0
    epoch_n_label = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            text = batch.text
            tags = batch.udtags

            predictions = model(text)
            loss = -model.crf(predictions, tags)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            # loss = criterion(predictions, tags)

            correct, n_labels = categorical_accuracy(
                predictions, tags, tag_pad_idx, tag_unk_idx
            )

            epoch_loss += loss.item()
            epoch_correct += correct.item()
            epoch_n_label += n_labels

    return epoch_loss / len(iterator), epoch_correct / epoch_n_label


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    main()
