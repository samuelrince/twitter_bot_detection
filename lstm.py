import argparse
import datetime
import os
import random
import time
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
from typing import Optional

from twitter_bot.io import write_json
from twitter_bot.model import LSTMForSequenceClassification
from twitter_bot.tokenizer import Tokenizer


# Typing
DataFrame = pd.DataFrame
TorchDataset = torch.utils.data.dataset.Dataset
TorchSampler = torch.utils.data.sampler.Sampler


def prepare_dataset(data: DataFrame,
                    tokenizer: Tokenizer) -> TorchDataset:
    input_ids = []

    for tweet in tqdm(data.Text):
        input_ids.append(torch.tensor([tokenizer.tokenize(tweet)]))

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.tensor(data.Label.apply(lambda x: 0 if x.lower() == 'human' else 1).to_numpy().astype(int))

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, labels)

    return dataset


def prepare_data_loader(dataset: TorchDataset,
                        batch_size: int,
                        sampler: Optional[TorchSampler] = None) -> DataLoader:
    if sampler is None:
        sampler = SequentialSampler
    return DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size, drop_last=True)


def flat_accuracy(predictions, labels) -> float:
    # pred_flat = np.argmax(predictions, axis=0).flatten()
    pred_flat = np.where(predictions > 0.5, 1, 0)
    labels_flat = labels.flatten()
    return float(np.sum(pred_flat == labels_flat) / len(labels_flat))


def format_time(elapsed: float) -> str:
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bert fine-tuning')
    parser.add_argument('--data', type=str, default='./data',
                        help='location of the data corpus')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='sequence length')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--lr', type=float, default=0.00002,
                        help='Adam learning rate parameter')
    parser.add_argument('--eps', type=float, default=0.00000001,
                        help='Adam epsilon parameter')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clipping')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--save-dir', type=str, default='./models/model',
                        help='path to save the final model')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
    write_json(os.path.join(args.save_dir, 'run_params.json'), args.__dict__)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Load tokenizer...')
    tokenizer = Tokenizer(max_sequence_length=args.seq_len)
    train_df = pd.read_csv(os.path.join(args.data, './training.csv'))
    tokenizer.build_vocab(list(train_df.Text))

    print('Tokenize datasets...')
    train_dataset = prepare_dataset(train_df, tokenizer)
    val_dataset = prepare_dataset(pd.read_csv(os.path.join(args.data, './validation.csv')), tokenizer)
    test_dataset = prepare_dataset(pd.read_csv(os.path.join(args.data, './testing.csv')), tokenizer)

    train_dataloader = prepare_data_loader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler)
    val_dataloader = prepare_data_loader(val_dataset, batch_size=args.batch_size)
    test_dataloader = prepare_data_loader(val_dataset, batch_size=args.batch_size)

    print('Load LSTM model for sequence classification...')
    model = LSTMForSequenceClassification(n_tokens=tokenizer.vocab_size,
                                          n_inputs=args.emsize,
                                          n_hidden=args.nhid,
                                          n_layers=args.nlayers,
                                          dropout=args.dropout)
    model.to(device)

    # Training

    training_stats = []
    total_t0 = time.time()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * args.epochs

    for epoch_i in range(0, args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
        print('Training...')

        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode.
        model.train()

        # Generate hidden state
        hidden = model.init_hidden(args.batch_size)

        for step, batch in enumerate(train_dataloader):

            # Progress update every <log-interval> batches.
            if step % args.log_interval == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack ther training batch and send them to GPU
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            # Clear Grads
            model.zero_grad()

            # Repackage hidden
            hidden = repackage_hidden(hidden)

            # Perform a forward pass
            outputs, hidden = model(b_input_ids, hidden)

            loss = criterion(outputs, b_labels.float())
            total_train_loss += loss

            # Perform a backward pass
            loss.backward()

            # Clip the norm of the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # Update parameters
            optimizer.step()

        # Calculate the average loss
        avg_train_loss = float(total_train_loss / len(train_dataloader))

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # Validation

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode
        model.eval()

        # Init hidden state
        hidden = model.init_hidden(args.batch_size)

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            # No need to accumulate the grads
            with torch.no_grad():
                outputs, hidden = model(b_input_ids, hidden)

            # Accumulate the validation loss.
            loss = criterion(outputs, b_labels.float())
            total_eval_loss += loss

            # Move logits and labels to CPU
            outputs_ids = outputs.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy
            total_eval_accuracy += flat_accuracy(outputs_ids, label_ids)

        # Report the final stats
        avg_val_accuracy = float(total_eval_accuracy / len(val_dataloader))
        avg_val_loss = float(total_eval_loss / len(val_dataloader))
        validation_time = format_time(time.time() - t0)

        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Save stats
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # Saving model and stats
    write_json(os.path.join(args.save_dir,  'training_stats.json'), training_stats)
    torch.save(model, os.path.join(args.save_dir, 'model'))

    # Testing
    # Tracking variables, append the labels and prediction using these two lists
    predictions, true_labels = [], []

    # Prediction on test set
    model.eval()

    # Init hidden state
    hidden = model.init_hidden(args.batch_size)

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)

        # No need to accumulate the grads
        with torch.no_grad():
            outputs, hidden = model(b_input_ids, hidden)

        # Move logits and labels to CPU
        outputs_ids = outputs.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(outputs_ids)
        true_labels.append(label_ids)

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    flat_predictions = np.where(flat_predictions > 0.5, 1, 0)

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    acc = accuracy_score(flat_true_labels, flat_predictions)
    f1 = f1_score(flat_true_labels, flat_predictions)
    pre = precision_score(flat_true_labels, flat_predictions)
    rec = recall_score(flat_true_labels, flat_predictions)

    print('Accuracy score: %.3f' % acc)
    print('F1 score: %.3f' % f1)
    print('Precision score: %.3f' % pre)
    print('Recall score: %.3f' % rec)

    write_json(os.path.join(args.save_dir,  'testing_stats.json'), {
        'accuracy_score': acc,
        'f1_score': f1,
        'precision': pre,
        'recall': rec
    })
