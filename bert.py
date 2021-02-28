import argparse
import datetime
import os
import random
import time
import torch
import transformers

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from typing import Optional

from twitter_bot.io import write_json


# Typing
DataFrame = pd.DataFrame
Tokenizer = transformers.PreTrainedTokenizer
TorchDataset = torch.utils.data.dataset.Dataset
TorchSampler = torch.utils.data.sampler.Sampler


def prepare_dataset(data: DataFrame,
                    tokenizer: Tokenizer,
                    seq_len: int) -> TorchDataset:
    input_ids = []
    attention_masks = []

    for tweet in tqdm(data.Text):
        encoded_dict = tokenizer.encode_plus(tweet,
                                             add_special_tokens=True,
                                             max_length=seq_len,
                                             padding='max_length',
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(data.Label.apply(lambda x: 0 if x.lower() == 'human' else 1).to_numpy().astype(int))

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset


def prepare_data_loader(dataset: TorchDataset,
                        batch_size: int,
                        sampler: Optional[TorchSampler] = None) -> DataLoader:
    if sampler is None:
        sampler = SequentialSampler
    return DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)


def flat_accuracy(predictions, labels) -> float:
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed: float) -> str:
    return str(datetime.timedelta(seconds=int(round(elapsed))))


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
    parser.add_argument('--lr', type=float, default=0.00002,
                        help='AdamW learning rate parameter')
    parser.add_argument('--eps', type=float, default=0.00000001,
                        help='AdamW epsilon parameter')
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

    print('Load BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    print('Tokenize datasets...')
    train_dataset = prepare_dataset(pd.read_csv(os.path.join(args.data, './training.csv')), tokenizer, args.seq_len)
    val_dataset = prepare_dataset(pd.read_csv(os.path.join(args.data, './validation.csv')), tokenizer, args.seq_len)
    test_dataset = prepare_dataset(pd.read_csv(os.path.join(args.data, './testing.csv')), tokenizer, args.seq_len)

    train_dataloader = prepare_data_loader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler)
    val_dataloader = prepare_data_loader(val_dataset, batch_size=args.batch_size)
    test_dataloader = prepare_data_loader(val_dataset, batch_size=args.batch_size)

    print('Load BERT model for sequence classification...')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=2,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(device)

    # AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.eps)

    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * args.epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # Training

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
        print('Training...')

        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode.
        model.train()

        for step, batch in enumerate(train_dataloader):

            # Progress update every <log-interval> batches.
            if step % args.log_interval == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack ther training batch and send them to GPU
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear Grads
            model.zero_grad()

            # Perform a forward pass
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss = total_train_loss + outputs.loss.item()

            # Perform a backward pass
            outputs.loss.backward()

            # Clip the norm of the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # Update parameters
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss
        avg_train_loss = total_train_loss / len(train_dataloader)

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

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # No need to accumulate the grads
            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += outputs.loss.item()

            # Move logits and labels to CPU
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final stats
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)
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
    model.save_pretrained(os.path.join(args.save_dir, 'model'))

    # Testing
    # Tracking variables, append the labels and prediction using these two lists
    predictions, true_labels = [], []

    # Prediction on test set
    model.eval()

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # No need to accumulate the grads
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

        # Move logits and labels to CPU
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)


    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Compute ROC AUC score
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
