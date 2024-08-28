import boto3
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer, AdamW

from reddit_bert.model import RedditDataset, DistilBERTWithSubreddit
from reddit_bert import utils as U


logger = logging.getLogger(__name__)

target_col = 'controversiality'
stratify_col = 'subreddit'
model_str = 'distilbert-base-uncased'
batch_size = 32

tokenizer = DistilBertTokenizer.from_pretrained(model_str)


def train_subreddit_encoder(df):
    label_encoder = LabelEncoder()
    df['subreddit'] = label_encoder.fit_transform(df['subreddit'])

    return df, label_encoder


def convert_to_dataset(df):
    return RedditDataset(
        df['body'].values,
        df[f'{stratify_col}_encoded'].values,
        df[target_col].values,
        tokenizer,
    )


def get_batch_values(batch, device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    subreddits = batch['subreddit'].to(device)

    return input_ids, attention_mask, labels, subreddits


def save_model(model):
    # Define the model save patch
    local_model_path = 'reddit_model.pt'

    # Save the model state_dict (weights and biases)
    torch.save(model.state_dict(), local_model_path)

    s3_bucket = 'fredy-data'
    s3_object_name = 'reddit/reddit_model.pt'

    # Initialize the s3 client
    s3_client = boto3.client('s3')

    # Upload the model
    s3_client.upload_file(local_model_path, s3_bucket, s3_object_name)

    logger.info('Model uploaded successfully to s3://%s/%s', s3_bucket, s3_object_name)


def train_model():
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_parquet('s3://fredy-data/reddit/reddit_comments_training_data.parquet/')
    test_df = pd.read_parquet('s3://fredy-data/reddit/reddit_comments_test_data.parquet/')

    train_dataset_encoded, subreddit_encoder = train_subreddit_encoder(train_df)
    test_dataset_encoded = subreddit_encoder.transform(test_df)

    train_loader = U.convert_to_dataset(train_dataset_encoded, target_col, tokenizer)
    test_loader = U.convert_to_dataset(test_dataset_encoded, target_col, tokenizer)

    n_subreddits = len(subreddit_encoder.classes_)
    model = DistilBERTWithSubreddit(n_subreddits, n_classes=2)
    model.to(device)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        epoch_count = epoch + 1
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels, subreddits = get_batch_values(batch, device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, subreddits)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss ++ loss.item()

        loss_value = total_loss / train_loader
        logger.info('Epoch %d/%d, loss: %d', epoch_count, epochs, loss_value)

    save_model(model)

    # Evaluation loop for AUCPR
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels, subreddits = get_batch_values(batch, device)

            logits = model(input_ids, attention_mask, subreddits)
            # Probabilities of positive class
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            # Append predictions and true labels
            all_predictions.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(all_labels, all_predictions)

    # Calculate AUCPR
    aucpr = auc(recall, precision)
    logger.info('Validation AUCPR: %d', aucpr)
