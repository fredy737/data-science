import boto3
import logging

import pandas as pd

import torch
from transformers import DistilBertModel, DistilBertTokenizer, AdamW

from reddit_bert.modeling.model import RedditDataset, DistilBERTWithSubreddit, \
    convert_to_dataset, model_train, model_predict, train_subreddit_encoder


logger = logging.getLogger(__name__)

target_col = 'controversiality'
model_str = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(model_str)


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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_parquet('s3://fredy-data/reddit/reddit_comments_training_data.parquet/')
    test_df = pd.read_parquet('s3://fredy-data/reddit/reddit_comments_test_data.parquet/')

    train_dataset_encoded, subreddit_encoder = train_subreddit_encoder(train_df)
    test_dataset_encoded, _ = train_subreddit_encoder(test_df, provided_encoder=subreddit_encoder)

    train_loader = convert_to_dataset(train_dataset_encoded, tokenizer, target_col=target_col)
    test_loader = convert_to_dataset(test_dataset_encoded, tokenizer, target_col=target_col)

    n_subreddits = len(subreddit_encoder.classes_)

    model = model_train(device, train_loader, n_subreddits)
    _, _ = model_predict(device, model, test_loader, evaluate=True)

    save_model(model)
