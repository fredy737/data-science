import boto3
import logging

import pandas as pd

import torch
from transformers import DistilBertModel, DistilBertTokenizer, AdamW

from reddit_bert.modeling.model import RedditDataset, DistilBERTWithSubreddit, \
    convert_to_dataset, model_train, model_predict, train_subreddit_encoder


logger = logging.getLogger(__name__)


def save_model(model):
    from reddit_bert.conf import settings
    # s3 parameters
    model_path = settings.S3['model_local_path']
    s3_bucket = settings.S3['s3_bucket']

    # Save the model state_dict (weights and biases)
    torch.save(model.state_dict(), model_path)

    s3_object_name = f'reddit/{model_path}'

    # Initialize the s3 client
    s3_client = boto3.client('s3')

    # Upload the model
    s3_client.upload_file(model_path, s3_bucket, s3_object_name)

    logger.info('Model uploaded successfully to s3://%s/%s', s3_bucket, s3_object_name)


def train():
    from reddit_bert.conf import settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = settings.MODEL['batch_size']
    target_col = settings.MODEL['target_col']

    tokenizer_model_str = settings.MODEL['tokenizer_model_str']
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_model_str)

    training_data_path = settings.DATA['training_data']
    test_data_path = settings.DATA['test_data']

    train_df = pd.read_parquet(training_data_path)
    test_df = pd.read_parquet(test_data_path)

    train_dataset_encoded, subreddit_encoder = train_subreddit_encoder(train_df)
    test_dataset_encoded, _ = train_subreddit_encoder(test_df, provided_encoder=subreddit_encoder)

    train_loader = convert_to_dataset(
        train_dataset_encoded,
        tokenizer,
        batch_size,
        target_col=target_col,
    )
    test_loader = convert_to_dataset(
        test_dataset_encoded,
        tokenizer,
        batch_size,
        target_col=target_col,
    )

    n_subreddits = len(subreddit_encoder.classes_)

    model = model_train(device, train_loader, n_subreddits)
    _, _ = model_predict(device, model, test_loader, evaluate=True)

    save_model(model)
