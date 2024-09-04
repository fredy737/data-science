import boto3

import pandas as pd

import torch
from transformers import DistilBertModel, DistilBertTokenizer, AdamW

from reddit_bert.modeling.model import RedditDataset, DistilBERTWithSubreddit, \
    convert_to_dataset, model_predict, train_subreddit_encoder


def load_model(n_subreddits):
    from reddit_bert.conf import settings
    # s3 parameters
    model_path = settings.S3['model_local_path']
    s3_bucket = settings.S3['s3_bucket']

    s3_object_name = f'reddit/{model_path}'
    download_path = model_path

    # Initialize the s3 client
    s3_client = boto3.client('s3')

    # Download the file from S3
    s3_client.download_file(s3_bucket, s3_object_name, download_path)

    # Load the model state_ict from the downloaded file
    model = DistilBERTWithSubreddit(n_subreddits, n_classes=2)
    model.load_state_dict(torch.load(download_path))
    return model


def predict():
    from reddit_bert.conf import settings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_str = settings.MODEL['tokenizer_model_str']
    tokenizer = DistilBertTokenizer.from_pretrained(model_str)

    data_path = settings.DATA['raw_data']
    data = pd.read_parquet(data_path)

    comment_ids = data['comment_id'].tolist()
    comments = data['body'].tolist()

    dataset_encoded, subreddit_encoder = train_subreddit_encoder(data)

    data_loader = convert_to_dataset(dataset_encoded, tokenizer)

    n_subreddits = len(subreddit_encoder.classes_)
    model = load_model(n_subreddits)

    predictions, subreddits = model_predict(device, model, data_loader)

    result_df = pd.DataFrame({
        'comment_id': comment_ids,
        'body': comments,
        'subreddit_encoded': subreddits,
        'controversiality_prediction': predictions,
    })

    result_df['subreddit'] = subreddit_encoder.inverse_transform(result_df['subreddit_encoded'])

    return result_df
