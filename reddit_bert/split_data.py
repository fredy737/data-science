import logging

import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

target_col = 'controversiality'
stratify_col = 'subreddit'


def extract_transform():
    data = pd.read_parquet('s3://fredy-data/reddit/downsampled_reddit_comments.parquet/')

    data['stratify_key'] = data[target_col].astype(str) + '_' + data[stratify_col].astype(str)

    # Perform stratified train-test split
    train_df, test_df = train_test_split(
        data,
        test_size=0.25,
        stratify=data['stratify_key'],
        random_state=42,
    )

    train_df = train_df.drop(columns=['stratify_key'])
    test_df = test_df.drop(columns=['stratify_key'])

    return train_df, test_df

    logger.info('Writing split data to s3.')
    train_df.to_parquet('s3://fredy-data/reddit/reddit_comments_training_data.parquet/')
    test_df.to_parquet('s3://fredy-data/reddit/reddit_comments_test_data.parquet/')
