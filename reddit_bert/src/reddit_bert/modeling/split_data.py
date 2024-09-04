import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

target_col = 'controversiality'
stratify_col = 'subreddit'


def downsample_data(df):
    noncontroversial_data = df[df[target_col] == 0]
    controversial_data = df[df[target_col] == 1]

    grouped_comments = noncontroversial_data.groupby('subreddit')

    sampled = grouped_comments.apply(
        lambda x: x.sample(frac=0.2, random_state=42)
    ).reset_index(drop=True)

    return pd.concat([
        sampled,
        controversial_data,
    ])


def extract_transform():
    from reddit_bert.conf import settings

    raw_data_path = settings.DATA['raw_data']
    data = pd.read_parquet(raw_data_path)

    sampled_data = downsample_data(data)

    sampled_data['stratify_key'] = \
        sampled_data[target_col].astype(str) + '_' + sampled_data[stratify_col].astype(str)

    # Perform stratified train-tests split
    train_df, test_df = train_test_split(
        sampled_data,
        test_size=0.25,
        stratify=sampled_data['stratify_key'],
        random_state=42,
    )

    train_df = train_df.drop(columns=['stratify_key'])
    test_df = test_df.drop(columns=['stratify_key'])

    return train_df, test_df
