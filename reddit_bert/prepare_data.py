import random
import string

import pandas as pd


target_col = 'controversiality'


def generate_random_string(length=12):
    random.state(42)

    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


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
    raw_data = pd.read_parquet('s3://fredy-data/reddit/reddit_comment_sample.parquet/')

    sampled_data = downsample_data(raw_data)

    sampled_data['comment_id'] = [generate_random_string() for _ in range(len(sampled_data))]

    return sampled_data
