from torch.utils.data import DataLoader
from reddit_bert.model import RedditDataset

batch_size = 32


def convert_to_dataset(df, target_col, tokenizer):
    reddit_dataset = RedditDataset(
        df['body'].values,
        df['subreddit_encoded'].values,
        df[target_col].values,
        tokenizer,
    )

    return DataLoader(reddit_dataset, batch_size=batch_size, shuffle=True)
