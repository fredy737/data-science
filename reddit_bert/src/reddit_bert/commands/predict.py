import logging

from reddit_bert.modeling.predict import predict

logger = logging.getLogger(__name__)


def main():
    logger.info('Running predict stage')
    predicted_df = predict()
    predicted_df.to_parquet(
        's3://fredy-data/reddit/reddit_comments_with_predictions.parquet/',
        partition_cols=['subreddit'],
    )


if __name__ == '__main__':
    main()
