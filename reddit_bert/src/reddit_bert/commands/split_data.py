import logging

from reddit_bert.modeling.split_data import extract_transform

logger = logging.getLogger(__name__)


def main():
    logger.info('Running split_data stage')
    train_data, test_data = extract_transform()

    logger.info('Writing training data to s3.')
    train_data.to_parquet('s3://fredy-data/reddit/reddit_comments_training_data.parquet/')

    logger.info('Writing tests data to s3.')
    test_data.to_parquet('s3://fredy-data/reddit/reddit_comments_test_data.parquet/')


if __name__ == '__main__':
    main()
