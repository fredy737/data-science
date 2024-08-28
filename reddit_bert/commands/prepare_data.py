import logging

from reddit_bert.prepare_data import extract_transform

logger = logging.getLogger(__name__)


def main():
    logger.info('Running prepare_data stage')
    processed_data = extract_transform()

    logger.info('Writing downsampled data to s3.')
    processed_data.to_parquet('s3://fredy-data/reddit/downsampled_reddit_comments.parquet/')


if __name__ == '__main__':
    main()
