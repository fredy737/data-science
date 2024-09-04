import logging

from reddit_bert.conf import settings
from reddit_bert.modeling.split_data import extract_transform

logger = logging.getLogger(__name__)


def main():
    logger.info('Running split_data stage')
    train_data, test_data = extract_transform()

    training_data_path = settings.DATA['training_data']
    test_data_path = settings.DATA['test_data']

    logger.info('Writing training data to s3.')
    train_data.to_parquet(training_data_path)

    logger.info('Writing tests data to s3.')
    test_data.to_parquet(test_data_path)


if __name__ == '__main__':
    main()
