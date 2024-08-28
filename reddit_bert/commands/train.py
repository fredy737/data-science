import logging

from reddit_bert.train import train_model

logger = logging.getLogger(__name__)


def main():
    logger.info('Running train stage')
    train_model()


if __name__ == '__main__':
    main()
