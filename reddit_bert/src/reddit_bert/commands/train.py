import logging

from reddit_bert.modeling.train import train

logger = logging.getLogger(__name__)


def main():
    logger.info('Running train stage')
    train()


if __name__ == '__main__':
    main()
