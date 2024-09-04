import logging

from reddit_bert.conf import settings
from reddit_bert.modeling.predict import predict

logger = logging.getLogger(__name__)


def main():
    logger.info('Running predict stage')
    predicted_df = predict()

    predict_output_path = settings.DATA['predict_output']
    predicted_df.to_parquet(predict_output_path)


if __name__ == '__main__':
    main()
