from unittest import mock, TestCase

import pandas as pd
from reddit_bert.modeling import split_data


class TestSplitData(TestCase):

    base = 'reddit_bert.modeling.split_data'

    def setUp(self):
        self.mock_df = pd.DataFrame([
            *[
                {
                    'comment_id': f'c1_{i}',
                    'body': f'comment1_{i}',
                    'subreddit': 's1',
                    'controversiality': 0,
                    'score': 10,
                }
                for i in range(30)
            ],
            *[
                {
                    'comment_id': f'c2_{i}',
                    'body': f'comment2_{i}',
                    'subreddit': 's1',
                    'controversiality': 1,
                    'score': 0,
                }
                for i in range(2)
            ],
            *[
                {
                    'comment_id': f'c3_{i}',
                    'body': f'comment3_{i}',
                    'subreddit': 's2',
                    'controversiality': 0,
                    'score': 10,
                }
                for i in range(30)
            ],
            *[
                {
                    'comment_id': f'c4_{i}',
                    'body': f'comment4_{i}',
                    'subreddit': 's2',
                    'controversiality': 1,
                    'score': 0,
                }
                for i in range(2)
            ],
        ])

    def test_downsample_data(self):
        # Call the function under tests
        result_df = split_data.downsample_data(self.mock_df)

        expected_df = pd.DataFrame([
            *[
                {
                    'comment_id': f'c1_{i}',
                    'body': f'comment1_{i}',
                    'subreddit': 's1',
                    'controversiality': 0,
                    'score': 10,
                }
                for i in [8, 9, 15, 17, 23, 27]
            ],
            *[
                {
                    'comment_id': f'c2_{i}',
                    'body': f'comment2_{i}',
                    'subreddit': 's1',
                    'controversiality': 1,
                    'score': 0,
                }
                for i in range(2)
            ],
            *[
                {
                    'comment_id': f'c3_{i}',
                    'body': f'comment3_{i}',
                    'subreddit': 's2',
                    'controversiality': 0,
                    'score': 10,
                }
                for i in [8, 9, 15, 17, 23, 27]
            ],
            *[
                {
                    'comment_id': f'c4_{i}',
                    'body': f'comment4_{i}',
                    'subreddit': 's2',
                    'controversiality': 1,
                    'score': 0,
                }
                for i in range(2)
            ],
        ])

        # Assert the result is the expected mock dataframe
        pd.testing.assert_frame_equal(
            result_df.sort_values(by='comment_id').reset_index(drop=True),
            expected_df.sort_values(by='comment_id').reset_index(drop=True),
        )

    @mock.patch(f'{base}.pd.read_parquet')
    @mock.patch('reddit_bert.conf.settings')
    def test_extract_transform(self, mock_settings, mock_read_parquet):
        mock_settings.DATA = {
            'raw_data': 'data_path',
        }
        # Configure the mock to return the mock DataFrame
        mock_read_parquet.return_value = self.mock_df

        # Call the function under tests
        train_df, test_df = split_data.extract_transform()

        # Assert the mock was called once with the expected filepath
        expected_path = 'data_path'
        mock_read_parquet.assert_called_once_with(expected_path)

        expected_train_df = pd.DataFrame([
            *[
                {
                    'comment_id': f'c1_{i}',
                    'body': f'comment1_{i}',
                    'subreddit': 's1',
                    'controversiality': 0,
                    'score': 10,
                }
                for i in [9, 15, 17, 27]
            ],
            {
                'comment_id': 'c2_0',
                'body': 'comment2_0',
                'subreddit': 's1',
                'controversiality': 1,
                'score': 0,
            },
            *[
                {
                    'comment_id': f'c3_{i}',
                    'body': f'comment3_{i}',
                    'subreddit': 's2',
                    'controversiality': 0,
                    'score': 10,
                }
                for i in [8, 9, 15, 17, 27]
            ],
            *[
                {
                    'comment_id': f'c4_{i}',
                    'body': f'comment4_{i}',
                    'subreddit': 's2',
                    'controversiality': 1,
                    'score': 0,
                }
                for i in range(2)
            ],
        ])

        # Assert the result is the expected mock dataframe
        pd.testing.assert_frame_equal(
            train_df.sort_values(by='comment_id').reset_index(drop=True),
            expected_train_df.sort_values(by='comment_id').reset_index(drop=True),
        )

        expected_test_df = pd.DataFrame([
            *[
                {
                    'comment_id': f'c1_{i}',
                    'body': f'comment1_{i}',
                    'subreddit': 's1',
                    'controversiality': 0,
                    'score': 10,
                }
                for i in [8, 23]
            ],
            {
                'comment_id': 'c2_1',
                'body': 'comment2_1',
                'subreddit': 's1',
                'controversiality': 1,
                'score': 0,
            },
            {
                'comment_id': 'c3_23',
                'body': 'comment3_23',
                'subreddit': 's2',
                'controversiality': 0,
                'score': 10,
            },
        ])

        # Assert the result is the expected mock dataframe
        pd.testing.assert_frame_equal(
            test_df.sort_values(by='comment_id').reset_index(drop=True),
            expected_test_df.sort_values(by='comment_id').reset_index(drop=True),
        )
