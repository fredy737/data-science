from unittest import mock, TestCase

import torch

from reddit_bert.modeling.predict import load_model

class TestLoadModel(TestCase):

    model_base = 'reddit_bert.modeling.model'
    predict_base = 'reddit_bert.modeling.predict'

    @mock.patch(f'{predict_base}.boto3.client')
    @mock.patch(f'{predict_base}.DistilBERTWithSubreddit')
    @mock.patch(f'{predict_base}.torch.load')
    @mock.patch('reddit_bert.conf.settings')
    def test_load_model(self, mock_settings, mock_torch_load, mock_distil_bert, mock_boto3_client):
        # Setup the mock objects
        mock_s3_client = mock.MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        mock_settings.S3 = {
            's3_bucket': 'bucket',
            'model_local_path': 'model.pt',
        }

        # Mock the model and its state_dict
        mock_model = mock_distil_bert.return_value
        correct_state_dict = {
            'subreddit_embedding.weight': torch.randn(10, 32),  # Adjust dimensions to match the model
            'classifier.weight': torch.randn(2, 32),
            'classifier.bias': torch.randn(2)
        }
        mock_torch_load.return_value = correct_state_dict

        # Mock load_state_dict
        mock_model.load_state_dict.return_value = None  # No side effect needed here, just ensure it's called

        # Run the function
        _ = load_model(10)

        # Assertions
        mock_boto3_client.assert_called_once_with('s3')
        mock_s3_client.download_file.assert_called_once_with('bucket', 'reddit/model.pt', 'model.pt')
        mock_distil_bert.assert_called_once_with(10, n_classes=2)  # Check model initialization
        mock_model.load_state_dict.assert_called_once_with(correct_state_dict)

