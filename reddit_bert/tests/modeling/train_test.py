from unittest import TestCase, mock
from reddit_bert.modeling.train import save_model, train


class TestModelFunctions(TestCase):
    train_base = 'reddit_bert.modeling.train'

    @mock.patch('torch.save')
    @mock.patch('boto3.client')
    @mock.patch('reddit_bert.conf.settings')
    def test_save_model(self, mock_settings, mock_boto_client, mock_torch_save):
        # Setup
        mock_s3_client = mock.MagicMock()
        mock_boto_client.return_value = mock_s3_client

        mock_settings.S3 = {
            's3_bucket': 'bucket',
            'model_local_path': 'model.pt',
        }

        model = mock.MagicMock()
        model.state_dict.return_value = {}

        # Run
        save_model(model)

        # Verify
        mock_torch_save.assert_called_once_with(
            model.state_dict(),
            'model.pt',
        )
        mock_s3_client.upload_file.assert_called_once_with(
            'model.pt',
            'bucket',
            'reddit/model.pt',
        )
