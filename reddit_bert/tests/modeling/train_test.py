from unittest import TestCase, mock
from reddit_bert.modeling.train import save_model, train


class TestModelFunctions(TestCase):

    model_base = 'reddit_bert.modeling.model'
    train_base = 'reddit_bert.modeling.train'

    @mock.patch('torch.save')
    @mock.patch('boto3.client')
    def test_save_model(self, mock_boto_client, mock_torch_save):
        # Setup
        mock_s3_client = mock.MagicMock()
        mock_boto_client.return_value = mock_s3_client

        model = mock.MagicMock()
        model.state_dict.return_value = {}

        # Run
        save_model(model)

        # Verify
        mock_torch_save.assert_called_once_with(
            model.state_dict(),
            'reddit_model.pt',
        )
        mock_s3_client.upload_file.assert_called_once_with(
            'reddit_model.pt',
            'fredy-data',
            'reddit/reddit_model.pt',
        )
