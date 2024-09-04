from unittest import mock, TestCase
from reddit_bert.modeling.model import DistilBERTWithSubreddit
from reddit_bert.modeling.predict import load_model  # Import your load_model function

class TestLoadModel(TestCase):

    model_base = 'reddit_bert.modeling.model'
    predict_base = 'reddit_bert.modeling.predict'

    @mock.patch(f'{predict_base}.boto3.client')
    @mock.patch(f'{model_base}.DistilBertModel.from_pretrained')
    @mock.patch.object(DistilBERTWithSubreddit, '__init__', return_value=None)  # Patch only the __init__ method
    @mock.patch(f'{predict_base}.torch.load')
    def test_load_model(self, mock_torch_load, mock_init, mock_from_pretrained, mock_boto3_client):
        # Setup the mock objects
        mock_s3_client = mock.MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        # Mock the state dict loading
        mock_state_dict = {'some_key': 'some_value'}
        mock_torch_load.return_value = mock_state_dict

        # Mock the model to verify load_state_dict is called correctly
        mock_model = DistilBERTWithSubreddit(n_subreddits=10, n_classes=2)
        mock_model.state_dict = mock_state_dict
        with mock.patch.object(mock_model, 'load_state_dict') as mock_load_state_dict:
            # Run the function
            result = load_model(10)

            # Assertions
            mock_boto3_client.assert_called_once_with('s3')
            mock_s3_client.download_file.assert_called_once_with('fredy-data', 'reddit/reddit_model.pt', 'model.pth')
            mock_init.assert_called_once_with(10, n_classes=2)
            mock_torch_load.assert_called_once_with('model.pth')
            mock_load_state_dict.assert_called_once_with(mock_state_dict)

            # Verify that the model returned is the mocked model
            self.assertEqual(result, mock_model)
