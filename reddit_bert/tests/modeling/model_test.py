from unittest import TestCase

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from reddit_bert.modeling.model import DistilBERTWithSubreddit, RedditDataset, convert_to_dataset, \
    model_predict, model_train, train_subreddit_encoder


model_str = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_str)

data_df = pd.DataFrame({
    'body': ['This is a test comment.', 'Another test comment.'],
    'subreddit_encoded': [0, 1],
    'target': [0, 1],
})


class TestRedditDataset(TestCase):

    def setUp(self):
        self.tokenizer = tokenizer
        self.texts = ['This is a test comment.', 'Another test comment.']
        self.encoded_subreddit = [0, 1]
        self.labels = [1, 0]
        self.dataset = RedditDataset(
            self.texts,
            self.encoded_subreddit,
            self.tokenizer,
            self.labels,
        )

    def test_length(self):
        self.assertEqual(len(self.dataset), len(self.texts))

    def test_item(self):
        item = self.dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('subreddit_encoded', item)
        self.assertIn('labels', item)


class TestDistilBERTWithSubreddit(TestCase):

    def setUp(self):
        self.model = DistilBERTWithSubreddit(n_subreddits=10, n_classes=2)

    def test_forward_pass(self):
        # Example tensor
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones((2, 128), dtype=torch.long)
        subreddits = torch.tensor([0, 1])

        logits = self.model(input_ids, attention_mask, subreddits)
        # Output size should match batch size and class count
        self.assertEqual(logits.shape,(2, 2))


class TestTrainSubredditEncoder(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'subreddit': ['subreddit1', 'subreddit2', 'subreddit1', 'subreddit3'],
        })

    def test_train_subreddit_encoder_no_encoder(self):
        # Test when no provided_encoder is passed
        df, encoder = train_subreddit_encoder(self.df)

        # Verify
        self.assertTrue('subreddit_encoded' in df.columns)
        self.assertEqual(len(encoder.classes_), 3)
        self.assertTrue(all(df['subreddit_encoded'] >= 0))

    def test_train_subreddit_encoder_with_provided_encoder(self):
        # Test when provided_encoder is provided
        existing_encoder = LabelEncoder()
        existing_encoder.fit(['subreddit1', 'subreddit2', 'subreddit3'])

        df, encoder = train_subreddit_encoder(self.df, existing_encoder)

        # Verify
        self.assertTrue('subreddit_encoded' in df.columns)
        self.assertIs(encoder, existing_encoder)
        self.assertTrue(all(df['subreddit_encoded']) >= 0)


class TestConvertToDataset(TestCase):

    def setUp(self):
        self.tokenizer = tokenizer
        self.df = data_df

    def test_convert_with_target(self):
        data_loader = convert_to_dataset(
            self.df,
            self.tokenizer,
            32,
            target_col='target',
        )
        self.assertIsInstance(data_loader, DataLoader)
        batch = next(iter(data_loader))
        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)
        self.assertIn('subreddit_encoded', batch)
        self.assertIn('labels', batch)


class TestModelTrainingAndPrediction(TestCase):

    base = 'reddit_bert.modeling.model'

    def setUp(self):
        self.device = torch.device('cpu')
        self.tokenizer = tokenizer
        self.df = data_df
        self.data_loader = convert_to_dataset(
            self.df,
            self.tokenizer,
            32,
            target_col='target',
        )

    def test_train_model(self):
        model = model_train(self.device, self.data_loader, n_subreddits=2)
        self.assertIsInstance(model, DistilBERTWithSubreddit)

    def test_model_predict(self):
        model = DistilBERTWithSubreddit(n_subreddits=2, n_classes=2).to(self.device)
        predictions, subreddits = model_predict(self.device, model, self.data_loader)
        self.assertEqual(len(predictions), len(self.df))
        self.assertEqual(len(subreddits), len(self.df))
