import logging

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, AdamW


logger = logging.getLogger(__name__)

model_str = 'distilbert-base-uncased'
batch_size = 32


class RedditDataset(Dataset):
    def __init__(self, texts, encoded_subreddits, tokenizer, labels=None, max_length=128):
        self.texts = texts
        self.encoded_subreddits = encoded_subreddits
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded_subreddit = self.encoded_subreddits[idx]

        # Tokenize text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        # Construct the item with necessary inputs
        item = {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'subreddit_encoded': torch.tensor(encoded_subreddit, dtype=torch.long),
        }

        # Include labels if not provided
        if self.labels is not None:
            label = self.labels[idx]
            item['labels'] = torch.tensor(label, dtype=torch.long)

        return item


class DistilBERTWithSubreddit(nn.Module):
    def __init__(self, n_subreddits, n_classes):
        super(DistilBERTWithSubreddit, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_str)
        # Embedding size for subreddit
        self.subreddit_embedding = nn.Embedding(n_subreddits, 32)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 32, n_classes)

    def forward(self, input_ids, attention_mask, subreddit):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Take the [CLS] token output
        bert_output = bert_output.last_hidden_state[:, 0, :]

        subreddit_embedding = self.subreddit_embedding(subreddit)
        # Concatenate the outputs
        combined_output = torch.cat((bert_output, subreddit_embedding), dim=1)

        logits = self.classifier(combined_output)
        return logits


def train_subreddit_encoder(df, provided_encoder=None):
    if provided_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(df['subreddit'])
    else:
        label_encoder = provided_encoder

    df['subreddit_encoded'] = label_encoder.transform(df['subreddit'])

    return df, label_encoder


def convert_to_dataset(df, tokenizer, target_col=None):
    if target_col:
        reddit_dataset = RedditDataset(
            df['body'].values,
            df['subreddit_encoded'].values,
            tokenizer,
            df[target_col].values,
        )
    else:
        reddit_dataset = RedditDataset(
            df['body'].values,
            df['subreddit_encoded'].values,
            tokenizer,
        )

    return DataLoader(reddit_dataset, batch_size=batch_size, shuffle=True)


def model_train(device, train_loader, n_subreddits):
    epochs = 3
    model = DistilBERTWithSubreddit(n_subreddits, n_classes=2)
    model.to(device)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        epoch_count = epoch + 1
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            subreddits = batch['subreddit_encoded'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, subreddits)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_value = total_loss / len(train_loader)
        logger.info('Epoch %d/%d, loss: %.4f', epoch_count, epochs, loss_value)

    return model


def model_predict(device, model, data_loader, evaluate=False):
    # Ensure model is in evaluation mode
    model.eval()

    all_labels = []
    all_probabilities = []
    all_predictions = []
    all_subreddits = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            subreddits = batch['subreddit_encoded'].to(device)
            if evaluate:
                labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, subreddits)
            # Probabilities of positive class
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=-1)

            # Append predictions and true labels
            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_subreddits.extend(subreddits.cpu().numpy())
            if evaluate:
                all_labels.extend(labels.cpu().numpy())

    if evaluate:
        # Convert lists to numpy arrays
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Calculate precision, recall, and thresholds
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probabilities)

        # Calculate AUCPR
        aucpr = auc(recall, precision)
        logger.info('Validation AUCPR: %.4f', aucpr)

    return predictions, subreddits
