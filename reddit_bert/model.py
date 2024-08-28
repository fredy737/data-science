import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import DistilBertModel

model_str = 'distilbert-base-uncased'


class RedditDataset(Dataset):
    def __init__(self, texts, subreddits, labels, tokenizer, max_length=128):
        self.texts = texts
        self.subreddits = subreddits
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        subreddit = self.subreddits[idx]

        # Tokenize text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'subreddit': torch.tensor(subreddit, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
        }


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
