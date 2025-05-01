# multitask_model.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", hidden_dim=768, num_classes_a=3, num_classes_b=2):
        """
        multi task model built on top of a shared transformer encoder

        task A -- classify sentences
        task B - sentiment analysis (chosen nlp item)

        args - 
            model_name (str): Hugging Face transformer to use as encoder
            hidden_dim (int): Size of transformer output (768 from last test as hidden size)
            num_classes_a (int): Number of classes for Task A
            num_classes_b (int): Number of classes for Task B
        """

        super(MultiTaskModel, self).__init__()
        # loaded tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load the model
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # pooling strategy like last time 
        self.pooling = "mean"

        # task a head
        self.classifier_a = nn.Linear(hidden_dim, num_classes_a)
        # task b head
        self.classifier_b = nn.Linear(hidden_dim, num_classes_b)

    def forward(self, sentences):
        """
        forward pass through encoder and both heads

        args - sentences (list of strings): Sentences to encode and classify

        returns - 
            logits_a (torch.Tensor): output scores for Task A
            logits_b (torch.Tensor): output scores for Task B
        """

        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # pass tokenizerd inputs through the encoder
        # returns all hidden states for each token
        # performs the forward pass
        outputs = self.encoder(**encoded_input)
        # gives us the last hidden state
        # grabs the token embeddings
        token_embeddings = outputs.last_hidden_state
        # grabs the atention masking bit -- 1 for real and 0 for padding
        attention_mask = encoded_input['attention_mask']

        # starting the mean pooling

        # expand the attention mask to match the token embeddings and shape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # sum the token embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # count of non padding tokens for each sentence so we can average
        sum_mask = input_mask_expanded.sum(dim=1)

        # divide summed embeddings by the sum mask to get the mean

        sentence_embeddings = sum_embeddings / sum_mask

        # heads for each task

        # pass shared sentence embeddings for each task head
        logits_a = self.classifier_a(sentence_embeddings)
        logits_b = self.classifier_b(sentence_embeddings)

        return logits_a, logits_b