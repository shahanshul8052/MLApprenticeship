# hugging face and pytorch

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class SentenceTransformer(nn.Module):
    """
    A wrapper around a Hugging Face transformer model to produce a fixed length embedding of our sentences
    """

    def __init__(self, model_name="distilbert-base-uncased", pooling="mean"):
        """
        Args: 
            model_name (str): The name of the Hugging Face transformer model to use.
            pooling (str): The pooling strategy to use. Can be "mean", "max", or "cls".
        """

        super(SentenceTransformer, self).__init__()
        # loaded tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load the model
        self.model = AutoModel.from_pretrained(model_name)
        # evaluate the model
        self.pooling = pooling

    
    def forward(self, sentences):
        """
        tokenizes the input and gives us our sentence emvedding 

        args:
            sentences (list): A list of sentences to encode.

        returns:
            torch.Tensor: The sentence embeddings with batch size and hidden size
        """

        # tokenize the input
        # we use the tokenizer to convert the sentences into a format that the model can understand
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # pass the input through the model
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        
        # apply the pooling strategy and return the output
        if self.pooling == "mean":
            # mean pooling
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
        elif self.pooling == "max":
            # max pooling
            sentence_embeddings, _ = torch.max(model_output.last_hidden_state, dim=1)
        elif self.pooling == "cls":
            # cls pooling
            sentence_embeddings = model_output.last_hidden_state[:, 0, :]
        else:
            raise ValueError("Pooling strategy not recognized. Use 'mean', 'max', or 'cls'.")
        return sentence_embeddings

