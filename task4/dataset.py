# dataset.py
import torch
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        title = str(self.data.loc[idx, 'title'])
        sentence = str(self.data.loc[idx, 'sentence'])
        score = self.data.loc[idx].get('score',0.0)

        inputs = self.tokenizer(title, sentence, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.max_len)
        

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
     
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'score': torch.tensor(score, dtype=torch.float)
        }