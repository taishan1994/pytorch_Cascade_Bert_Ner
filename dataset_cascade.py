import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import BertFeature
from utils import commonUtils


class NerDataset(Dataset):
    def __init__(self, features):
        # self.callback_info = callback_info
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks, dtype=torch.uint8) for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.bio_labels = [torch.tensor(example.bio_labels).long() for example in features]
        self.att_labels = [torch.tensor(example.att_labels).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': self.token_ids[index],
            'attention_masks': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index]
        }

        data['bio_labels'] = self.bio_labels[index]
        data['att_labels'] = self.att_labels[index]
        return data
