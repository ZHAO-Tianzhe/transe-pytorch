import torch
from torch.utils.data import Dataset


class TripleSet(Dataset):

    def __init__(self, num_of_triples):
        self.num_of_triples = num_of_triples
        self.triple_id_list = torch.arange(self.num_of_triples, dtype=torch.int64)

    def __len__(self):
        return self.num_of_triples

    def __getitem__(self, item):
        return self.triple_id_list[item]

