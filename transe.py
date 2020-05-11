# the neural network of this implementation
# storing all entity and relation embeddings
# taking a batch as input and returning the loss of this batch
import torch
import torch.nn as nn
import torch.nn.functional as func


class TransE:
    def __init__(self, num_of_entities, num_of_relations, entity_dimension, relation_dimension, margin, norm):
        super(TransE, self).__init__()
        tmp_entity_embeddings = (6. / (entity_dimension ** 0.5) + 6. / (entity_dimension ** 0.5)) * torch.rand(
            num_of_entities, entity_dimension) - 6. / (entity_dimension ** 0.5)
        self.entity_embeddings = nn.Embedding.from_pretrained(tmp_entity_embeddings)

        tmp_relation_embeddings = (6. / (relation_dimension ** 0.5) + 6. / (relation_dimension ** 0.5)) * torch.rand(
            num_of_relations, relation_dimension) - 6. / (relation_dimension ** 0.5)
        tmp_relation_embeddings = func.normalize(tmp_relation_embeddings, 2, 1)
        self.relation_embeddings = nn.Embedding.from_pretrained(tmp_relation_embeddings)

        print(self.entity_embeddings.weight)
        print(self.relation_embeddings.weight)
