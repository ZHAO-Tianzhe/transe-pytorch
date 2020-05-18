# the neural network of this implementation
# storing all entity and relation embeddings
# taking a batch as input and returning the loss of this batch
import torch
import torch.nn as nn
import torch.nn.functional as func


class TransE(nn.Module):
    def __init__(self, num_of_entities, num_of_relations, entity_dimension, relation_dimension, margin, norm):
        super(TransE, self).__init__()
        self.num_of_entities = num_of_entities
        tmp_entity_embeddings = (6. / (entity_dimension ** 0.5) + 6. / (entity_dimension ** 0.5)) * torch.rand(
            self.num_of_entities, entity_dimension) - 6. / (entity_dimension ** 0.5)
        tmp_entity_embeddings = func.normalize(tmp_entity_embeddings, 2, 1)
        self.entity_embeddings = nn.Embedding.from_pretrained(tmp_entity_embeddings)

        tmp_relation_embeddings = (6. / (relation_dimension ** 0.5) + 6. / (relation_dimension ** 0.5)) * torch.rand(
            num_of_relations, relation_dimension) - 6. / (relation_dimension ** 0.5)
        tmp_relation_embeddings = func.normalize(tmp_relation_embeddings, 2, 1)
        self.relation_embeddings = nn.Embedding.from_pretrained(tmp_relation_embeddings)

        self.margin = margin
        self.norm = norm

    def forward(self, positive_head_batch, positive_relation_batch, positive_tail_batch, negative_head_batch,
                negative_relation_batch, negative_tail_batch):

        positive_head_embeddings = self.entity_embeddings(positive_head_batch)
        positive_relation_embeddings = self.relation_embeddings(positive_relation_batch)
        positive_tail_embeddings = self.entity_embeddings(positive_tail_batch)

        negative_head_embeddings = self.entity_embeddings(negative_head_batch)
        negative_relation_embeddings = self.relation_embeddings(negative_relation_batch)
        negative_tail_embeddings = self.entity_embeddings(negative_tail_batch)

        positive_losses = torch.norm(positive_head_embeddings + positive_relation_embeddings - positive_tail_embeddings,
                                   self.norm, 1)
        negative_losses = torch.norm(negative_head_embeddings + negative_relation_embeddings - negative_tail_embeddings,
                                   self.norm, 1)
        losses = torch.cat((positive_losses.unsqueeze(0), negative_losses.unsqueeze(0)), 0)
        return losses

    def validate_and_test(self, head_batch, relation_batch, tail_batch):
        head_embeddings = self.entity_embeddings(head_batch)
        relation_embeddings = self.relation_embeddings(relation_batch)
        tail_embeddings = self.entity_embeddings(tail_batch)
        target_losses = torch.norm(head_embeddings + relation_embeddings - tail_embeddings, self.norm, 1).unsqueeze(
            1).repeat(1, self.num_of_entities)

        candidate_head_losses = torch.norm(
            self.entity_embeddings.weight.data.unsqueeze(0) + relation_embeddings.unsqueeze(
                1) - tail_embeddings.unsqueeze(1), self.norm, 2)
        candidate_tail_losses = torch.norm(head_embeddings.unsqueeze(1) + relation_embeddings.unsqueeze(
            1) - self.entity_embeddings.weight.data.unsqueeze(0), self.norm, 2)

        return torch.cat((torch.cat((target_losses.unsqueeze(0), candidate_head_losses.unsqueeze(0)), 0),
                          candidate_tail_losses.unsqueeze(0)), 0)






