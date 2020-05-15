# the main class of this implementation
# including configurations and every function needed for training
import data_input
import transe
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import dataset_process


class Train:
    def __init__(self):
        self.dataset_name = "TestData"
        self.continue_or_not = False
        self.existing_embeddings_path = "./data/output/"
        self.entity_dimension = 5
        self.relation_dimension = 5
        self.num_of_epochs = 20
        self.output_frequency = 5
        self.num_of_batches = 2
        self.num_of_valid_batches = 1
        self.num_of_test_batches = 1
        self.learning_rate = 0.01
        self.margin = 1.0
        self.norm = 1
        self.early_stop_patience = 5

        print("---Configurations---")
        print("dataset name: %s" % self.dataset_name)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print("training on %s" % self.device)
        if self.continue_or_not:
            print("continue training based on: %s" % self.existing_embeddings_path)
        else:
            print("a new training")
        print("entity dimension: %d" % self.entity_dimension)
        print("relation dimension: %d" % self.relation_dimension)
        print("number of epochs: %d" % self.num_of_epochs)
        print("output embeddings every %d epochs" % self.output_frequency)
        print("number of training batches: %d" % self.num_of_batches)
        print("number of validating batches: %d" % self.num_of_valid_batches)
        print("number of testing batches: %d" % self.num_of_test_batches)
        print("learning rate: %f" % self.learning_rate)
        print("learning margin: %f" % self.margin)
        print("learning norm: %d" % self.norm)

        print("---Data Input---")
        input_data = data_input.DataInput(self.dataset_name)
        input_data.read_data()
        self.num_of_entities = input_data.numbers["entity"]
        self.num_of_relations = input_data.numbers["relation"]
        self.num_of_train_triples = input_data.numbers["train_triple"]
        self.num_of_valid_triples = input_data.numbers["valid_triple"]
        self.num_of_test_triples = input_data.numbers["test_triple"]
        self.train_triples = input_data.triples[
            "train_triple"]  # {training_triple_id: [head entity, relation, tail entity]}
        self.valid_triples = input_data.triples[
            "valid_triple"]  # {validating_triple_id: [head entity, relation, tail entity]}
        self.valid_batches = self.validate_and_test_batches(self.num_of_valid_triples, self.num_of_valid_batches)
        self.test_triples = input_data.triples[
            "test_triple"]  # {testing_triple_id: [head entity, relation, tail entity]}
        self.test_batches = self.validate_and_test_batches(self.num_of_test_triples, self.num_of_test_batches)
        print("number of entities: %d" % self.num_of_entities)
        print("number of relations: %d" % self.num_of_relations)
        print("number of training triples: %d" % self.num_of_train_triples)
        #print(self.train_triples)
        print("number of validating triples: %d" % self.num_of_valid_triples)
        #print(self.valid_triples)
        print("number of testing triples: %d" % self.num_of_test_triples)
        #print(self.test_triples)
        if self.continue_or_not:
            with open(("%s_%s_entity_embeddings.pickle" % (self.existing_embeddings_path, self.dataset_name)),
                      "r") as f:
                self.existing_entity_embeddings = pickle.load(f)
            with open(("%s_%s_relation_embeddings.pickle" % (self.existing_embeddings_path, self.dataset_name)),
                          "r") as f:
                self.existing_relation_embeddings = pickle.load(f)

        print("---Training---")
        self.transe_network = None
        self.training()

        print("---Testing---")
        self.testing()

    def negative_triple_sampling(self):
        negative_triples = {}
        head_or_tail = torch.randint(low=0, high=2, size=(self.num_of_train_triples,)).tolist()
        for tmp_id in range(self.num_of_train_triples):
            if head_or_tail[tmp_id]:
                negative_heads = torch.randperm(n=self.num_of_entities).tolist()
                tmp_head_id = 0
                while [negative_heads[tmp_head_id], self.train_triples[tmp_id][1],
                       self.train_triples[tmp_id][2]] in self.train_triples.values():
                    tmp_head_id += 1
                negative_triples[tmp_id] = [negative_heads[tmp_head_id], self.train_triples[tmp_id][1],
                                            self.train_triples[tmp_id][2]]
            else:
                negative_tails = torch.randperm(n=self.num_of_entities).tolist()
                tmp_tail_id = 0
                while [self.train_triples[tmp_id][0], self.train_triples[tmp_id][1],
                       negative_tails[tmp_tail_id]] in self.train_triples.values():
                    tmp_tail_id += 1
                negative_triples[tmp_id] = [self.train_triples[tmp_id][0], self.train_triples[tmp_id][1],
                                            negative_tails[tmp_tail_id]]
        return negative_triples

    @staticmethod
    def validate_and_test_batches(num_of_triples, num_of_batches):
        triple_set = dataset_process.TripleSet(num_of_triples)
        batch_size = int(num_of_triples / num_of_batches)
        data_loader = DataLoader(dataset=triple_set, batch_size=batch_size)
        return [batch.tolist() for batch in data_loader]

    def ranking_computing(self, batches, triples):
        ranking_func = nn.MarginRankingLoss(margin=0, reduction="none").to(self.device)
        mean_ranks = 0
        for batch in batches:
            head_batch = torch.tensor([triples[tmp_id][0] for tmp_id in batch], dtype=torch.int64)
            relation_batch = torch.tensor([triples[tmp_id][1] for tmp_id in batch], dtype=torch.int64)
            tail_batch = torch.tensor([triples[tmp_id][2] for tmp_id in batch], dtype=torch.int64)

            valid_losses = self.transe_network.validate_and_test(head_batch.to(self.device),
                                                            relation_batch.to(self.device),
                                                            tail_batch.to(self.device))
            comparing_heads = ranking_func(valid_losses[0], valid_losses[1],
                                           torch.tensor([-1], dtype=torch.float).to(self.device))
            head_mean_rank = torch.nonzero(comparing_heads).size()[0] / len(batch)
            comparing_tails = ranking_func(valid_losses[0], valid_losses[2],
                                           torch.tensor([-1], dtype=torch.float).to(self.device))
            tail_mean_rank = torch.nonzero(comparing_tails).size()[0] / len(batch)
            mean_ranks += (head_mean_rank + tail_mean_rank) / 2
        return mean_ranks / len(batches)

    def training(self):
        self.transe_network = transe.TransE(self.num_of_entities, self.num_of_relations, self.entity_dimension,
                                       self.relation_dimension, self.margin, self.norm)
        if self.continue_or_not:
            self.transe_network.entity_embeddings.weight.data = self.existing_entity_embeddings
            self.transe_network.relation_embeddings.weight.data = self.existing_relation_embeddings
        self.transe_network.to(self.device)

        loss_func = nn.MarginRankingLoss(margin=self.margin, reduction="sum").to(self.device)
        optimizer = optim.SGD(params=self.transe_network.parameters(), lr=self.learning_rate)

        triple_set = dataset_process.TripleSet(self.num_of_train_triples)
        batch_size = int(self.num_of_train_triples / self.num_of_batches)
        print("batch size: %d" % batch_size)
        data_loader = DataLoader(dataset=triple_set, batch_size=batch_size, shuffle=True)

        for epoch in range(self.num_of_epochs):
            epoch_loss = 0.
            best_valid_result = self.num_of_entities
            fall_times = 0
            negative_triples = self.negative_triple_sampling()
            for batch in data_loader:
                batch = batch.tolist()
                optimizer.zero_grad()

                positive_head_batch = torch.tensor(data=[self.train_triples[tmp_id][0] for tmp_id in batch])
                positive_relation_batch = torch.tensor(data=[self.train_triples[tmp_id][1] for tmp_id in batch])
                positive_tail_batch = torch.tensor(data=[self.train_triples[tmp_id][2] for tmp_id in batch])

                negative_head_batch = torch.tensor(data=[negative_triples[tmp_id][0] for tmp_id in batch])
                negative_relation_batch = torch.tensor(data=[negative_triples[tmp_id][1] for tmp_id in batch])
                negative_tail_batch = torch.tensor(data=[negative_triples[tmp_id][2] for tmp_id in batch])

                batch_losses = self.transe_network(positive_head_batch.to(self.device),
                                              positive_relation_batch.to(self.device),
                                              positive_tail_batch.to(self.device),
                                              negative_head_batch.to(self.device),
                                              negative_relation_batch.to(self.device),
                                              negative_tail_batch.to(self.device))
                batch_loss = loss_func(batch_losses[0], batch_losses[1],
                                       torch.tensor([-1], dtype=torch.float).to(self.device))
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
            print("epoch: %d, loss: %f" % (epoch, epoch_loss))
            valid_mean_rank = self.ranking_computing(self.valid_batches, self.train_triples)
            if valid_mean_rank < best_valid_result:
                print("validating mean rank drops from: %d to %d" % (best_valid_result, valid_mean_rank))
                best_valid_result = valid_mean_rank
                fall_times = 0
                self.existing_entity_embeddings = self.transe_network.entity_embeddings.weight.data.clone()
                self.existing_relation_embeddings = self.transe_network.relation_embeddings.weight.data.clone()
            else:
                print("validating mean rank increases from: %d to %d" % (best_valid_result, valid_mean_rank))
                fall_times += 1
                if fall_times == self.early_stop_patience:
                    break
        with open(("%s_%s_entity_embeddings.pickle" % (self.existing_embeddings_path, self.dataset_name)),
                  "w") as f:
            pickle.dump(self.existing_entity_embeddings.to("cpu"), f)
        with open(("%s_%s_relation_embeddings.pickle" % (self.existing_embeddings_path, self.dataset_name)),
                  "w") as f:
            pickle.dump(self.existing_relation_embeddings.to("cpu"), f)

    def testing(self):
        test_mean_rank = self.ranking_computing(self.test_batches, self.test_triples)
        print("testing mean rank: %d" % test_mean_rank)


if __name__ == "__main__":
    train = Train()


