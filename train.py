# the main class of this implementation
# including configurations and every function needed for training
import data_input
import transe


class Train:
    def __init__(self):
        self.dataset_name = "TestData"
        self.continue_or_not = False
        self.existing_embeddings_path = None
        self.entity_dimension = 5
        self.relation_dimension = 5
        self.num_of_epochs = 20
        self.output_frequency = 5
        self.num_of_batches = 2
        self.learning_rate = 0.01
        self.margin = 1.0
        self.norm = 1

        print("---Configurations---")
        print("dataset name: %s" % self.dataset_name)
        if self.continue_or_not:
            print("continue training based on: %s" % self.existing_embeddings_path)
        else:
            print("a new training")
        print("entity dimension: %d" % self.entity_dimension)
        print("relation dimension: %d" % self.relation_dimension)
        print("number of epochs: %d" % self.num_of_epochs)
        print("output embeddings every %d epochs" % self.output_frequency)
        print("number of batches: %d" % self.num_of_batches)
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
        self.train_triples = input_data.triples["train_triple"]  # {training_triple_id: [head entity, relation, tail entity]}
        self.valid_triples = input_data.triples["valid_triple"]  # {validating_triple_id: [head entity, relation, tail entity]}
        self.test_triples = input_data.triples["test_triple"]  # {testing_triple_id: [head entity, relation, tail entity]}
        print("number of entities: %d" % self.num_of_entities)
        print("number of relations: %d" % self.num_of_relations)
        print("number of training triples: %d" % self.num_of_train_triples)
        #print(self.train_triples)
        print("number of validating triples: %d" % self.num_of_valid_triples)
        #print(self.valid_triples)
        print("number of testing triples: %d" % self.num_of_test_triples)
        #print(self.test_triples)

        print("---Training---")
        transe_network = transe.TransE(self.num_of_entities, self.num_of_relations, self.entity_dimension,
                                       self.relation_dimension, self.margin, self.norm)





if __name__ == "__main__":
    train = Train()


