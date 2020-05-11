# read training, validating, and testing data


class DataInput:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.file_path = "./data/" + self.dataset_name
        self.numbers = {"entity": 0, "relation": 0, "train_triple": 0, "valid_triple": 0, "test_triple": 0}
        self.train_triples = {}  # {training_triple_id: [head entity, relation, tail entity]}
        self.valid_triples = {}  # {validating_triple_id: [head entity, relation, tail entity]}
        self.test_triples = {}  # {testing_triple_id: [head entity, relation, tail entity]}

        self.read_data()

    def read_data(self):
        with open(self.file_path + "/entity2id.txt", "r") as f:
            self.numbers["entity"] = int(f.readline())
            print("number of entities: " + str(self.numbers["entity"]))
        with open(self.file_path + "/relation2id.txt", "r") as f:
            self.numbers["relation"] = int(f.readline())
            print("number of relations: " + str(self.numbers["relation"]))
        names = ["/train2id.txt", "/valid2id.txt", "/test2id.txt"]
        triple_names = ["train_triple", "valid_triple", "test_triple"]
        triples = [self.train_triples, self.valid_triples, self.test_triples]
        for tmp in [0, 1, 2]:
            self.read_triples(names[tmp], triple_names[tmp], triples[tmp])
        print("number of training triples: " + str(self.numbers["train_triple"]))
        # print(self.train_triples)
        print("number of validating triples: " + str(self.numbers["valid_triple"]))
        # print(self.valid_triples)
        print("number of testing triples: " + str(self.numbers["test_triple"]))
        # print(self.test_triples)

    def read_triples(self, name, triple_name, triples):
        with open(self.file_path + name, "r") as f:
            self.numbers[triple_name] = int(f.readline())
            tmp_id = 0
            tmp_line = f.readline()
            while tmp_line:
                triples[tmp_id] = [int(tmp_line.split()[0]), int(tmp_line.split()[2]), int(tmp_line.split()[1])]
                tmp_id += 1
                tmp_line = f.readline()


if __name__ == "__main__":
    data_input = DataInput("FB15K")