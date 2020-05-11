# read training, validating, and testing data


class DataInput:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.file_path = "./data/" + self.dataset_name
        self.numbers = {"entity": 0, "relation": 0, "train_triple": 0, "valid_triple": 0, "test_triple": 0}
        self.triples = {"train_triple": {}, "valid_triple": {}, "test_triple": {}}

    def read_data(self):
        with open(self.file_path + "/entity2id.txt", "r") as f:
            self.numbers["entity"] = int(f.readline())
        with open(self.file_path + "/relation2id.txt", "r") as f:
            self.numbers["relation"] = int(f.readline())
        names = ["/train2id.txt", "/valid2id.txt", "/test2id.txt"]
        triple_names = ["train_triple", "valid_triple", "test_triple"]
        for tmp in [0, 1, 2]:
            self.read_triples(names[tmp], triple_names[tmp])

    def read_triples(self, name, triple_name):
        with open(self.file_path + name, "r") as f:
            self.numbers[triple_name] = int(f.readline())
            tmp_id = 0
            tmp_line = f.readline()
            while tmp_line:
                self.triples[triple_name][tmp_id] = [int(tmp_line.split()[0]), int(tmp_line.split()[2]), int(tmp_line.split()[1])]
                tmp_id += 1
                tmp_line = f.readline()


if __name__ == "__main__":
    data_input = DataInput("FB15K")