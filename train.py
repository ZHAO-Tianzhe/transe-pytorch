import data_input


class Train:
    def __init__(self):
        self.dataset_name = "TestData"

        print("---Data Input---")
        input_data = data_input.DataInput(self.dataset_name)



if __name__ == "__main__":
    train = Train()


