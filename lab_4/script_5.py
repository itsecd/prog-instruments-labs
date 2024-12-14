import os
import csv


class Iterator:
    def __init__(self, class_name, dataset_name):
        self.dataset_name = dataset_name
        self.counter = 0
        self.class_name = class_name
        self.data = os.listdir(os.path.join(dataset_name, self.class_name))
        self.limit = len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.limit:
            next_path = os.path.join(
                self.dataset_name, self.class_name, self.data[self.counter])
            self.counter += 1
            return next_path
        else:
            return None


if __name__ == "__main__":

    leopard = Iterator('leopard')
    tiger = Iterator('tiger')

    print(next(leopard))
    print(next(leopard))
    print(next(leopard))
    print(next(tiger))
    print(next(tiger))
    print(next(tiger))
