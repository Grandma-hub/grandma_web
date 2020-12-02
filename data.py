import numpy as np


class Data:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def split_train_test(self, train_percent, random=False):
        percent = int(len(self.data_x) * train_percent / 100)
        if random is True:
            ind = np.random.permutation(self.data_x.index)
            self.data_x = self.data_x.reindex(ind)
            self.data_y = self.data_y.reindex(ind)

        train_x = self.data_x.iloc[:percent]
        train_y = self.data_y.iloc[:percent]
        test_x = self.data_x.iloc[percent:]
        test_y = self.data_y.iloc[percent:]

        return train_x, train_y, test_x, test_y
