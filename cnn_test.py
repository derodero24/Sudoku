
import numpy as np
import pandas as pd

import Deropy.neural as nrl


def to_data(df):
    data = []
    for i in range(9):
        one_hot = np.eye(10)[df[i].values]
        data.append(one_hot)
    return np.array([data])


def from_data(predict):
    df = pd.DataFrame()
    for i in range(9):
        tmp = [predict[i][j].argmax() for j in range(9)]
        df[i] = tmp
    return df


def main():
    df = pd.read_csv('sudoku1.csv', header=None)
    df = df.fillna(0).astype('uint8')
    print(df)

    data = to_data(df)
    print(data.shape)

    model = nrl.load_model('result/cnn_model2')
    predict = model.predict(data)
    print(predict[0].shape)

    df = from_data(predict[0])
    print(df)


if __name__ == '__main__':
    main()
