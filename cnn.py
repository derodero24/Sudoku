
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, Concatenate, Conv2D, Dense, Flatten,
                          Input, Reshape)
from keras.models import Model

import Deropy.common as cmn
import Deropy.neural as nrl
import Deropy.visual as vsl

'''オプション'''
train_size = 50000
test_size = 500
batch_size = 64
epochs = 5
modelname = 'result/cnn_model3'
''''''


def create_model():
    inp = Input((9, 9, 10))
    # (None, 9, 9, 10)
    x1 = Conv2D(filters=32, kernel_size=(3, 3),
                strides=(3, 3), activation='relu')(inp)
    x2 = Conv2D(filters=32, kernel_size=(9, 1), activation='relu')(inp)
    x3 = Conv2D(filters=32, kernel_size=(1, 9), activation='relu')(inp)

    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    x3 = Flatten()(x3)
    # (None, 90)

    x = Concatenate()([x1, x2, x3])  # -> (None, 270)
    # x = Dense(9 * 9 * 10)(x)
    x = Dense(9 * 9 * 10)(x)
    x = Reshape((9, 9, 10))(x)
    x = Activation('softmax')(x)

    model = Model(inp, x)
    model.summary()
    return model


class sudoku_generator:
    def __init__(self, train_size, test_size):
        df = pd.read_csv(cmn.dpath('sudoku.csv'))
        df_s = df.sample(frac=1, random_state=0)  # シャッフル
        self.train_df = df[:train_size]
        self.test_df = df[train_size:train_size + test_size]

    def reshape(self, num_string):
        tmp = np.array([int(num) for num in num_string]).reshape(9, 9)
        return [np.eye(10)[row] for row in tmp]  # one-hot表現

    def gen(self, mode):
        df = self.train_df if mode == 'train' else self.test_df
        quizze, solution = [], []
        while True:
            for index, row in df.iterrows():
                quizze.append(self.reshape(row['quizzes']))
                solution.append(self.reshape(row['solutions']))
                if len(quizze) == batch_size:
                    yield np.array(quizze), np.array(solution)
                    quizze, solution = [], []


def main():
    sudoku_gen = sudoku_generator(train_size=train_size, test_size=test_size)
    train_gen = sudoku_gen.gen(mode='train')
    test_gen = sudoku_gen.gen(mode='test')

    # model = create_model()
    model = nrl.load_model(modelname)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    es_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    histry = model.fit_generator(train_gen,
                                 validation_data=test_gen,
                                 steps_per_epoch=train_size // batch_size,
                                 validation_steps=test_size // batch_size,
                                 epochs=epochs,
                                 callbacks=[es_callback])

    nrl.save_model(model, filename=modelname)
    nrl.save_hist(histry, filename=modelname + '_hist')
    vsl.plot_tsv(filename=modelname + '_hist', ylim=(0, 1.05))


if __name__ == '__main__':
    main()
