
import threading
import time

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('sudoku1.csv', dtype='object', header=None)
df = df.fillna('')


def plot_sudoku(firstdf):
    while True:
        plt.axis('off')  # グラフ外枠削除
        plt.hlines([0, 3, 6, 9], 0, 9)
        plt.vlines([0, 3, 6, 9], 0, 9)
        plt.hlines([1, 2, 4, 5, 7, 8], 0, 9, linestyles='dotted')
        plt.vlines([1, 2, 4, 5, 7, 8], 0, 9, linestyles='dotted')

        non_sum = (df == '').values.sum()

        for i in range(9 * 9):
            x, y = i % 9, i // 9
            plt.text(0.35 + x, 0.3 + y, df[x][8 - y], fontsize=14)
        for i in range(9 * 9):
            x, y = i % 9, i // 9
            plt.text(0.35 + x, 0.3 + y, firstdf[x][8 - y],
                     fontsize=14, color='r')

        if non_sum:  # 空欄あり
            print(non_sum)
            plt.pause(0.01)  # 描画(待ち時間)
            plt.cla()  # 描写消去
        else:
            print('Finish !!!\n', df)
            plt.show()
            return


def check_non(df, x, y):
    col = df[x].values
    col_non = [str(i + 1) for i in range(9) if not str(i + 1) in col]

    row = df.loc[y].values
    row_non = [str(i + 1) for i in range(9) if not str(i + 1) in row]

    box = [df[x // 3 * 3 + i][y // 3 * 3 + j]
           for i in range(3) for j in range(3)]
    box_non = [str(i + 1) for i in range(9) if not str(i + 1) in box]

    return set(row_non) & set(col_non) & set(box_non)


def monitor(pos):
    time.sleep(1)
    x, y = pos % 9, pos // 9
    # 空欄判定
    if df[x][y] != '':
        return
    while True:
        # 入力可能な数字
        non_list = check_non(df, x, y)
        if len(non_list) == 1:
            df[x][y] = list(non_list)[0]
            print((x, y), ':', list(non_list)[0])
            return
        time.sleep(2)


def main():
    print(df)

    # threading.Thread(target=plot_sudoku, args=([df.copy()])).start()

    for i in range(9 * 9):
        threading.Thread(target=monitor, args=([i])).start()
    # print(df)

    plot_sudoku(df.copy())


if __name__ == '__main__':
    main()
