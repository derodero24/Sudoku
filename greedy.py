
import matplotlib.pyplot as plt
import pandas as pd


def plot_sudoku(firstdf, df, fin=False):
    plt.axis('off')  # グラフ外枠削除
    plt.hlines([0, 3, 6, 9], 0, 9)
    plt.vlines([0, 3, 6, 9], 0, 9)
    plt.hlines([1, 2, 4, 5, 7, 8], 0, 9, linestyles='dotted')
    plt.vlines([1, 2, 4, 5, 7, 8], 0, 9, linestyles='dotted')

    for i in range(9 * 9):
        x, y = i % 9, i // 9
        plt.text(0.35 + x, 0.3 + y, df[x][8 - y], fontsize=14)
    for i in range(9 * 9):
        x, y = i % 9, i // 9
        plt.text(0.35 + x, 0.3 + y, firstdf[x][8 - y],
                 fontsize=14, color='r')

    if fin:
        print('Finish !!!\n', df)
        plt.show()
    else:
        plt.pause(0.001)  # 描画(待ち時間)
        plt.cla()  # 描写消去


def check_non(df, x, y):
    col = df[x].values
    col_non = [str(i + 1) for i in range(9) if not str(i + 1) in col]

    row = df.loc[y].values
    row_non = [str(i + 1) for i in range(9) if not str(i + 1) in row]

    box = [df[x // 3 * 3 + i][y // 3 * 3 + j]
           for i in range(3) for j in range(3)]
    box_non = [str(i + 1) for i in range(9) if not str(i + 1) in box]

    return set(row_non) & set(col_non) & set(box_non)


def sudoku_step(firstdf, df, pos):
    x, y = pos % 9, pos // 9
    # 終了判定
    if pos == 9 * 9:
        plot_sudoku(firstdf, df, fin=True)
        exit()
    # 空欄判定
    if df[x][y] != '':
        sudoku_step(firstdf, df, pos + 1)
        return
    # 入力可能な数字
    non_list = check_non(df, x, y)
    # 矛盾判定
    if not non_list:
        return
    # 書き込み
    for num in non_list:
        df[x][y] = num
        plot_sudoku(firstdf, df, fin=False)
        sudoku_step(firstdf, df, pos + 1)
    df[x][y] = ''


def main():
    df = pd.read_csv('sudoku1.csv', dtype='object', header=None)
    df = df.fillna('')
    print(df)

    sudoku_step(df.copy(), df, pos=0)


if __name__ == '__main__':
    main()
