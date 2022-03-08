# def get_board(n):
#     # get the numbers
#     numbers = [i for i in range(16 * 14)]
#
#     # create the nested list representing the board
#     #rev_board = [numbers[i:i+n][::1] for i in range(0, len(numbers), n)]
#     print(numbers)
#     return #rev_board
#
# board = get_board(3)
# print(board)
import numpy as np
#w = np.ones((5,224))
m = np.zeros((16,14))
#m[0][13] = 1
w = 16
h = 14
num_list = [i for i in range(1, w * h +1)]
print(num_list)
n = 1
for i in range(w):
    for j in range(h):
        m[i][j] = n
        n += 1

def get_state (m):
    s = np.array(np.where(m == 1)).T.flatten()
    print(s)

    x = s[0]
    y = s[1]


    return x,y

print(get_state(m))
print()