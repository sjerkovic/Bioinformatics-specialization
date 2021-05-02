import random

import numpy as np
import pandas as pd


def dp_change(money, coins):
    min_num_coins = {0: 0}
    for m in range(1, money + 1):
        min_num_coins[m] = np.inf
        for i in range(len(coins)):
            if m >= coins[i]:
                if min_num_coins[m - coins[i]] + 1 < min_num_coins[m]:
                    min_num_coins[m] = min_num_coins[m - coins[i]] + 1
    return min_num_coins[money]


def string_to_arrays(string):
    arrays = string.split('-')
    arr1 = arrays[0].split('\n')[:-1]
    arr2 = arrays[1].split('\n')[1:]
    arr1 = [i.split(' ') for i in arr1]
    arr2 = [i.split(' ') for i in arr2]
    arr1 = [[int(i) for i in j] for j in arr1]
    arr2 = [[int(i) for i in j] for j in arr2]
    return arr1, arr2


def manhattan_tourist(n, m, down, right):
    s = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        s[i][0] = s[i - 1][0] + down[i - 1][0]
    for j in range(1, m + 1):
        s[0][j] = s[0][j - 1] + right[0][j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s[i][j] = max(s[i - 1][j] + down[i - 1][j], s[i][j - 1] + right[i][j - 1])
    return int(s[n][m])


def lcs_backtrack(v, w):
    s = np.zeros((len(v) + 1, len(w) + 1))
    backtrack = np.chararray((len(v), len(w)), unicode=True)
    for i in range(1, len(v) + 1):
        for j in range(1, len(w) + 1):
            match = 0
            if v[i - 1] == w[j - 1]:
                match = 1
            s[i][j] = max(s[i - 1][j], s[i][j - 1], s[i - 1][j - 1] + match)
            if s[i][j] == s[i - 1][j]:
                backtrack[i - 1][j - 1] = '↓'
            elif s[i][j] == s[i][j - 1]:
                backtrack[i - 1][j - 1] = '→'
            elif s[i][j] == s[i - 1][j - 1] + match:
                backtrack[i - 1][j - 1] = '↘'
    return backtrack


def output_lcs(backtrack, v, i, j):
    if i == 0 or j == 0:
        return ''
    if backtrack[i - 1][j - 1] == '↓':
        return output_lcs(backtrack, v, i - 1, j)
    elif backtrack[i - 1][j - 1] == '→':
        return output_lcs(backtrack, v, i, j - 1)
    else:
        return output_lcs(backtrack, v, i - 1, j - 1) + v[i - 1]


def topological_order(string):
    split_string = string.split('\n')
    edges_weight = [(v.split(':')[0], int(v.split(':')[1])) for v in split_string]
    prev_node = list()
    next_node = list()

    for i in edges_weight:
        prev_node.append(int(i[0].split('->')[0]))
        next_node.append(int(i[0].split('->')[1]))
    # zipped_graph = zip(prev_node,edges_weight)
    # sorted_zipped_graph = sorted(zipped_graph)
    # sorted_graph = [v[1] for v in sorted_zipped_graph]

    while len(prev_node) != 0:
        current_node_index = random.randint(len(prev_node))
        current_node = prev_node[current_node_index]
        if

    return prev_node, next_node


def global_aligment(v, w):
    GAP_PENALTY = -2
    n = len(v)
    m = len(w)

    backtrack = [[(-1, -1) for j in range(m + 1)] for i in range(n + 1)]
    s = [[0 for j in range(m + 1)] for i in range(n + 1)]

    for i in range(1, n + 1):
        s[i][0] = s[i - 1][0] + GAP_PENALTY
        backtrack[i][0] = (i - 1, 0)

    for j in range(1, m + 1):
        s[0][j] = s[0][j - 1] + GAP_PENALTY
        backtrack[0][j] = (0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s[i][j] = max(
                s[i - 1][j] + GAP_PENALTY,
                s[i][j - 1] + GAP_PENALTY,
                s[i - 1][j - 1] + match_score(v[i - 1], w[j - 1]))
            if s[i][j] == s[i - 1][j] + GAP_PENALTY:
                backtrack[i][j] = (i - 1, j)
            elif s[i][j] == s[i][j - 1] + GAP_PENALTY:
                backtrack[i][j] = (i, j - 1)
            else:
                backtrack[i][j] = (i - 1, j - 1)

    v_p = ''
    w_p = ''
    i = n
    j = m
    while (i, j) != (0, 0):
        if backtrack[i][j] == (i - 1, j - 1):
            v_p = v[i - 1] + v_p
            w_p = w[j - 1] + w_p

        elif backtrack[i][j] == (i - 1, j):
            v_p = v[i - 1] + v_p
            w_p = '-' + w_p

        else:
            v_p = '-' + v_p
            w_p = w[j - 1] + w_p

        (i, j) = backtrack[i][j]

    return s, v_p, w_p


def hamming_distance(s1, s2):
    score = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            score += 1
    return score


def local_aligment_numpy(v, w):
    m = len(v)
    n = len(w)

    s = np.zeros((m + 1, n + 1))

    backtrack = np.chararray((m + 1, n + 1), unicode=True)
    for i in range(1, m + 1):
        backtrack[i][0] = '↓'

    for j in range(1, n + 1):
        backtrack[0][j] = '→'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s[i][j] = max(0,
                          s[i - 1][j] - 1,
                          s[i][j - 1] - 1,
                          s[i - 1][j - 1] + match_score(v[i - 1], w[j - 1]))
            if s[i][j] == s[i - 1][j] - 1:
                backtrack[i][j] = '↓'
            elif s[i][j] == s[i][j - 1] - 1:
                backtrack[i][j] = '→'
            else:
                backtrack[i][j] = '↘'
    score = np.amax(s)
    v_p = ''
    w_p = ''
    x = np.where(s == np.amax(s))[0][0]
    y = np.where(s == np.amax(s))[1][0]

    while s[x][y] != 0:
        if backtrack[x][y] == '↘':
            v_p += v[x - 1]
            w_p += w[y - 1]
            x = x - 1
            y = y - 1
        elif backtrack[x][y] == '→':
            v_p = v_p + '-'
            w_p += w[y - 1]
            y = y - 1
        else:
            v_p += v[x - 1]
            w_p = w_p + '-'
            x = x - 1
    v_p = v_p[::-1]
    w_p = w_p[::-1]
    print(int(score))
    print(v_p)
    print(w_p)
    return (s)


def local_aligment(v, w):
    m = len(v)
    n = len(w)
    s = pd.DataFrame(0, columns=list('0' + v), index=list('0' + w))
    backtrack = pd.DataFrame(columns=list('0' + v), index=list('0' + w))
    backtrack['0'] = '↓'
    backtrack.loc['0'] = '→'
    backtrack.iloc[0][0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s.iloc[i][j] = max(0,
                               s.iloc[i - 1][j] - 5,
                               s.iloc[i][j - 1] - 5,
                               s.iloc[i - 1][j - 1] + match_score(v[j - 1], w[i - 1]))
            if s.iloc[i][j] == s.iloc[i - 1][j] - 5:
                backtrack.iloc[i][j] = '↓'
            elif s.iloc[i][j] == s.iloc[i][j - 1] - 5:
                backtrack.iloc[i][j] = '→'
            else:
                backtrack.iloc[i][j] = '↘'
    score = s.max().max()
    v_p = ''
    w_p = ''
    current_node = s.unstack().idxmax()
    while s[current_node[0]][current_node[1]] != 0:
        if backtrack[current_node[0]][current_node[1]] == '↘':
            v_p += current_node[0]
            w_p += current_node[1]


def fitting_aligment_numpy(v, w):
    m = len(v)
    n = len(w)

    s = np.zeros((m + 1, n + 1))

    backtrack = np.chararray((m + 1, n + 1), unicode=True)
    for i in range(1, m + 1):
        backtrack[i][0] = '↓'

    for j in range(1, n + 1):
        backtrack[0][j] = '→'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s[i][j] = max(0,
                          s[i - 1][j] - 1,
                          s[i][j - 1] - 1,
                          s[i - 1][j - 1] + match_score(v[i - 1], w[j - 1]))
            if s[i][j] == s[i - 1][j] - 1:
                backtrack[i][j] = '↓'
            elif s[i][j] == s[i][j - 1] - 1:
                backtrack[i][j] = '→'
            else:
                backtrack[i][j] = '↘'
    score = np.amax(s[:, -1])
    v_p = ''
    w_p = ''
    x = np.where(s[:, -1] == score)[0][0]
    y = len(w)

    while y != 0:
        if backtrack[x][y] == '↘':
            v_p += v[x - 1]
            w_p += w[y - 1]
            x = x - 1
            y = y - 1
        elif backtrack[x][y] == '→':
            v_p = v_p + '-'
            w_p += w[y - 1]
            y = y - 1
        else:
            v_p += v[x - 1]
            w_p = w_p + '-'
            x = x - 1
    v_p = v_p[::-1]
    w_p = w_p[::-1]
    print(int(score))
    print(v_p)
    print(w_p)


def overlap_aligment_numpy(v, w):
    m = len(v)
    n = len(w)

    s = np.zeros((m + 1, n + 1))

    backtrack = np.chararray((m + 1, n + 1), unicode=True)
    for i in range(1, m + 1):
        backtrack[i][0] = '↓'

    for j in range(1, n + 1):
        backtrack[0][j] = '→'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s[i][j] = max(
                s[i - 1][j] - 2,
                s[i][j - 1] - 2,
                s[i - 1][j - 1] + match_score(v[i - 1], w[j - 1]))
            if s[i][j] == s[i - 1][j] - 2:
                backtrack[i][j] = '↓'
            elif s[i][j] == s[i][j - 1] - 2:
                backtrack[i][j] = '→'
            else:
                backtrack[i][j] = '↘'
    score = np.amax(s[-1])
    v_p = ''
    w_p = ''
    x = len(v)
    y = np.where(s[-1] == np.amax(s[-1]))[0][0]

    while y != 0:
        if backtrack[x][y] == '↘':
            v_p += v[x - 1]
            w_p += w[y - 1]
            x = x - 1
            y = y - 1
        elif backtrack[x][y] == '→':
            v_p = v_p + '-'
            w_p += w[y - 1]
            y = y - 1
        else:
            v_p += v[x - 1]
            w_p = w_p + '-'
            x = x - 1
    v_p = v_p[::-1]
    w_p = w_p[::-1]
    print(int(score))
    print(v_p)
    print(w_p)
    return s


def affine_gap_aligment_numpy(v, w):
    m = len(v)
    n = len(w)
    s = 11
    e = 1

    s_lower = np.zeros((m + 1, n + 1))
    s_upper = np.zeros((m + 1, n + 1))
    s_middle = np.zeros((m + 1, n + 1))
    s_lower[1][0] = -s
    s_upper[0][1] = -s
    for i in range(2, m + 1):
        s_lower[i][0] = s_lower[i - 1][0] - e
    for i in range(2, n + 1):
        s_upper[0][i] = s_upper[0][i - 1] - e
    backtrack = np.chararray((m + 1, n + 1), unicode=True)

    for i in range(1, m + 1):
        backtrack[i][0] = '↓'

    for j in range(1, n + 1):
        backtrack[0][j] = '→'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s_lower[i][j] = max(
                s_lower[i - 1][j] - e,
                s_middle[i - 1][j] - s)
            s_upper[i][j] = max(
                s_upper[i][j - 1] - e,
                s_middle[i][j - 1] - s)
            s_middle[i][j] = max(
                s_lower[i][j],
                s_middle[i - 1][j - 1] + match_score(v[i - 1], w[j - 1]),
                s_upper[i][j])
            if s_middle[i][j] == s_middle[i - 1][j - 1] + match_score(v[i - 1], w[j - 1]):
                backtrack[i][j] = '↘'
            elif s_middle[i][j] == s_lower[i][j]:
                backtrack[i][j] = '↓'
            else:
                backtrack[i][j] = '→'
    score = s_middle[-1][-1]
    v_p = ''
    w_p = ''
    x = len(v)
    y = len(w)

    while x != 0 and y != 0:
        if backtrack[x][y] == '↘':
            v_p += v[x - 1]
            w_p += w[y - 1]
            x = x - 1
            y = y - 1
        elif backtrack[x][y] == '→':
            v_p = v_p + '-'
            w_p += w[y - 1]
            y = y - 1
        else:
            v_p += v[x - 1]
            w_p = w_p + '-'
            x = x - 1
    v_p = v_p[::-1]
    w_p = w_p[::-1]
    print(int(score))
    print(v_p)
    print(w_p)


def middle_edge_in_lin_space(v, w):
    w_form_source = w[:int(len(w) / 2)]
    w_to_sink = w[int(len(w) / 2):]
    GAP_PENALTY = -2
    n = len(v)
    m = len(w)
    from_source, _, _ = global_aligment(v, w_form_source)
    to_sink, _, _ = global_aligment(v[::-1], w_to_sink[::-1])
    middle_column1 = np.array([node[-1] for node in from_source])
    middle_column2 = np.array([node[-1] for node in to_sink][::-1])
    score = middle_column1 + middle_column2
    x1 = np.argmax(score)
    y1 = int(len(w) / 2)
    if max(score) == score[np.argmax(score) + 1]:
        x2 = x1 + 1
        y2 = y1
    if max(score) == to_sink[np.argmax(score[::-1])][y1 - 1]:
        x2 = x1
        y2 = y1 + 1
    else:
        x2 = x1 + 1
        y2 = y1 + 1

    return (x1, y1), (x2, y2)


def make_grid(a, b, c):
    grid = []
    for i in range(len(a)):
        for j in range(len(b)):
            for k in range(len(c)):
                grid.append((i, j, k))
    return grid


def match_score_3d(c1, c2, c3):
    if c1 == c2 == c3:
        return 1
    else:
        return 0


def multiple_aligment(x, y, z):
    n = len(x)
    m = len(y)
    l = len(z)
    backtrack = make_grid(x, y, z)
    matrix = dict((el, 0) for el in backtrack)
    true_backtrack = dict((el, '') for el in backtrack)

    for a in range(1, n):
        for b in range(1, m):
            for c in range(1, l):
                matrix[(a, b, c)] = max(
                    matrix[(a - 1, b, c)],
                    matrix[(a, b - 1, c)],
                    matrix[(a, b, c - 1)],
                    matrix[(a - 1, b - 1, c)],
                    matrix[(a - 1, b, c - 1)],
                    matrix[(a, b - 1, c - 1)],
                    matrix[(a - 1, b - 1, c - 1)] + match_score_3d(x[a - 1], y[b - 1], z[c - 1]))

                if matrix[(a, b, c)] == matrix[(a - 1, b, c)]:
                    true_backtrack[(a, b, c)] = 1
                elif matrix[(a, b, c)] == matrix[(a, b - 1, c)]:
                    true_backtrack[(a, b, c)] = 2
                elif matrix[(a, b, c)] == matrix[(a, b, c - 1)]:
                    true_backtrack[(a, b, c)] = 3
                elif matrix[(a, b, c)] == matrix[(a - 1, b - 1, c)]:
                    true_backtrack[(a, b, c)] = 4
                elif matrix[(a, b, c)] == matrix[(a - 1, b, c - 1)]:
                    true_backtrack[(a, b, c)] = 5
                elif matrix[(a, b, c)] == matrix[(a, b - 1, c - 1)]:
                    true_backtrack[(a, b, c)] = 6
                else:
                    true_backtrack[(a, b, c)] = 7
    for key in true_backtrack.keys():
        if 0 in key:
            true_backtrack[key] = 1
    x_p = ''
    y_p = ''
    z_p = ''
    i = n - 1
    j = m - 1
    k = l - 1
    while i != 0 and j != 0 and k != 0:
        if true_backtrack[(i, j, k)] == 1:
            x_p += x[i - 1]
            y_p += '-'
            z_p += '-'
            i = i - 1
        elif true_backtrack[(i, j, k)] == 2:
            x_p += '-'
            y_p += y[j - 1]
            z_p += '-'
            j = j - 1
        elif true_backtrack[(i, j, k)] == 3:
            x_p += '-'
            y_p += '-'
            z_p += z[k - 1]
            k = k - 1
        elif true_backtrack[(i, j, k)] == 4:
            x_p += x[i - 1]
            y_p += '-'
            z_p += z[k - 1]
            i = i - 1
            k = k - 1
        elif true_backtrack[(i, j, k)] == 5:
            x_p += x[i - 1]
            y_p += '-'
            z_p += z[k - 1]
            i = i - 1
            k = k - 1
        elif true_backtrack[(i, j, k)] == 6:
            x_p += '-'
            y_p += y[j - 1]
            z_p += z[k - 1]
            j = j - 1
            k = k - 1
        else:
            x_p += x[i - 1]
            y_p += y[j - 1]
            z_p += z[k - 1]
            i = i - 1
            j = j - 1
            k = k - 1
    x_p = x_p
    y_p = y_p
    z_p = z_p

    # return s,v_p,w_p
    print(matrix)
    print(x_p)
    print(y_p)
    print(z_p)
