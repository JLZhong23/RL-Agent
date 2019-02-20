# -*- coding:utf-8 -*-
#|-----|-----|-----|
#|     |     |     |
#|  0  |  1  |  2  |
#|_____|_____|_____|
#|     |     |     |
#|  3  |  4  |  5  |
#|_____|_____|_____|
# Entering room 0 can get 10 reward, else get -1 reward

import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument('-g', type=float, default=0.8, dest='gamma')
parser.add_argument('-s', type=int, default=1000, dest='step_num')
parser.add_argument('-e', type=float, default=1.0, dest='epsilon')
args = parser.parse_args()

reward = [[0, -1, 0, -1, 0, 0],
          [10, 0, -1, 0, -1, 0],
          [0, -1, 0, 0, 0, -1],
          [10, 0, 0, 0, -1, 0],
          [0, -1, 0, -1, 0, -1],
          [0, 0, -1, 0, -1, 0]]
q_table = [[0 for x in range(6)] for y in range(6)]
step = 0
while step < args.step_num:
    s = random.randint(1, 5)
    while s != 0:
        if random.random() < args.epsilon:
            action_list = [x for x in range(6) if reward[s][x] != 0]
            s2 = random.choice(action_list)
        else:
            s2 = q_table[s].index(max(q_table[s]))
        q_table[s][s2] = reward[s][s2] + args.gamma * max([q_table[s2][y] for y in range(6) if y != s2])
        s = s2
    step += 1
print(q_table)
# 0, 0, 0, 0, 0, 0
# 10, 0, 4.6 0 4.6 0
# 0, 7, 0, 0, 0, 2.68
# 10, 0, 0, 0, 4.6 0
# 0, 7, 0, 7, 0, 2.68
# 0, 0, 4.6, 0, 4.6, 0