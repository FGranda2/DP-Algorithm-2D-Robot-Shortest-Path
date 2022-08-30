"""
OTTO Coding Challenge
@author: Francisco Granda
"""

import numpy as np
import math
from io import StringIO
import time
import sys


def cost2go(wp_start, wp_goal):
    # Compute time from start position to goal position
    vel = 2
    x1 = wp_start[0]
    x2 = wp_goal[0]
    y1 = wp_start[1]
    y2 = wp_goal[1]
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance/vel


def main():

    # Read input files for test cases
    data = sys.stdin.readlines()
    # t = time.time()

    # Obtain waypoint arrays
    waypoints = [np.loadtxt(StringIO(line)) for line in data]

    # Organize test cases
    split = np.array([num for num in waypoints if num.size == 1], dtype=int)
    wp = np.array([num for num in waypoints if num.size == 3], dtype=int)
    cases = []
    idx = 0
    for i in range(len(split[:-1])):
        cases.append(wp[idx:idx + split[i], :])
        idx = idx + split[i]

    # Iteration over available cases
    results = []
    for n_case in range(len(cases)):
        # Select case
        sel = n_case
        test_case = cases[sel]

        # Dynamic Programming Algorithm
        start = [0, 0, 0]
        goal = [100, 100, 0]
        test_case = np.vstack([start, test_case])
        nodes = test_case
        moves = 1
        stop_time = 10
        curr_cost = []

        # Backwards iteration
        while True:
            if moves == 1:
                for i in range(len(nodes)):
                    penalty = sum(nodes[i+1:, 2])
                    curr_cost.append(cost2go(nodes[i], goal) + stop_time + penalty)

            else:
                for i in range(len(curr_cost)-1):
                    # We can only move forward in order so subsets created
                    subsets = nodes[i+1:]
                    current_node = nodes[i]
                    best_cost = curr_cost[i]

                    for j in range(len(subsets)):
                        # Compute cost to move from each i to j in subset
                        c2g = cost2go(current_node, subsets[j]) + stop_time
                        # Need to add penalties from skipping
                        penalty = sum(subsets[:j, 2])
                        partial_cost = c2g + penalty + curr_cost[i+j+1]
                        # Minimization
                        if partial_cost < best_cost:
                            best_cost = partial_cost

                    curr_cost[i] = best_cost
            moves = moves + 1

            if moves == len(nodes)+1:
                break

        final_cost = np.round_(curr_cost[0], decimals=3)
        results.append(final_cost)

    # Processing time
    # elapsed = time.time() - t
    # print('Processing Time:', np.round_(elapsed, decimals=3))

    # Output results
    for line in results:
        sys.stdout.write(str(line))
        sys.stdout.write('\n')


if __name__ == '__main__':
    main()

