"""
This file contains a graph class, which is used to represent social connections
among social dilemma games. Each of the N nodes (players) has K edges
(connections) before a rewiring process with probability P that each edge may
rewire randomly. If K == N - 1, it is a well-mixed graph and P doesn't matter.
After the graph is set, edges can be drawn to represent a game between two
players, for all players. It is likely that a player is drawn twice in one
selection so that all players are drawn at least once.

Author: Liyao Zhu  liyaoz@student.unimelb.edu.au
Date:   Apr. 2019
"""

import numpy as np


class Graph:
    def __init__(self, N, K, P):
        self.N = N     # Number of players
        self.K = K     # Number of edges/connections each player has originally
        self.P = P     # Rewiring probability
        self.edges = []
        self.selectedNodes = {}

        if K == N - 1:
            """Well-mixed graph, no rewiring"""
            for i in range(N):
                for j in range(i + 1, N):
                    self.edges.append((i, j))

        elif K < N - 1:
            assert K % 2 == 0
            k_half = int(K/2)

            """Create the original graph (equal to p = 0)"""
            for i in range(N):
                for j in range(1, k_half + 1):
                    self.edges.append((i, (i + j) % N))

            """Randomly rewire each edge with prob p, start from distance 1"""
            for j in range(1, k_half + 1):
                for i in range(N):
                    if P > np.random.uniform(0, 1):
                        new_set = [v for v in range(N) if v != i and (i, v) not
                                   in self.edges and (v, i) not in self.edges]
                        if len(new_set) > 0:
                            new = np.random.choice(new_set)
                            self.edges.append((i, new))
                            old = (i + j) % self.N
                            self.edges.remove((i, old))
                            # print("Rewiring (", i, old, ") to: (", i, new)

        else:
            print("ERROR: Illegal K or N value.")
            exit(3)

    def select(self):
        """
        Randomly select edges from the graph, so that each player is drawn at
        least once.
        :return: A list of tuples, containing players' index
        """
        edges = self.edges
        nodes = list(range(self.N))
        select = []
        selectedNodes = {i: 0 for i in range(self.N)}

        if self.K == self.N - 1:     #Well-mixed graph
            permutation = np.random.permutation(self.N)
            selectedNodes = {i: 1 for i in range(self.N)}
            if self.N % 2 == 1:
                extraNode = np.random.randint(0, self.N)
                while extraNode == permutation[self.N - 1]:
                    extraNode = np.random.randint(0, self.N)
                np.append(permutation, extraNode)
                selectedNodes[extraNode] += 1
            select = permutation.reshape((int(len(permutation)/2), 2))
        else:
            while edges:       # Loop when edges is not empty
                i, j = edges[np.random.randint(0, len(edges))]
                # print("selected nodes:", i, j)
                select.append((i, j))
                nodes.remove(i)
                nodes.remove(j)
                selectedNodes[i] += 1
                selectedNodes[j] += 1
                # print("Remaining nodes:", nodes)
                edges = [(a, b) for (a, b) in edges if (a != i) and (a != j)
                         and (b != i) and (b != j)]
                # print("after removal", edges)

            while nodes:
                v = nodes.pop(np.random.randint(0, len(nodes)))
                v_edges = [(i, j) for (i, j) in self.edges if i == v or j == v]
                i, j = v_edges[np.random.randint(len(v_edges))]
                select.append((i, j))
                selectedNodes[i] += 1
                selectedNodes[j] += 1

            # print("Number of each nodes selected:", selectedNodes)
        self.selectedNodes = selectedNodes
        return select


    def getNodesNumber(self):
        """
        :return: A dictionary specify how many times each player are drawn from
        the last select()
        """
        return self.selectedNodes

    def getEdgeList(self):
        return self.edges