# # SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
# #
# # SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def modularity_matrix(adjacency_matrix, degree_vector, total_degree):
    g_size = adjacency_matrix.shape[0]
    inv_total_degree = 1 / total_degree

    # Pre-calculate the term for subtraction which is used for both diagonal and off-diagonal elements
    subtraction_term = inv_total_degree * np.outer(degree_vector, degree_vector)

    B_hat_g = adjacency_matrix - subtraction_term

    # Adjust diagonal elements
    B_g_rowsum = adjacency_matrix.sum(axis=1)
    np.fill_diagonal(
        B_hat_g, B_hat_g.diagonal() - inv_total_degree * (B_g_rowsum * degree_vector)
    )

    return B_hat_g


def modularity_spectral_optimization(adjacency_matrix, return_communities=False):
    n = len(adjacency_matrix)
    total_degree = np.sum(adjacency_matrix)
    degree_vector = np.sum(adjacency_matrix, axis=1)
    subgraphs = [np.arange(n)]

    Q_g_dict = {}
    partition = np.zeros(n, dtype=int)

    i = 0
    while subgraphs:
        current_subgraph = subgraphs.pop(0)
        B = modularity_matrix(
            adjacency_matrix[current_subgraph][:, current_subgraph],
            degree_vector[current_subgraph],
            total_degree,
        )
        if np.isinf(B).any() or np.isnan(B).any():
            print("Nan or inf detected in B")
            B = np.nan_to_num(B, copy=True, nan=0.0, posinf=n, neginf=-n)

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        leading_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

        positive_indices = np.where(leading_eigenvector >= 0)[0]
        negative_indices = np.where(leading_eigenvector < 0)[0]

        if len(positive_indices) == 0 or len(negative_indices) == 0:
            continue

        partition[positive_indices] = 2**i
        partition[negative_indices] = (2**i) + 1
        positive_subgraph = current_subgraph[positive_indices]
        negative_subgraph = current_subgraph[negative_indices]

        subgraphs.append(positive_subgraph)
        subgraphs.append(negative_subgraph)

        B_pos = modularity_matrix(
            adjacency_matrix[positive_indices][:, positive_indices],
            degree_vector[positive_indices],
            total_degree,
        )
        B_neg = modularity_matrix(
            adjacency_matrix[negative_indices][:, negative_indices],
            degree_vector[negative_indices],
            total_degree,
        )

        Qg = (np.sum(B_pos) + np.sum(B_neg)) / total_degree

        Q_g_dict[(i, Qg)] = (positive_subgraph, negative_subgraph)
        i += 1

        if Qg <= 0:
            # Create a new list without the positive and negative subgraphs
            subgraphs = [
                graph
                for graph in subgraphs
                if not np.array_equal(graph, positive_subgraph)
                and not np.array_equal(graph, negative_subgraph)
            ]

    partition = np.unique(partition, return_inverse=True)[1]

    if return_communities:
        communities = []
        for el in np.unique(partition):
            communities.append(list(np.where(partition == el)[0]))

        return communities, partition

    return partition


##########################################################################


def modularity_spectral_threshold(
    adjacency_matrix, threshold=30, return_communities=True
):
    partition = modularity_spectral_optimization(adjacency_matrix)

    def iterative_community_detection(initial_indices):
        community_list = [initial_indices]

        while community_list:
            subgraph_indices = community_list.pop()

            if len(subgraph_indices) <= threshold:
                yield subgraph_indices
                continue

            sub_adj_matrix = adjacency_matrix[
                np.ix_(subgraph_indices, subgraph_indices)
            ]
            sub_degrees = np.sum(sub_adj_matrix, axis=1)
            total_sub_degree = np.sum(sub_degrees)

            B_hat_g = modularity_matrix(sub_adj_matrix, sub_degrees, total_sub_degree)

            eigenvalues, eigenvectors = np.linalg.eigh(B_hat_g)
            leading_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

            positive_indices = np.where(leading_eigenvector >= 0)[0]
            negative_indices = np.where(leading_eigenvector < 0)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                yield subgraph_indices
            else:
                positive_subgraph = [subgraph_indices[idx] for idx in positive_indices]
                negative_subgraph = [subgraph_indices[idx] for idx in negative_indices]
                community_list.append(positive_subgraph)
                community_list.append(negative_subgraph)

    communities = list(
        iterative_community_detection(list(range(len(adjacency_matrix))))
    )

    # Create a partition map with community labels
    partition_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition_map[node] = i

    partition = np.array([partition_map[node] for node in range(len(adjacency_matrix))])

    if return_communities:
        return communities, partition
    else:
        return partition
