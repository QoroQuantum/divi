# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from divi.qprog.problems import GraphPartitioningConfig, QUBOPartitioningConfig

CONFIGS = [
    pytest.param(GraphPartitioningConfig, "max_n_nodes_per_cluster", id="graph"),
    pytest.param(QUBOPartitioningConfig, "max_n_variables_per_cluster", id="qubo"),
]


@pytest.mark.parametrize("cls, size_field", CONFIGS)
def test_size_limits_are_validated(cls, size_field):
    with pytest.raises(ValueError, match="At least one constraint"):
        cls()
    with pytest.raises(ValueError, match=f"'{size_field}' must be a positive"):
        cls(**{size_field: 0})
    with pytest.raises(ValueError, match="'minimum_n_clusters' must be a positive"):
        cls(minimum_n_clusters=0)


@pytest.mark.parametrize("cls, size_field", CONFIGS)
def test_max_cluster_size_reads_the_domain_field(cls, size_field):
    config = cls(**{size_field: 7}, minimum_n_clusters=2)
    assert config.max_cluster_size == getattr(config, size_field) == 7
    assert config.minimum_n_clusters == 2


@pytest.mark.parametrize("cls, size_field", CONFIGS)
def test_configs_are_frozen(cls, size_field):
    config = cls(minimum_n_clusters=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.minimum_n_clusters = 3


def test_qubo_config_rejects_unknown_method():
    with pytest.raises(ValueError, match="modularity.*spectral"):
        QUBOPartitioningConfig(minimum_n_clusters=2, method="bogus")


def test_graph_config_positional_order_is_unchanged():
    config = GraphPartitioningConfig(8, 3, "metis")
    assert (config.max_n_nodes_per_cluster, config.minimum_n_clusters) == (8, 3)
    assert config.partitioning_algorithm == "metis"
