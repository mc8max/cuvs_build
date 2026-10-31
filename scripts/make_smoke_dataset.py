#!/usr/bin/env python3
"""Generate a small deterministic ANN smoke workload using only the standard library."""

import argparse
import hashlib
import heapq
import json
import random
import struct
from pathlib import Path


def write_matrix(path, rows, kind):
    """cuVS .fbin/.ibin: little-endian uint32 dimensions, then row-major values."""
    columns = len(rows[0])
    row_format = struct.Struct("<" + kind * columns)
    with path.open("wb") as output:
        output.write(struct.pack("<II", len(rows), columns))
        for row in rows:
            output.write(row_format.pack(*row))


def generate(output, algorithm, seed):
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    row_count, query_count, dimensions, k = 2048, 128, 32, 10
    # Binary fractions keep values exact in float32. Squared distances can also be
    # represented exactly at these sizes, making the reference neighbors stable.
    base = [[rng.randrange(-128, 128) for _ in range(dimensions)] for _ in range(row_count)]
    queries, neighbors = [], []
    while len(queries) < query_count:
        query = [rng.randrange(-128, 128) for _ in range(dimensions)]
        nearest = heapq.nsmallest(
            k + 1,
            ((sum((a - b) ** 2 for a, b in zip(query, row)), index)
             for index, row in enumerate(base)),
        )
        # Ties across the k-th neighbor would make two correct rankings disagree.
        if nearest[k - 1][0] == nearest[k][0]:
            continue
        queries.append(query)
        neighbors.append([index for _, index in nearest[:k]])

    write_matrix(output / "base.fbin", [[value / 128 for value in row] for row in base], "f")
    write_matrix(output / "query.fbin", [[value / 128 for value in row] for row in queries], "f")
    write_matrix(output / "groundtruth.ibin", neighbors, "I")

    parameters = {
        "cuvs_brute_force": ({}, {}),
        "cuvs_cagra": (
            {"graph_degree": 32, "intermediate_graph_degree": 64},
            {"itopk": 64, "search_width": 1},
        ),
        "cuvs_ivf_flat": ({"nlist": 32}, {"nprobe": 8}),
    }
    build_parameters, search_parameters = parameters[algorithm]
    config = {
        "dataset": {
            "name": "deterministic-smoke-2048x32",
            "base_file": "base.fbin",
            "query_file": "query.fbin",
            "groundtruth_neighbors_file": "groundtruth.ibin",
            "distance": "euclidean",
            "dtype": "float",
        },
        "search_basic_param": {"batch_size": 128, "k": k},
        "index": [{
            "name": algorithm + "-smoke",
            "algo": algorithm,
            "build_param": build_parameters,
            "file": algorithm + ".idx",
            "search_params": [search_parameters],
        }],
    }
    (output / "benchmark.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "purpose": "Development smoke benchmark; not a representative performance dataset.",
        "seed": seed,
        "base_rows": row_count,
        "query_rows": query_count,
        "dimensions": dimensions,
        "k": k,
        "algorithm": algorithm,
        "sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("base.fbin", "query.fbin", "groundtruth.ibin", "benchmark.json")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument(
        "--algorithm", default="cuvs_brute_force",
        choices=("cuvs_brute_force", "cuvs_cagra", "cuvs_ivf_flat"),
    )
    arguments = parser.parse_args()
    generate(arguments.output, arguments.algorithm, arguments.seed)


if __name__ == "__main__":
    main()
