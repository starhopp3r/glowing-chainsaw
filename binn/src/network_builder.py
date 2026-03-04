"""
Reactome + STRING constrained network construction for the BINN.

Converts the biological mapping (gene → UniProt → Reactome + STRING PPI)
into a layered directed graph and binary connectivity masks used as
weight sparsity patterns in the neural network.

Layer structure (information flows bottom → top)
────────────────────────────────────────────────
  Layer 0          : gene nodes (input)
  Layer 1 .. L     : Reactome pathway nodes (specific → general)
  Layer L+1        : synthetic OUTPUT node

Edge direction in the graph follows information flow:
  gene → leaf_pathway → ... → root_pathway → OUTPUT

Reactome hierarchy edges (parent→child = general→specific) are REVERSED
so they run child→parent (specific→general, i.e. toward output).
"""
from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict, deque

import networkx as nx
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _longest_path_depth(dag: nx.DiGraph) -> dict[str, int]:
    """
    Longest-path depth from any root (nodes with in_degree 0).
    Root nodes get depth 0; depth increases toward leaves.
    Uses topological sort — O(V+E).
    Falls back to BFS if a cycle is detected (shouldn't happen for Reactome).
    """
    depth: dict[str, int] = {}
    try:
        for node in nx.topological_sort(dag):
            preds = list(dag.predecessors(node))
            depth[node] = 0 if not preds else max(depth.get(p, 0) for p in preds) + 1
    except nx.NetworkXUnfeasible:
        log.warning("Cycle detected in sub-hierarchy; falling back to BFS depth.")
        for root in (n for n in dag.nodes() if dag.in_degree(n) == 0):
            for node, d in nx.single_source_shortest_path_length(dag, root).items():
                depth[node] = max(depth.get(node, 0), d)
    return depth


# ── Build fold graph ──────────────────────────────────────────────────────────

def build_fold_graph(fold_genes: list[str], bio_map: dict) -> nx.DiGraph:
    """
    Build the directed constraint graph for a given fold's gene list.

    Parameters
    ----------
    fold_genes : gene symbols present in this fold (after MAD filtering)
    bio_map    : output of ``build_full_biological_map()``

    Returns
    -------
    nx.DiGraph
        Node attributes: layer (int), node_type (str), name (str), is_copy (bool)
        Edge attributes: via_ppi (bool), is_copy_edge (bool)

    Notes
    -----
    Genes with no Reactome mapping are excluded from the graph but tracked
    in the ``excluded_genes`` graph attribute.  Baseline models (SVM, RF, …)
    still use ALL genes.

    PPI edges between fold genes are expressed as additional gene→pathway
    connections: if gene_i has a STRING PPI edge with gene_k, then gene_i
    also connects to every Reactome pathway that gene_k connects to.  This
    enriches the receptive field of the first hidden layer without adding
    extra layers.
    """
    hierarchy: nx.DiGraph = bio_map["reactome_hierarchy"]
    gene_to_reactome: dict = bio_map["gene_to_reactome"]
    reactome_names: dict = bio_map["reactome_names"]
    ppi_edges_list: list = bio_map["ppi_edges"]

    fold_gene_set = set(fold_genes)
    hierarchy_depth = _longest_path_depth(hierarchy)
    mapping_mode = str(getattr(config, "GENE_PATHWAY_MAPPING_MODE", "deepest")).lower()

    # ── 1. Filter gene → pathway map to fold genes ────────────────────
    fold_g2r: dict[str, set[str]] = {}
    dropped_nonhier = 0
    for g in fold_genes:
        rids = gene_to_reactome.get(g, set())
        if not rids:
            continue
        valid_rids = {rid for rid in rids if hierarchy.has_node(rid)}
        dropped_nonhier += len(rids) - len(valid_rids)
        if mapping_mode == "deepest" and valid_rids:
            deepest = max(hierarchy_depth.get(rid, 0) for rid in valid_rids)
            valid_rids = {rid for rid in valid_rids if hierarchy_depth.get(rid, 0) == deepest}
        if valid_rids:
            fold_g2r[g] = valid_rids
    genes_with_reactome = sorted(fold_g2r)
    genes_no_reactome = sorted(fold_gene_set - set(genes_with_reactome))

    log.info(
        f"Fold genes: {len(genes_with_reactome)} with Reactome, "
        f"{len(genes_no_reactome)} excluded (no Reactome mapping)."
    )
    if dropped_nonhier:
        log.info(
            f"Dropped {dropped_nonhier} gene→Reactome mappings not present "
            "in the human Reactome hierarchy graph."
        )
    if not genes_with_reactome:
        raise ValueError(
            "No fold genes have Reactome pathway mappings. "
            "Check biological_mapping output."
        )

    # ── 2. Collect all relevant pathways (gene mappings + all ancestors) ──
    mapped_pathways: set[str] = {r for rids in fold_g2r.values() for r in rids}
    relevant_pathways: set[str] = set(mapped_pathways)
    frontier = set(mapped_pathways)
    while frontier:
        next_frontier: set[str] = set()
        for pid in frontier:
            for parent in hierarchy.predecessors(pid):   # parent = more general
                if parent not in relevant_pathways:
                    relevant_pathways.add(parent)
                    next_frontier.add(parent)
        frontier = next_frontier

    # ── 3. Layer assignment ────────────────────────────────────────────
    # Induced sub-hierarchy preserves edge direction: parent → child.
    sub_hier = hierarchy.subgraph(relevant_pathways).copy()
    layering_mode = str(
        getattr(config, "PATHWAY_LAYERING_MODE", "distance_from_mapped")
    ).lower()

    pathway_layer: dict[str, int] = {}
    if layering_mode == "distance_from_mapped":
        # In flow graph (child→parent), mapped pathways are distance 0 and all
        # mapped pathways therefore sit directly above genes (layer 1).
        flow = sub_hier.reverse(copy=True)
        dist: dict[str, int] = {n: -1 for n in flow.nodes()}
        q: deque[str] = deque()
        for src in mapped_pathways:
            if src in dist:
                dist[src] = 0
                q.append(src)
        while q:
            node = q.popleft()
            for parent in flow.successors(node):
                if dist[parent] == -1 or dist[parent] > dist[node] + 1:
                    dist[parent] = dist[node] + 1
                    q.append(parent)

        for pid in relevant_pathways:
            d = dist.get(pid, -1)
            pathway_layer[pid] = 1 if d < 0 else d + 1
    else:
        depth_map = _longest_path_depth(sub_hier)
        for pid in relevant_pathways:
            depth_map.setdefault(pid, 0)
        max_depth = max(depth_map.values(), default=0)
        for pid in relevant_pathways:
            pathway_layer[pid] = max_depth - depth_map.get(pid, 0) + 1

    max_pathway_layer = max(pathway_layer.values(), default=1)
    output_layer = max_pathway_layer + 1
    max_hier_depth = max_pathway_layer - 1

    # ── 4. Populate the directed graph ────────────────────────────────
    G = nx.DiGraph()

    # Gene nodes — layer 0
    for gene in genes_with_reactome:
        G.add_node(gene, layer=0, node_type="gene", name=gene, is_copy=False)

    # Pathway nodes
    for pid in relevant_pathways:
        G.add_node(
            pid,
            layer=pathway_layer[pid],
            node_type="pathway",
            name=reactome_names.get(pid, pid),
            is_copy=False,
        )

    # Output node
    G.add_node(
        "OUTPUT",
        layer=output_layer,
        node_type="output",
        name="Output",
        is_copy=False,
    )

    # Gene → directly-mapped pathway edges
    for gene, pathways in fold_g2r.items():
        for pid in pathways:
            if pid in G:
                G.add_edge(gene, pid, via_ppi=False, is_copy_edge=False)

    # Pathway → parent-pathway edges (REVERSED Reactome hierarchy).
    # Keep only edges that move to a strictly higher layer.
    dropped_nonascending = 0
    for parent, child in sub_hier.edges():
        if parent in G and child in G and pathway_layer[parent] > pathway_layer[child]:
            G.add_edge(child, parent, via_ppi=False, is_copy_edge=False)
        else:
            dropped_nonascending += 1

    if dropped_nonascending:
        log.info(
            f"Dropped {dropped_nonascending} non-ascending pathway hierarchy edges "
            f"under layering mode '{layering_mode}'."
        )

    # Highest-layer pathways → OUTPUT
    for pid in (n for n in relevant_pathways if pathway_layer[n] == max_pathway_layer):
        if pid in G:
            G.add_edge(pid, "OUTPUT", via_ppi=False, is_copy_edge=False)

    # Optional bridge chain: route lower-layer pathway sinks upward so every
    # pathway can contribute to OUTPUT with only adjacent-layer edges.
    bridge_added = 0
    if (
        layering_mode == "distance_from_mapped"
        and getattr(config, "ENABLE_SINK_BRIDGES", True)
    ):
        sink_pathways = [
            n
            for n in relevant_pathways
            if G.has_node(n)
            and G.nodes[n]["node_type"] == "pathway"
            and G.out_degree(n) == 0
        ]
        if sink_pathways:
            bridge_nodes = []
            for lyr in range(2, output_layer):
                bridge_id = f"__bridge__L{lyr}"
                if not G.has_node(bridge_id):
                    G.add_node(
                        bridge_id,
                        layer=lyr,
                        node_type="bridge",
                        name=f"Bridge L{lyr}",
                        is_copy=True,
                    )
                    bridge_nodes.append(bridge_id)

            for lyr in range(2, output_layer - 1):
                src = f"__bridge__L{lyr}"
                dst = f"__bridge__L{lyr + 1}"
                if G.has_node(src) and G.has_node(dst) and not G.has_edge(src, dst):
                    G.add_edge(src, dst, via_ppi=False, is_copy_edge=True)
                    bridge_added += 1

            last_bridge = f"__bridge__L{output_layer - 1}"
            if G.has_node(last_bridge) and not G.has_edge(last_bridge, "OUTPUT"):
                G.add_edge(last_bridge, "OUTPUT", via_ppi=False, is_copy_edge=True)
                bridge_added += 1

            sink_edges_added = 0
            for pid in sink_pathways:
                lyr = G.nodes[pid]["layer"]
                if lyr == output_layer - 1:
                    if not G.has_edge(pid, "OUTPUT"):
                        G.add_edge(pid, "OUTPUT", via_ppi=False, is_copy_edge=True)
                        sink_edges_added += 1
                else:
                    dst = f"__bridge__L{lyr + 1}"
                    if G.has_node(dst) and not G.has_edge(pid, dst):
                        G.add_edge(pid, dst, via_ppi=False, is_copy_edge=True)
                        sink_edges_added += 1

            if bridge_nodes or sink_edges_added:
                log.info(
                    f"Sink bridges: {len(bridge_nodes)} bridge nodes, "
                    f"{sink_edges_added + bridge_added} bridge edges added."
                )

    # ── 5. PPI-enriched gene → pathway edges ─────────────────────────
    # gene_i –PPI– gene_k  →  add gene_i → (every pathway of gene_k)
    # This widens the receptive field at layer 0→1 without adding layers.
    ppi_added = 0
    if getattr(config, "USE_PPI_ENRICHMENT", True):
        ppi_neighbors: dict[str, set[str]] = defaultdict(set)
        for g1, g2, _score in ppi_edges_list:
            if g1 in fold_gene_set and g2 in fold_gene_set:
                ppi_neighbors[g1].add(g2)
                ppi_neighbors[g2].add(g1)

        for gene in genes_with_reactome:
            for neighbor in ppi_neighbors.get(gene, set()):
                for pid in fold_g2r.get(neighbor, set()):
                    if pid in G and not G.has_edge(gene, pid):
                        G.add_edge(gene, pid, via_ppi=True, is_copy_edge=False)
                        ppi_added += 1
    else:
        log.info("PPI enrichment disabled via config.USE_PPI_ENRICHMENT=False.")

    # Store excluded genes as a graph attribute for downstream reporting
    G.graph["excluded_genes"] = genes_no_reactome
    G.graph["max_hier_depth"] = max_hier_depth
    G.graph["pathway_layering_mode"] = layering_mode

    log.info(
        f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"({ppi_added} via PPI).  Pathway layers: 1–{max_hier_depth + 1}.  "
        f"Output layer: {output_layer}."
    )
    return G


def pad_graph_to_uniform_depth(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Insert identity copy nodes so every gene→OUTPUT path has the same length.

    For each edge u → v where v.layer − u.layer > 1, the direct edge is
    replaced by a chain through (gap − 1) copy nodes:

        u  →  copy_L(u+1)  →  …  →  copy_L(v-1)  →  v

    Each copy node is uniquely named after its (source, target) pair and
    carries ``is_copy=True``.  In the BINN, copy nodes receive identity-
    initialised weight matrices with a single nonzero connection, so they
    relay their input unchanged.

    Two classes of gap arise:
    - Gene (layer 0) connected to a high-level (root) pathway.
    - Pathway-to-pathway edges that skip layers (25 in the full Reactome DAG).
    """
    layer_nodes = get_layer_node_names(graph)
    layer_sizes = [len(nodes) for nodes in layer_nodes]
    padded_sizes = list(layer_sizes)
    estimated_copy_nodes = 0
    estimated_edges_to_pad = 0
    for u, v in graph.edges():
        l_u = graph.nodes[u]["layer"]
        l_v = graph.nodes[v]["layer"]
        gap = l_v - l_u
        if gap > 1:
            estimated_edges_to_pad += 1
            estimated_copy_nodes += gap - 1
            for lyr in range(l_u + 1, l_v):
                padded_sizes[lyr] += 1

    est_dense_params = sum(a * b for a, b in zip(padded_sizes[:-1], padded_sizes[1:]))
    dense_limit = int(getattr(config, "MAX_DENSE_PARAMS_AFTER_PADDING", 50_000_000))
    if est_dense_params > dense_limit:
        log.warning(
            "Skipping depth padding: estimated dense params after padding "
            f"({est_dense_params:,}) exceed limit ({dense_limit:,})."
        )
        log.warning(
            "Some skip-layer edges will be dropped when building consecutive "
            "connectivity matrices."
        )
        graph.graph["padding_skipped"] = True
        graph.graph["estimated_padded_dense_params"] = est_dense_params
        graph.graph["estimated_copy_nodes"] = estimated_copy_nodes
        graph.graph["estimated_edges_to_pad"] = estimated_edges_to_pad
        return graph

    G = graph.copy()

    # Snapshot edges before modifying the graph
    edges_to_pad = [
        (u, v, dict(edata))
        for u, v, edata in G.edges(data=True)
        if G.nodes[v]["layer"] - G.nodes[u]["layer"] > 1
    ]

    copy_count = 0
    for u, v, edata in edges_to_pad:
        if not G.has_edge(u, v):
            continue        # safety: already removed in a previous iteration
        l_u = G.nodes[u]["layer"]
        l_v = G.nodes[v]["layer"]

        G.remove_edge(u, v)
        prev = u

        for lyr in range(l_u + 1, l_v):
            copy_id = f"__copy__{u}__{v}__L{lyr}"
            G.add_node(
                copy_id,
                layer=lyr,
                node_type="copy",
                name=f"↑Copy(L{lyr})",
                is_copy=True,
            )
            G.add_edge(prev, copy_id, via_ppi=False, is_copy_edge=True)
            prev = copy_id
            copy_count += 1

        G.add_edge(prev, v, **{**edata, "is_copy_edge": True})

    log.info(
        f"Depth padding: {len(edges_to_pad)} edges padded, "
        f"{copy_count} copy nodes inserted."
    )
    return G


# ── Connectivity matrices ─────────────────────────────────────────────────────

def get_layer_node_names(graph: nx.DiGraph) -> list[list[str]]:
    """
    Return node names grouped by layer index, each group in sorted order.

    Returns
    -------
    list[list[str]]
        ``result[l]`` = sorted node IDs for layer ``l``.
    """
    layer_map: dict[int, list[str]] = defaultdict(list)
    for node, data in graph.nodes(data=True):
        layer_map[data["layer"]].append(node)

    max_layer = max(layer_map) if layer_map else 0
    return [sorted(layer_map.get(l, [])) for l in range(max_layer + 1)]


def build_connectivity_matrices(graph: nx.DiGraph) -> list[torch.Tensor]:
    """
    Build binary connectivity tensors C^(ℓ) for every consecutive layer pair.

    C^(ℓ)  ∈  {0, 1}^{n_ℓ × n_{ℓ+1}}
    C^(ℓ)[i, j] = 1  iff  node i (layer ℓ) has an edge to node j (layer ℓ+1).

    PPI-enriched gene→pathway edges (via_ppi=True) are already present in the
    graph from ``build_fold_graph``, so no special handling is needed here.

    Matrices are created on CPU as float32 tensors.
    Device transfer is handled later by the training stack.

    Returns
    -------
    list[torch.Tensor]  —  length = number_of_layers − 1
    """
    layer_nodes = get_layer_node_names(graph)
    matrices: list[torch.Tensor] = []

    for l, (src_nodes, tgt_nodes) in enumerate(
        zip(layer_nodes[:-1], layer_nodes[1:])
    ):
        src_idx = {n: i for i, n in enumerate(src_nodes)}
        tgt_idx = {n: i for i, n in enumerate(tgt_nodes)}

        C = torch.zeros(len(src_nodes), len(tgt_nodes), dtype=torch.float32)

        for u, v in graph.edges():
            ul = graph.nodes[u]["layer"]
            vl = graph.nodes[v]["layer"]
            if ul == l and vl == l + 1 and u in src_idx and v in tgt_idx:
                C[src_idx[u], tgt_idx[v]] = 1.0

        matrices.append(C)

    return matrices


# ── Node metadata ─────────────────────────────────────────────────────────────

def build_node_metadata(graph: nx.DiGraph, bio_map: dict) -> dict:
    """
    Build interpretability metadata for every node in the graph.

    Returns
    -------
    dict[node_id, dict]  with keys:
        layer         int
        type          "gene" | "pathway" | "copy" | "output"
        name          human-readable name
        reactome_id   Reactome pathway ID, or None
        uniprot_ids   list of UniProt accessions (genes only), or None
        string_ids    list of STRING IDs (genes only), or None
        n_children    in-degree  (nodes feeding INTO this node)
        n_parents     out-degree (nodes this node feeds INTO)
    """
    gene_to_uniprot = bio_map.get("gene_to_uniprot", {})
    uniprot_to_string = bio_map.get("uniprot_to_string", {})

    metadata: dict = {}
    for node, data in graph.nodes(data=True):
        ntype = data.get("node_type", "unknown")
        uniprot_ids = string_ids = reactome_id = None

        if ntype == "gene":
            uniprot_ids = gene_to_uniprot.get(node, [])
            string_ids = [
                uniprot_to_string[uid]
                for uid in uniprot_ids
                if uid in uniprot_to_string
            ]
        elif ntype == "pathway":
            reactome_id = node

        metadata[node] = {
            "layer": data.get("layer"),
            "type": ntype,
            "name": data.get("name", node),
            "reactome_id": reactome_id,
            "uniprot_ids": uniprot_ids,
            "string_ids": string_ids,
            "n_children": graph.in_degree(node),
            "n_parents": graph.out_degree(node),
        }

    return metadata


# ── Validation ────────────────────────────────────────────────────────────────

def validate_graph(
    graph: nx.DiGraph, connectivity_matrices: list[torch.Tensor]
) -> dict:
    """
    Sanity checks and structural statistics.

    Checks
    ------
    - Every gene node has ≥ 1 outgoing edge.
    - OUTPUT is reachable from every gene node.
    - Connectivity matrix dimensions chain consistently
      (cols of C^(ℓ) = rows of C^(ℓ+1)).

    Prints
    ------
    Network Construction Summary with layer sizes, sparsity, compression ratio.

    Returns
    -------
    dict  —  all statistics collected
    """
    layer_nodes = get_layer_node_names(graph)
    layer_sizes = [len(ln) for ln in layer_nodes]
    n_layers = len(layer_nodes)

    gene_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("node_type") == "gene"
    ]
    isolated = [g for g in gene_nodes if graph.out_degree(g) == 0]
    if isolated:
        log.warning(f"{len(isolated)} gene node(s) have no outgoing edges: {isolated[:5]}")

    output_unreachable: list[str] = []
    if "OUTPUT" in graph:
        output_unreachable = [
            g for g in gene_nodes if not nx.has_path(graph, g, "OUTPUT")
        ]
        if output_unreachable:
            log.warning(
                f"OUTPUT unreachable from {len(output_unreachable)} gene node(s)."
            )

    dim_ok = True
    for l in range(len(connectivity_matrices) - 1):
        if connectivity_matrices[l].shape[1] != connectivity_matrices[l + 1].shape[0]:
            log.error(
                f"Dimension mismatch at boundary {l}/{l+1}: "
                f"cols={connectivity_matrices[l].shape[1]} ≠ "
                f"rows={connectivity_matrices[l+1].shape[0]}"
            )
            dim_ok = False

    edges_per_layer: list[int] = []
    sparsity_per_layer: list[float] = []
    for C in connectivity_matrices:
        nnz = int(C.sum().item())
        total = C.numel()
        edges_per_layer.append(nnz)
        sparsity_per_layer.append(round(1.0 - nnz / total, 4) if total else 1.0)

    total_sparse = sum(edges_per_layer)
    total_dense = sum(a * b for a, b in zip(layer_sizes[:-1], layer_sizes[1:]))
    compression = round(total_dense / total_sparse, 2) if total_sparse else float("inf")

    n_copy = sum(1 for _, d in graph.nodes(data=True) if d.get("is_copy"))
    n_ppi = sum(1 for _, _, d in graph.edges(data=True) if d.get("via_ppi"))

    stats = {
        "n_layers": n_layers,
        "layer_sizes": layer_sizes,
        "edges_per_layer": edges_per_layer,
        "sparsity_per_layer": sparsity_per_layer,
        "total_sparse_params": total_sparse,
        "total_dense_params": total_dense,
        "compression_ratio": compression,
        "copy_nodes": n_copy,
        "ppi_edges_integrated": n_ppi,
        "isolated_genes": len(isolated),
        "output_unreachable": len(output_unreachable),
        "dimension_check_passed": dim_ok,
    }

    print()
    print("Network Construction Summary")
    print("==============================")
    print(f"Number of layers:        {n_layers}")
    print(f"Nodes per layer:         {layer_sizes}")
    print(f"Edges per layer:         {edges_per_layer}")
    print(f"Sparsity per layer:      {sparsity_per_layer}")
    print(f"Total parameters:        {total_sparse:,}  (vs {total_dense:,} dense)")
    print(f"Compression ratio:       {compression:.1f}x")
    print(f"Copy nodes inserted:     {n_copy}")
    print(f"PPI edges integrated:    {n_ppi}")

    return stats


# ── Orchestrator ──────────────────────────────────────────────────────────────

def build_fold_network(fold_genes: list[str], bio_map: dict) -> dict:
    """
    End-to-end network construction for a single CV fold.

    Pipeline
    --------
    build_fold_graph  →  pad_graph_to_uniform_depth
                      →  build_connectivity_matrices
                      →  build_node_metadata
                      →  validate_graph

    Returns
    -------
    dict with keys:
        connectivity_matrices  list[torch.Tensor]
        layer_node_names       list[list[str]]
        node_metadata          dict
        graph                  nx.DiGraph
        stats                  dict
        n_layers               int
        layer_sizes            list[int]
    """
    log.info(f"Building fold network for {len(fold_genes)} genes.")

    graph = build_fold_graph(fold_genes, bio_map)
    graph = pad_graph_to_uniform_depth(graph)

    connectivity_matrices = build_connectivity_matrices(graph)
    layer_node_names = get_layer_node_names(graph)
    node_metadata = build_node_metadata(graph, bio_map)
    stats = validate_graph(graph, connectivity_matrices)

    return {
        "connectivity_matrices": connectivity_matrices,
        "layer_node_names": layer_node_names,
        "node_metadata": node_metadata,
        "graph": graph,
        "stats": stats,
        "n_layers": stats["n_layers"],
        "layer_sizes": stats["layer_sizes"],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    genes_path = config.GENE_LIST_FILE
    if not os.path.exists(genes_path):
        raise FileNotFoundError(
            f"Gene list not found at {genes_path}. Run data_acquisition.py first."
        )
    with open(genes_path) as f:
        gene_list = [ln.strip() for ln in f if ln.strip()]

    bio_map_path = config.BIO_MAP_FILE
    if os.path.exists(bio_map_path):
        log.info(f"Loading cached biological map from {bio_map_path}")
        with open(bio_map_path, "rb") as f:
            bio_map = pickle.load(f)
    else:
        from src.biological_mapping import build_full_biological_map, save_biological_map
        bio_map = build_full_biological_map(gene_list)
        save_biological_map(bio_map, config.DATA_PROCESSED)
        with open(bio_map_path, "wb") as f:
            pickle.dump(bio_map, f)

    fold_net = build_fold_network(gene_list, bio_map)
    print("\nLayer sizes:", fold_net["layer_sizes"])
    print("Matrix shapes:", [tuple(C.shape) for C in fold_net["connectivity_matrices"]])
