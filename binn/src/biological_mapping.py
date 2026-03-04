"""
Biological mapping: Gene → UniProt → Reactome pathways + STRING PPI.
Foundation for the BINN sparse connectivity construction.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
import time

import networkx as nx
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_MYGENE_CACHE_FILE = os.path.join(config.DATA_PROCESSED, "mygene_cache.json")


# ── ID normalisation ──────────────────────────────────────────────────────────

def _clean_reactome_id(raw_id: str) -> str | None:
    """
    Extract canonical Reactome ID (R-HSA-<digits>) from a raw token.

    Examples
    --------
    "R-HSA-9028335 [P62993-1]" -> "R-HSA-9028335"
    "R-HSA-1234567"            -> "R-HSA-1234567"
    """
    m = re.search(r"R-HSA-\d+", str(raw_id))
    return m.group(0) if m else None


# ── MyGene.info helpers ───────────────────────────────────────────────────────

def _load_mygene_cache() -> dict[str, list[str]]:
    if os.path.exists(_MYGENE_CACHE_FILE):
        with open(_MYGENE_CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_mygene_cache(cache: dict[str, list[str]]) -> None:
    os.makedirs(os.path.dirname(_MYGENE_CACHE_FILE), exist_ok=True)
    with open(_MYGENE_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _query_mygene(genes: list[str], cache: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Query MyGene.info REST API in batches of 1000.
    Updates and returns the cache dict with {gene_symbol: [uniprot_ids]}.
    Swiss-Prot (reviewed) IDs are preferred; TrEMBL IDs appended after.
    """
    to_query = [g for g in genes if g not in cache]
    if not to_query:
        return cache

    url = "https://mygene.info/v3/query"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    batch_size = 1000

    for batch_start in range(0, len(to_query), batch_size):
        batch = to_query[batch_start : batch_start + batch_size]
        log.info(
            f"MyGene.info: querying {len(batch)} genes "
            f"(batch {batch_start // batch_size + 1}/"
            f"{(len(to_query) - 1) // batch_size + 1})"
        )
        try:
            resp = requests.post(
                url,
                data={
                    "q": ",".join(batch),
                    "scopes": "symbol",
                    "fields": "uniprot",
                    "species": "human",
                },
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            hits = resp.json()
        except Exception as exc:
            log.warning(f"MyGene.info query failed: {exc}. Marking batch as empty.")
            for g in batch:
                cache.setdefault(g, [])
            continue

        for hit in hits:
            if hit.get("notfound"):
                cache.setdefault(hit.get("query", "").upper(), [])
                continue
            query_gene = hit.get("query", "").upper()
            uniprot = hit.get("uniprot", {})
            ids: list[str] = []
            sp = uniprot.get("Swiss-Prot")
            if sp:
                ids += [sp] if isinstance(sp, str) else list(sp)
            trembl = uniprot.get("TrEMBL")
            if trembl:
                ids += [trembl] if isinstance(trembl, str) else list(trembl)
            cache[query_gene] = ids

        time.sleep(0.1)  # polite rate limiting

    _save_mygene_cache(cache)
    return cache


# ── Gene → UniProt ────────────────────────────────────────────────────────────

def build_gene_to_uniprot(uniprotkb_path: str) -> dict[str, list[str]]:
    """
    Build gene_symbol → [UniProt Entry IDs] from uniprotkb.tsv.

    Primary source
    --------------
    ``Entry Name`` column uses the format ``GENENAME_HUMAN``.
    The part before ``_HUMAN`` is treated as the gene symbol.

    Secondary source
    ----------------
    Short parenthetical abbreviations in the ``Protein names`` column
    (e.g. ``"Tumor protein p53 (TP53)"`` → TP53).  Only tokens that
    look like standard gene symbols (2–10 uppercase alphanumeric chars)
    are accepted.

    Returns
    -------
    dict mapping uppercase gene symbol → list of UniProt accession IDs.
    """
    log.info(f"Parsing {uniprotkb_path} for gene → UniProt mapping.")
    df = pd.read_csv(uniprotkb_path, sep="\t", dtype=str)
    human = df[df["Entry Name"].str.endswith("_HUMAN", na=False)].copy()
    log.info(f"Human entries in uniprotkb.tsv: {len(human):,}")

    gene_to_uniprot: dict[str, list[str]] = {}

    # Primary: GENENAME_HUMAN → gene symbol
    for _, row in human.iterrows():
        gene = str(row["Entry Name"]).replace("_HUMAN", "").upper().strip()
        uid = str(row["Entry"]).strip()
        if gene:
            gene_to_uniprot.setdefault(gene, [])
            if uid not in gene_to_uniprot[gene]:
                gene_to_uniprot[gene].append(uid)

    # Secondary: parenthetical gene abbreviations in Protein names
    # Match tokens like (TP53) or (BRCA1) – 2 to 10 uppercase alphanum chars
    paren_re = re.compile(r"\(([A-Z][A-Z0-9\-]{1,9})\)")
    for _, row in human.iterrows():
        protein_names = str(row.get("Protein names", ""))
        uid = str(row["Entry"]).strip()
        for sym in paren_re.findall(protein_names):
            sym = sym.upper()
            # Skip generic abbreviations that aren't gene symbols
            if sym in {"EC", "SU", "TM", "GP", "CD"}:
                continue
            gene_to_uniprot.setdefault(sym, [])
            if uid not in gene_to_uniprot[sym]:
                gene_to_uniprot[sym].append(uid)

    log.info(f"Gene → UniProt (local): {len(gene_to_uniprot):,} distinct gene symbols.")
    return gene_to_uniprot


# ── UniProt → Reactome ────────────────────────────────────────────────────────

def build_uniprot_to_reactome(uniprot2reactome_path: str) -> dict[str, set[str]]:
    """
    Parse UniProt2Reactome.txt; filter to Homo sapiens rows.

    Returns
    -------
    dict: uniprot_id → {reactome_pathway_ids}
    """
    log.info(f"Parsing {uniprot2reactome_path}")
    df = pd.read_csv(
        uniprot2reactome_path,
        sep="\t",
        header=None,
        names=["uniprot", "reactome_id", "url", "pathway_name", "evidence", "species"],
        dtype=str,
    )
    hs = df[df["species"].str.strip() == "Homo sapiens"]
    result: dict[str, set[str]] = {}
    for _, row in hs.iterrows():
        uid = str(row["uniprot"]).strip()
        rid = _clean_reactome_id(str(row["reactome_id"]).strip())
        if uid and rid:
            result.setdefault(uid, set()).add(rid)
    log.info(
        f"UniProt → Reactome (UniProt2Reactome.txt): "
        f"{len(result):,} proteins, {sum(len(v) for v in result.values()):,} mappings."
    )
    return result


def build_uniprot_to_reactome_from_kb(uniprotkb_path: str) -> dict[str, set[str]]:
    """
    Parse the ``Reactome`` column in uniprotkb.tsv for human entries.
    Values are semicolon-separated Reactome IDs, e.g. ``R-HSA-1236974;R-HSA-1236977;``.

    Returns
    -------
    dict: uniprot_id → {reactome_pathway_ids}
    """
    log.info(f"Parsing Reactome column from {uniprotkb_path}")
    df = pd.read_csv(uniprotkb_path, sep="\t", dtype=str)
    human = df[df["Entry Name"].str.endswith("_HUMAN", na=False)]
    has_reactome = human[human["Reactome"].notna()]

    result: dict[str, set[str]] = {}
    for _, row in has_reactome.iterrows():
        uid = str(row["Entry"]).strip()
        for raw_rid in str(row["Reactome"]).split(";"):
            rid = _clean_reactome_id(raw_rid.strip())
            if rid:
                result.setdefault(uid, set()).add(rid)

    log.info(
        f"UniProt → Reactome (uniprotkb.tsv): "
        f"{len(result):,} proteins, {sum(len(v) for v in result.values()):,} mappings."
    )
    return result


# ── Reactome hierarchy ────────────────────────────────────────────────────────

def load_reactome_hierarchy(
    relations_path: str,
    pathways_path: str,
) -> tuple[nx.DiGraph, dict[str, str]]:
    """
    Build the human Reactome pathway hierarchy as a directed graph.

    Edges go parent → child (top-down).
    Both endpoints must have the ``R-HSA-`` prefix (human pathways only).

    Returns
    -------
    hierarchy : nx.DiGraph
    reactome_names : dict[str, str]
        reactome_id → human-readable pathway name
    """
    log.info(f"Loading pathway names from {pathways_path}")
    pathways_df = pd.read_csv(
        pathways_path,
        sep="\t",
        header=None,
        names=["reactome_id", "name", "species"],
        dtype=str,
    )
    human_pathways = pathways_df[pathways_df["species"].str.strip() == "Homo sapiens"]
    reactome_names: dict[str, str] = dict(
        zip(human_pathways["reactome_id"], human_pathways["name"])
    )

    log.info(f"Loading hierarchy from {relations_path}")
    rel_df = pd.read_csv(
        relations_path,
        sep="\t",
        header=None,
        names=["parent", "child"],
        dtype=str,
    )
    human_rel = rel_df[
        rel_df["parent"].str.startswith("R-HSA-")
        & rel_df["child"].str.startswith("R-HSA-")
    ]

    hierarchy = nx.DiGraph()
    hierarchy.add_nodes_from(reactome_names.keys())
    hierarchy.add_edges_from(
        zip(human_rel["parent"], human_rel["child"])
    )

    log.info(
        f"Reactome hierarchy: {hierarchy.number_of_nodes():,} nodes, "
        f"{hierarchy.number_of_edges():,} edges."
    )
    return hierarchy, reactome_names


def get_pathway_depth(hierarchy: nx.DiGraph) -> dict[str, int]:
    """
    Compute each node's depth as the longest path from any root.
    Roots (no incoming edges) have depth 0.
    Uses a topological traversal — O(V + E).

    Returns
    -------
    dict: reactome_id → depth (int)
    """
    depth: dict[str, int] = {}
    for node in nx.topological_sort(hierarchy):
        preds = list(hierarchy.predecessors(node))
        depth[node] = 0 if not preds else max(depth.get(p, 0) for p in preds) + 1
    return depth


# ── STRING PPI ────────────────────────────────────────────────────────────────

def build_string_to_uniprot_map(
    uniprotkb_path: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build bidirectional STRING ↔ UniProt ID maps from uniprotkb.tsv.

    STRING IDs look like ``9606.ENSP00000401566``.
    Multiple STRING IDs per UniProt entry are split on ``;``.

    Returns
    -------
    (string_to_uniprot, uniprot_to_string)
        ``uniprot_to_string`` stores the *first* STRING ID when multiple exist.
    """
    log.info(f"Building STRING ↔ UniProt map from {uniprotkb_path}")
    df = pd.read_csv(uniprotkb_path, sep="\t", dtype=str)
    human = df[df["Entry Name"].str.endswith("_HUMAN", na=False)]
    has_string = human[human["STRING"].notna()]

    string_to_uniprot: dict[str, str] = {}
    uniprot_to_string: dict[str, str] = {}

    for _, row in has_string.iterrows():
        uid = str(row["Entry"]).strip()
        string_ids = [s.strip() for s in str(row["STRING"]).split(";") if s.strip()]
        for sid in string_ids:
            string_to_uniprot[sid] = uid
        if string_ids and uid not in uniprot_to_string:
            uniprot_to_string[uid] = string_ids[0]

    log.info(
        f"STRING ↔ UniProt: {len(string_to_uniprot):,} STRING IDs, "
        f"{len(uniprot_to_string):,} UniProt entries."
    )
    return string_to_uniprot, uniprot_to_string


def load_string_interactions(
    string_links_path: str,
    string_to_uniprot: dict[str, str],
    gene_to_uniprot: dict[str, list[str]],
    confidence_threshold: int = 400,
) -> list[tuple[str, str, int]]:
    """
    Load STRING PPI edges for genes present in the expression matrix.

    Strategy
    --------
    1. Build a direct STRING ID → gene symbol lookup for our genes only.
    2. Read the large links file in chunks; filter each chunk immediately.
    3. Normalize edge direction (protein1 < protein2) and map to gene symbols.
    4. Deduplicate at the gene-pair level, keeping the highest score.

    Returns
    -------
    list of (gene1, gene2, combined_score) — one entry per unique gene pair.
    """
    if not os.path.exists(string_links_path):
        log.warning(
            f"STRING links file not found: {string_links_path}. "
            "Skipping PPI loading. Place 9606.protein.links.v12.0.txt "
            f"in {os.path.dirname(string_links_path)}."
        )
        return []

    # Build string_id → gene_symbol for our gene set
    uniprot_to_gene: dict[str, str] = {}
    for gene, uids in gene_to_uniprot.items():
        for uid in uids:
            if uid not in uniprot_to_gene:
                uniprot_to_gene[uid] = gene

    string_to_gene: dict[str, str] = {
        sid: uniprot_to_gene[uid]
        for sid, uid in string_to_uniprot.items()
        if uid in uniprot_to_gene
    }
    relevant_string = set(string_to_gene.keys())

    log.info(
        f"Relevant STRING IDs: {len(relevant_string):,} "
        f"(spanning {len(uniprot_to_gene):,} UniProt entries)."
    )

    best_score: dict[tuple[str, str], int] = {}
    chunk_size = 500_000

    log.info(f"Streaming STRING links: {string_links_path}")
    for chunk in pd.read_csv(
        string_links_path,
        sep=" ",
        chunksize=chunk_size,
        dtype={"protein1": str, "protein2": str, "combined_score": "int32"},
    ):
        mask = (
            chunk["protein1"].isin(relevant_string)
            & chunk["protein2"].isin(relevant_string)
            & (chunk["combined_score"] >= confidence_threshold)
        )
        filtered = chunk[mask].copy()
        if filtered.empty:
            continue

        # Normalize direction: ensure protein1 < protein2
        swap = filtered["protein1"] > filtered["protein2"]
        p1 = filtered["protein1"].where(~swap, filtered["protein2"])
        p2 = filtered["protein2"].where(~swap, filtered["protein1"])
        filtered = filtered.copy()
        filtered["protein1"] = p1
        filtered["protein2"] = p2

        # Map to gene symbols
        filtered["gene1"] = filtered["protein1"].map(string_to_gene)
        filtered["gene2"] = filtered["protein2"].map(string_to_gene)
        filtered = filtered.dropna(subset=["gene1", "gene2"])
        filtered = filtered[filtered["gene1"] != filtered["gene2"]]

        for row in filtered.itertuples(index=False):
            key = (min(row.gene1, row.gene2), max(row.gene1, row.gene2))
            score = int(row.combined_score)
            if score > best_score.get(key, -1):
                best_score[key] = score

    ppi_edges = [(g1, g2, s) for (g1, g2), s in best_score.items()]
    log.info(f"PPI edges after deduplication: {len(ppi_edges):,}")
    return ppi_edges


def build_string_ppi_graph(ppi_edges: list[tuple[str, str, int]]) -> nx.Graph:
    """
    Build an undirected PPI graph.
    Edge attribute: ``weight`` = combined_score.
    Node attribute: ``gene_symbol`` = node label.
    """
    G = nx.Graph()
    for g1, g2, score in ppi_edges:
        G.add_edge(g1, g2, weight=score)
    nx.set_node_attributes(G, {n: n for n in G.nodes()}, "gene_symbol")
    log.info(
        f"PPI graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges."
    )
    return G


# ── Full assembly ─────────────────────────────────────────────────────────────

def build_full_biological_map(gene_list: list[str]) -> dict:
    """
    Assemble the complete biological map for genes in the expression matrix.

    Chain
    -----
    Gene → UniProt (local uniprotkb.tsv + MyGene.info fallback)
         → Reactome pathways (UniProt2Reactome.txt merged with uniprotkb.tsv)
         → STRING PPI edges (9606.protein.links.v12.0.txt)

    Returns
    -------
    dict with keys:
        gene_to_uniprot, uniprot_to_reactome, gene_to_reactome,
        reactome_hierarchy, reactome_names, ppi_edges, ppi_graph,
        string_to_uniprot, uniprot_to_string, unmapped_genes, coverage_stats
    """
    genes = [g.upper().strip() for g in gene_list]
    log.info(f"Building biological map for {len(genes):,} genes.")

    # ── Gene → UniProt (local lookup) ─────────────────────────────────
    all_local = build_gene_to_uniprot(config.UNIPROTKB_FILE)
    gene_to_uniprot: dict[str, list[str]] = {
        g: all_local[g] for g in genes if g in all_local and all_local[g]
    }
    missing_local = [g for g in genes if g not in gene_to_uniprot]
    log.info(
        f"Local coverage: {len(gene_to_uniprot):,}/{len(genes):,} genes. "
        f"{len(missing_local):,} genes to query via MyGene.info."
    )

    # ── MyGene.info fallback ───────────────────────────────────────────
    if missing_local:
        cache = _load_mygene_cache()
        cache = _query_mygene(missing_local, cache)
        for g in missing_local:
            uids = cache.get(g, [])
            if uids:
                gene_to_uniprot[g] = uids

    unmapped_genes = [g for g in genes if not gene_to_uniprot.get(g)]
    log.info(
        f"After MyGene.info: {len(gene_to_uniprot):,}/{len(genes):,} genes mapped. "
        f"{len(unmapped_genes):,} remain unmapped."
    )

    # ── UniProt → Reactome (two sources merged) ────────────────────────
    u2r_1 = build_uniprot_to_reactome(config.UNIPROT_REACTOME_FILE)
    u2r_2 = build_uniprot_to_reactome_from_kb(config.UNIPROTKB_FILE)

    uniprot_to_reactome: dict[str, set[str]] = {}
    for source in (u2r_1, u2r_2):
        for uid, rids in source.items():
            uniprot_to_reactome.setdefault(uid, set()).update(rids)

    # ── Gene → Reactome (transitive through UniProt) ───────────────────
    gene_to_reactome: dict[str, set[str]] = {}
    for gene, uids in gene_to_uniprot.items():
        pathway_set: set[str] = set()
        for uid in uids:
            pathway_set.update(uniprot_to_reactome.get(uid, set()))
        if pathway_set:
            gene_to_reactome[gene] = pathway_set

    # ── Reactome hierarchy ─────────────────────────────────────────────
    hierarchy, reactome_names = load_reactome_hierarchy(
        config.REACTOME_RELATIONS_FILE,
        config.REACTOME_PATHWAYS_FILE,
    )
    depth_map = get_pathway_depth(hierarchy)
    max_depth = max(depth_map.values(), default=0)

    # ── STRING PPI ─────────────────────────────────────────────────────
    string_to_uniprot, uniprot_to_string = build_string_to_uniprot_map(
        config.UNIPROTKB_FILE
    )
    ppi_edges = load_string_interactions(
        config.STRING_LINKS_FILE,
        string_to_uniprot,
        gene_to_uniprot,
        confidence_threshold=config.STRING_CONFIDENCE_THRESHOLD,
    )
    ppi_graph = build_string_ppi_graph(ppi_edges)

    # ── Coverage stats ─────────────────────────────────────────────────
    genes_with_string = {
        gene
        for gene, uids in gene_to_uniprot.items()
        if any(uid in uniprot_to_string for uid in uids)
    }
    coverage_stats: dict = {
        "total_genes": len(genes),
        "genes_with_uniprot": len([g for g in genes if gene_to_uniprot.get(g)]),
        "genes_with_reactome": len(gene_to_reactome),
        "genes_with_string": len(genes_with_string),
        "total_pathways": len(reactome_names),
        "hierarchy_max_depth": max_depth,
        "total_ppi_edges": len(ppi_edges),
        "ppi_confidence_threshold": config.STRING_CONFIDENCE_THRESHOLD,
    }

    # ── Print summary ──────────────────────────────────────────────────
    n = len(genes)
    print()
    print("Biological Mapping Summary")
    print("============================")
    print(f"Total input genes:        {n:,}")
    print(f"Genes → UniProt:          {coverage_stats['genes_with_uniprot']:,}  ({100*coverage_stats['genes_with_uniprot']/n:.1f}%)")
    print(f"Genes → Reactome:         {coverage_stats['genes_with_reactome']:,}  ({100*coverage_stats['genes_with_reactome']/n:.1f}%)")
    print(f"Genes → STRING:           {coverage_stats['genes_with_string']:,}  ({100*coverage_stats['genes_with_string']/n:.1f}%)")
    print(f"Total Reactome pathways:  {coverage_stats['total_pathways']:,}")
    print(f"Hierarchy depth:          {max_depth} layers")
    print(f"PPI edges (gene-level):   {coverage_stats['total_ppi_edges']:,}")
    print(f"Unmapped genes:           {len(unmapped_genes):,}")

    return {
        "gene_to_uniprot": gene_to_uniprot,
        "uniprot_to_reactome": uniprot_to_reactome,
        "gene_to_reactome": gene_to_reactome,
        "reactome_hierarchy": hierarchy,
        "reactome_names": reactome_names,
        "ppi_edges": ppi_edges,
        "ppi_graph": ppi_graph,
        "string_to_uniprot": string_to_uniprot,
        "uniprot_to_string": uniprot_to_string,
        "unmapped_genes": unmapped_genes,
        "coverage_stats": coverage_stats,
    }


# ── Serialization ─────────────────────────────────────────────────────────────

def save_biological_map(bio_map: dict, output_dir: str) -> None:
    """
    Serialize the biological map to ``output_dir``.

    Files written
    -------------
    gene_to_reactome.json         gene → sorted list of Reactome IDs
    reactome_hierarchy_edges.csv  parent, child columns
    reactome_names.json           reactome_id → pathway name
    ppi_edges.csv                 gene1, gene2, combined_score
    coverage_stats.json           summary statistics
    unmapped_genes.txt            one gene per line
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "gene_to_reactome.json"), "w") as f:
        json.dump(
            {g: sorted(rids) for g, rids in bio_map["gene_to_reactome"].items()},
            f,
            indent=2,
        )

    hierarchy: nx.DiGraph = bio_map["reactome_hierarchy"]
    pd.DataFrame(
        list(hierarchy.edges()), columns=["parent", "child"]
    ).to_csv(
        os.path.join(output_dir, "reactome_hierarchy_edges.csv"), index=False
    )

    with open(os.path.join(output_dir, "reactome_names.json"), "w") as f:
        json.dump(bio_map["reactome_names"], f, indent=2)

    pd.DataFrame(
        bio_map["ppi_edges"], columns=["gene1", "gene2", "combined_score"]
    ).to_csv(os.path.join(output_dir, "ppi_edges.csv"), index=False)

    with open(os.path.join(output_dir, "coverage_stats.json"), "w") as f:
        json.dump(bio_map["coverage_stats"], f, indent=2)

    with open(os.path.join(output_dir, "unmapped_genes.txt"), "w") as f:
        f.write("\n".join(bio_map["unmapped_genes"]))

    log.info(f"Biological map saved to {output_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_biological_mapping() -> dict:
    """Build and persist the biological map from the preprocessed gene list."""
    genes_path = config.GENE_LIST_FILE
    if not os.path.exists(genes_path):
        raise FileNotFoundError(
            f"Gene list not found at {genes_path}. "
            "Run src/data_acquisition.py first."
        )
    with open(genes_path) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    bio_map = build_full_biological_map(gene_list)
    save_biological_map(bio_map, config.DATA_PROCESSED)
    with open(config.BIO_MAP_FILE, "wb") as f:
        pickle.dump(bio_map, f)
    log.info(f"Biological map pickle saved to {config.BIO_MAP_FILE}")
    return bio_map


if __name__ == "__main__":
    run_biological_mapping()
