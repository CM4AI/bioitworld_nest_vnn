"""
NeST-VNN Explainability: RLIPP Scores + Annotated Hierarchy Visualization

Computes system-level importance (RLIPP) scores from prediction hidden embeddings,
and generates an annotated interactive hierarchy visualization.

This is a standalone version inspired by CM4AI's annotate.py, without CX2/NDEx
dependencies. It produces:
  1. rlipp_scores.txt     — RLIPP scores per ontology term
  2. gene_scores.txt      — Gene-level correlation scores
  3. hierarchy_annotated.graphml — NetworkX graph with RLIPP as node attributes
  4. hierarchy_viz.html    — Interactive HTML visualization

Usage:
    # After running predict.py with hidden output:
    python annotate_hierarchy.py \\
        -hidden result/hidden/ \\
        -ontology nest_vnn_input/ontology.txt \\
        -test nest_vnn_input/training_data.txt \\
        -predicted result/predict.txt \\
        -gene2id nest_vnn_input/gene2ind.txt \\
        -cell2id nest_vnn_input/cell2ind.txt \\
        -label binary_os -task binary \\
        -outdir explainability/ \\
        -genotype_hiddens 4 \\
        -cpu_count 4
"""

import argparse
import os
import numpy as np
import pandas as pd
import time
import json
import warnings
from pathlib import Path
from scipy import stats
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

warnings.filterwarnings('ignore')

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from ndex2.cx2 import CX2Network
    HAS_NDEX2 = True
except ImportError:
    HAS_NDEX2 = False


# ── RLIPP Calculator (adapted for clinical prediction) ──────────────────────

class ClinicalRLIPPCalculator:
    """
    Compute RLIPP (Relative Local Improvement in Predictive Power) scores.

    Adapted from NeST-VNN's rlipp_calculator.py for clinical outcome prediction
    (no drug grouping — all samples treated as one group).
    """

    def __init__(self, args):
        self.ontology = pd.read_csv(
            args.ontology, sep='\t', header=None,
            names=['S', 'T', 'I'], dtype={0: str, 1: str, 2: str}
        )
        self.terms = self.ontology['S'].unique().tolist()
        self.genes = pd.read_csv(
            args.gene2id, sep='\t', header=None, names=['I', 'G']
        )['G']
        self.cell_index = pd.read_csv(
            args.cell2id, sep='\t', header=None, names=['I', 'C']
        )

        # Load predicted values
        self.predicted_vals = np.loadtxt(args.predicted)

        # Load test labels
        self._load_test_labels(args)

        self.hidden_dir = args.hidden.rstrip('/') + '/'
        self.num_hiddens_genotype = args.genotype_hiddens
        self.cpu_count = args.cpu_count
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def _load_test_labels(self, args):
        """Load test labels, handling both legacy and new header formats."""
        with open(args.test) as f:
            first_line = f.readline().strip()

        if 'cell_line' in first_line:
            # New header format
            df = pd.read_csv(args.test, sep='\t')
            label_col = args.label if args.label else 'binary_os'
            df = df.dropna(subset=[label_col])
            self.test_labels = df[label_col].astype(float).values
        else:
            # Legacy 4-column format
            df = pd.read_csv(args.test, sep='\t', header=None,
                             names=['C', 'D', 'AUC', 'DS'])
            self.test_labels = df['AUC'].values

        print(f"Loaded {len(self.test_labels)} test labels, "
              f"{len(self.predicted_vals)} predictions")

    def load_feature(self, element, size):
        file_name = self.hidden_dir + element + '.hidden'
        if not os.path.exists(file_name):
            return None
        return np.loadtxt(file_name, usecols=range(size))

    def load_all_features(self):
        """Load hidden embeddings for all terms and genes."""
        feature_map = {}

        print("Loading term features ...")
        with Pool(self.cpu_count) as p:
            results = p.starmap(
                self.load_feature,
                [(t, self.num_hiddens_genotype) for t in self.terms]
            )
        for i, t in enumerate(self.terms):
            if results[i] is not None:
                feature_map[t] = results[i]

        print("Loading gene features ...")
        with Pool(self.cpu_count) as p:
            results = p.starmap(
                self.load_feature, [(g, 1) for g in self.genes]
            )
        for i, g in enumerate(self.genes):
            if results[i] is not None:
                feature_map[g] = results[i]

        # Build child feature map
        child_feature_map = {}
        for term in self.terms:
            children = [
                row['T'] for _, row in self.ontology.iterrows()
                if row['S'] == term
            ]
            child_feature_map[term] = [
                feature_map[c] for c in children if c in feature_map
            ]

        print(f"Loaded features for {len([t for t in self.terms if t in feature_map])}/{len(self.terms)} terms, "
              f"{len([g for g in self.genes if g in feature_map])}/{len(self.genes)} genes")
        return feature_map, child_feature_map

    def exec_lm(self, X, y):
        """Ridge regression with PCA → Spearman correlation."""
        if X.shape[0] < 5 or X.shape[1] == 0:
            return 0.0, 1.0
        n_components = min(self.num_hiddens_genotype, X.shape[0], X.shape[1])
        if n_components < 1:
            return 0.0, 1.0
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        regr = RidgeCV(cv=min(5, X.shape[0]))
        regr.fit(X_pca, y)
        y_pred = regr.predict(X_pca)
        return stats.spearmanr(y_pred, y)

    def calc_term_rlipp(self, term_features, term_child_features, term):
        """Calculate RLIPP for a single term (no drug grouping)."""
        X_parent = term_features
        if len(term_child_features) == 0:
            return None
        X_child = np.column_stack(term_child_features)
        y = self.predicted_vals

        # Trim to matching length
        n = min(X_parent.shape[0], X_child.shape[0], len(y))
        X_parent = X_parent[:n]
        X_child = X_child[:n]
        y = y[:n]

        p_rho, p_pval = self.exec_lm(X_parent, y)
        c_rho, c_pval = self.exec_lm(X_child, y)

        rlipp = p_rho / c_rho if abs(c_rho) > 1e-10 else 0.0

        return {
            'term': term,
            'p_rho': p_rho,
            'p_pval': p_pval,
            'c_rho': c_rho,
            'c_pval': c_pval,
            'rlipp': rlipp,
        }

    def calc_gene_rho(self, gene_features, gene):
        """Correlation between gene embedding and predictions."""
        y = self.predicted_vals
        n = min(len(gene_features), len(y))
        rho, p_val = stats.spearmanr(gene_features[:n], y[:n])
        return {'gene': gene, 'rho': rho, 'p_val': p_val}

    def calc_scores(self):
        """Calculate all RLIPP and gene scores."""
        print("\nCalculating RLIPP scores ...")
        start = time.time()
        feature_map, child_feature_map = self.load_all_features()
        print(f"Features loaded in {time.time() - start:.1f}s")

        # Term RLIPP scores
        start = time.time()
        rlipp_results = []
        with Parallel(backend="multiprocessing", n_jobs=self.cpu_count) as parallel:
            results = parallel(
                delayed(self.calc_term_rlipp)(
                    feature_map[term], child_feature_map[term], term
                )
                for term in self.terms if term in feature_map
            )
            rlipp_results = [r for r in results if r is not None]

        rlipp_df = pd.DataFrame(rlipp_results)
        rlipp_df = rlipp_df.sort_values('rlipp', ascending=False)

        rlipp_path = self.outdir / 'rlipp_scores.txt'
        rlipp_df.to_csv(rlipp_path, sep='\t', index=False)
        print(f"RLIPP scores: {len(rlipp_df)} terms → {rlipp_path}")

        # Gene correlation scores
        gene_results = []
        with Parallel(backend="multiprocessing", n_jobs=self.cpu_count) as parallel:
            results = parallel(
                delayed(self.calc_gene_rho)(feature_map[gene], gene)
                for gene in self.genes if gene in feature_map
            )
            gene_results = [r for r in results if r is not None]

        gene_df = pd.DataFrame(gene_results)
        gene_df = gene_df.sort_values('rho', ascending=False, key=abs)

        gene_path = self.outdir / 'gene_scores.txt'
        gene_df.to_csv(gene_path, sep='\t', index=False)
        print(f"Gene scores:  {len(gene_df)} genes → {gene_path}")
        print(f"Scores computed in {time.time() - start:.1f}s")

        return rlipp_df, gene_df


# ── Hierarchy Builder ────────────────────────────────────────────────────────

def build_annotated_hierarchy(ontology_path, rlipp_df, gene_df, outdir):
    """
    Build an annotated hierarchy graph from the ontology + RLIPP scores.
    Exports GraphML and interactive HTML.
    """
    ontology = pd.read_csv(
        ontology_path, sep='\t', header=None,
        names=['parent', 'child', 'relation']
    )

    # Build score lookups
    rlipp_scores = {}
    if not rlipp_df.empty:
        rlipp_scores = dict(zip(rlipp_df['term'], rlipp_df['rlipp']))

    gene_scores = {}
    if not gene_df.empty:
        gene_scores = dict(zip(gene_df['gene'], gene_df['rho']))

    # Identify terms vs genes
    parents = set(ontology['parent'])
    children = set(ontology['child'])
    genes = set(ontology[ontology['relation'] == 'gene']['child'])
    terms = (parents | children) - genes

    # Build graph
    if not HAS_NETWORKX:
        print("⚠ networkx not installed — skipping GraphML export")
        G = None
    else:
        G = nx.DiGraph()

        # Add term nodes
        for term in terms:
            attrs = {
                'node_type': 'term',
                'rlipp': float(rlipp_scores.get(term, 0.0)),
            }
            # Add p_rho, c_rho if available
            if not rlipp_df.empty and term in rlipp_scores:
                row = rlipp_df[rlipp_df['term'] == term].iloc[0]
                attrs['p_rho'] = float(row['p_rho'])
                attrs['c_rho'] = float(row['c_rho'])
                attrs['p_pval'] = float(row['p_pval'])
                attrs['c_pval'] = float(row['c_pval'])
            G.add_node(term, **attrs)

        # Add gene nodes
        for gene in genes:
            G.add_node(gene, **{
                'node_type': 'gene',
                'rho': float(gene_scores.get(gene, 0.0)),
            })

        # Add edges
        for _, row in ontology.iterrows():
            G.add_edge(row['parent'], row['child'], relation=row['relation'])

        # Export GraphML
        graphml_path = outdir / 'hierarchy_annotated.graphml'
        nx.write_graphml(G, graphml_path)
        print(f"GraphML:      {graphml_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    # Export CX2
    build_cx2_hierarchy(ontology, terms, genes, rlipp_scores, rlipp_df, gene_scores, gene_df, outdir)

    # Build interactive HTML visualization
    html_path = outdir / 'hierarchy_viz.html'
    build_html_viz(ontology, terms, genes, rlipp_scores, rlipp_df, gene_scores, gene_df, html_path)
    print(f"HTML viz:     {html_path}")

    # Summary table
    summary_path = outdir / 'top_systems.txt'
    if not rlipp_df.empty:
        top = rlipp_df.head(20)
        top.to_csv(summary_path, sep='\t', index=False, float_format='%.4f')
        print(f"\nTop 20 systems by RLIPP:")
        for _, row in top.iterrows():
            print(f"  {row['term']:30s}  RLIPP={row['rlipp']:.3f}  "
                  f"P_rho={row['p_rho']:.3f}  C_rho={row['c_rho']:.3f}")

    return G


def build_cx2_hierarchy(ontology, terms, genes, rlipp_scores, rlipp_df, gene_scores, gene_df, outdir):
    """Build a CX2 hierarchy file compatible with Cytoscape Web and NDEx."""
    if not HAS_NDEX2:
        print("CX2:          ⚠ ndex2 not installed — skipping (pip install ndex2)")
        return

    import math

    def safe_float(val, default=0.0):
        """Convert to float, replacing NaN/inf with default."""
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return default
            return f
        except (ValueError, TypeError):
            return default

    # Build gene p_val lookup
    gene_pvals = {}
    if not gene_df.empty and 'p_val' in gene_df.columns:
        gene_pvals = dict(zip(gene_df['gene'], gene_df['p_val']))

    # Build parent→children map from ontology for recursive gene collection
    ont_children = {}
    ont_gene_children = {}
    for _, row in ontology.iterrows():
        parent = row['parent']
        child = row['child']
        relation = row['relation']
        if relation == 'gene':
            ont_gene_children.setdefault(parent, []).append(child)
        else:
            ont_children.setdefault(parent, []).append(child)

    def get_all_descendant_genes(term, visited=None):
        if visited is None:
            visited = set()
        if term in visited:
            return []
        visited.add(term)
        result = list(ont_gene_children.get(term, []))
        for child_term in ont_children.get(term, []):
            result.extend(get_all_descendant_genes(child_term, visited))
        return result

    net = CX2Network()
    net.set_network_attributes({
        'name': 'NeST-VNN Annotated Hierarchy',
        'description': 'Ontology hierarchy annotated with RLIPP system importance scores',
    })

    # Track node name → CX2 node ID mapping
    name_to_id = {}

    # Add term nodes
    for term in sorted(terms):
        node_id = net.add_node(attributes={'name': term, 'type': 'term'})
        name_to_id[term] = node_id

        rlipp = safe_float(rlipp_scores.get(term, 0.0))
        net.add_node_attribute(node_id, 'RLIPP', rlipp, datatype='double')

        if not rlipp_df.empty and term in rlipp_scores:
            row = rlipp_df[rlipp_df['term'] == term]
            if not row.empty:
                row = row.iloc[0]
                net.add_node_attribute(node_id, 'P_rho', safe_float(row['p_rho']), datatype='double')
                net.add_node_attribute(node_id, 'P_pval', safe_float(row['p_pval'], 1.0), datatype='double')
                net.add_node_attribute(node_id, 'C_rho', safe_float(row['c_rho']), datatype='double')
                net.add_node_attribute(node_id, 'C_pval', safe_float(row['c_pval'], 1.0), datatype='double')

        # All descendant genes (recursive)
        all_desc_genes = sorted(set(get_all_descendant_genes(term)))
        net.add_node_attribute(node_id, 'gene_count', len(all_desc_genes), datatype='integer')
        if all_desc_genes:
            net.add_node_attribute(node_id, 'descendant_genes',
                                   all_desc_genes, datatype='list_of_string')

    # Add gene nodes
    for gene in sorted(genes):
        node_id = net.add_node(attributes={'name': gene, 'type': 'gene'})
        name_to_id[gene] = node_id

        rho = safe_float(gene_scores.get(gene, 0.0))
        net.add_node_attribute(node_id, 'rho', rho, datatype='double')

        p_val = safe_float(gene_pvals.get(gene, 1.0), 1.0)
        net.add_node_attribute(node_id, 'p_val', p_val, datatype='double')

    # Add edges
    for _, row in ontology.iterrows():
        parent = row['parent']
        child = row['child']
        if parent in name_to_id and child in name_to_id:
            net.add_edge(source=name_to_id[parent], target=name_to_id[child],
                         attributes={'interaction': row['relation']})

    # Write CX2 file
    cx2_path = outdir / 'hierarchy_annotated.cx2'
    net.write_as_raw_cx2(str(cx2_path))
    n_nodes = len(net.get_nodes())
    n_edges = len(net.get_edges())
    print(f"CX2:          {cx2_path} ({n_nodes} nodes, {n_edges} edges)")


def build_html_viz(ontology, terms, genes, rlipp_scores, rlipp_df, gene_scores, gene_df, outpath):
    """Build a standalone interactive HTML visualization of the annotated hierarchy."""
    import math

    def sf(val, default=0.0):
        try:
            f = float(val)
            return default if math.isnan(f) or math.isinf(f) else round(f, 4)
        except (ValueError, TypeError):
            return default

    # Build RLIPP lookup by term
    rlipp_rows = {}
    if not rlipp_df.empty:
        for _, row in rlipp_df.iterrows():
            rlipp_rows[row['term']] = row

    # Build gene p_val lookup
    gene_pvals = {}
    if not gene_df.empty and 'p_val' in gene_df.columns:
        gene_pvals = dict(zip(gene_df['gene'], gene_df['p_val']))

    # Prepare data for JavaScript
    nodes = []
    for term in sorted(terms):
        node = {
            'id': term, 'type': 'term', 'label': term,
            'rlipp': sf(rlipp_scores.get(term, 0.0)),
            'p_rho': 0.0, 'p_pval': 1.0,
            'c_rho': 0.0, 'c_pval': 1.0,
        }
        if term in rlipp_rows:
            r = rlipp_rows[term]
            node['p_rho'] = sf(r.get('p_rho', 0.0))
            node['p_pval'] = sf(r.get('p_pval', 1.0), 1.0)
            node['c_rho'] = sf(r.get('c_rho', 0.0))
            node['c_pval'] = sf(r.get('c_pval', 1.0), 1.0)
        nodes.append(node)

    # Only include genes with significant correlations (top 50 by |rho|)
    gene_items = sorted(
        [(g, gene_scores.get(g, 0.0)) for g in genes],
        key=lambda x: abs(x[1]), reverse=True
    )
    top_genes = set()
    for g, rho in gene_items[:50]:
        nodes.append({
            'id': g, 'type': 'gene',
            'rho': sf(rho),
            'p_val': sf(gene_pvals.get(g, 1.0), 1.0),
            'label': g
        })
        top_genes.add(g)

    edges = []
    for _, row in ontology.iterrows():
        if row['relation'] == 'gene' and row['child'] not in top_genes:
            continue
        edges.append({
            'source': row['parent'], 'target': row['child'],
            'relation': row['relation']
        })

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    # Compute RLIPP range for color scaling
    rlipp_vals = [s for s in rlipp_scores.values() if abs(s) > 0]
    max_rlipp = max(rlipp_vals) if rlipp_vals else 1.0

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NeST-VNN Annotated Hierarchy</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0a0a; color: #e0e0e0; }}
#header {{ padding: 16px 24px; background: #1a1a1a; border-bottom: 1px solid #333;
           display: flex; justify-content: space-between; align-items: center; }}
#header h1 {{ font-size: 18px; font-weight: 600; }}
#header .stats {{ font-size: 13px; color: #888; }}
#controls {{ padding: 12px 24px; background: #141414; border-bottom: 1px solid #222;
             display: flex; gap: 16px; align-items: center; font-size: 13px; }}
#controls label {{ color: #aaa; }}
#controls input, #controls select {{ background: #222; border: 1px solid #444; color: #e0e0e0;
                                      padding: 4px 8px; border-radius: 4px; }}
#main {{ display: flex; height: calc(100vh - 100px); }}
#table-panel {{ width: 580px; overflow-y: auto; border-right: 1px solid #222;
                padding: 8px; font-size: 12px; }}
#table-panel table {{ width: 100%; border-collapse: collapse; }}
#table-panel th {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #333;
                   position: sticky; top: 0; background: #141414; color: #aaa; font-size: 11px; }}
#table-panel td {{ padding: 5px 8px; border-bottom: 1px solid #1a1a1a; cursor: pointer; }}
#table-panel tr:hover {{ background: #1a2a3a; }}
#table-panel tr.selected {{ background: #1a3a2a; }}
.bar {{ display: inline-block; height: 12px; border-radius: 2px; min-width: 2px; }}
#detail {{ flex: 1; padding: 24px; overflow-y: auto; }}
#detail h2 {{ font-size: 16px; margin-bottom: 12px; color: #7eb8da; }}
#detail .meta {{ color: #888; font-size: 13px; margin-bottom: 16px; }}
#detail .children {{ margin-top: 12px; }}
#detail .child {{ display: inline-block; background: #1a1a2e; padding: 4px 10px; margin: 3px;
                  border-radius: 4px; font-size: 12px; cursor: pointer; }}
#detail .child:hover {{ background: #2a2a4e; }}
#detail .child.gene {{ background: #1e2a1e; }}
.rlipp-high {{ color: #f5a623; }}
.rlipp-mid {{ color: #7eb8da; }}
.rlipp-low {{ color: #666; }}
</style>
</head>
<body>
<div id="header">
    <h1>NeST-VNN Annotated Hierarchy</h1>
    <div class="stats" id="stats"></div>
</div>
<div id="controls">
    <label>Sort by:</label>
    <select id="sort-select">
        <option value="rlipp">RLIPP (desc)</option>
        <option value="name">Name</option>
        <option value="p_rho">P_rho (desc)</option>
    </select>
    <label>Filter:</label>
    <input type="text" id="filter-input" placeholder="Search terms...">
    <label>Min RLIPP:</label>
    <input type="number" id="min-rlipp" value="0" step="0.1" style="width:70px">
</div>
<div id="main">
    <div id="table-panel"><table>
        <thead><tr><th>System</th><th>RLIPP</th><th>P_rho</th><th>P_pval</th><th>C_rho</th><th>C_pval</th><th></th></tr></thead>
        <tbody id="table-body"></tbody>
    </table></div>
    <div id="detail" id="detail-panel">
        <p style="color:#666">Select a system from the table to see details.</p>
    </div>
</div>
<script>
const nodes = {nodes_json};
const edges = {edges_json};
const maxRlipp = {max_rlipp};

const termNodes = nodes.filter(n => n.type === 'term');
const geneNodes = nodes.filter(n => n.type === 'gene');
const nodeMap = {{}};
nodes.forEach(n => nodeMap[n.id] = n);

// Build parent/child maps
const childrenOf = {{}};
const parentsOf = {{}};
edges.forEach(e => {{
    if (!childrenOf[e.source]) childrenOf[e.source] = [];
    childrenOf[e.source].push({{id: e.target, relation: e.relation}});
    if (!parentsOf[e.target]) parentsOf[e.target] = [];
    parentsOf[e.target].push(e.source);
}});

document.getElementById('stats').textContent =
    `${{termNodes.length}} systems · ${{geneNodes.length}} top genes · ${{edges.length}} edges`;

function rlippColor(val) {{
    if (val > 1.2) return '#f5a623';
    if (val > 1.0) return '#7eb8da';
    return '#555';
}}

function renderTable() {{
    const sort = document.getElementById('sort-select').value;
    const filter = document.getElementById('filter-input').value.toLowerCase();
    const minRlipp = parseFloat(document.getElementById('min-rlipp').value) || 0;

    let data = termNodes.filter(n => {{
        if (filter && !n.id.toLowerCase().includes(filter)) return false;
        if (n.rlipp < minRlipp) return false;
        return true;
    }});

    if (sort === 'rlipp') data.sort((a, b) => b.rlipp - a.rlipp);
    else if (sort === 'name') data.sort((a, b) => a.id.localeCompare(b.id));
    else if (sort === 'p_rho') data.sort((a, b) => (b.p_rho||0) - (a.p_rho||0));

    const tbody = document.getElementById('table-body');
    tbody.innerHTML = data.map(n => {{
        const w = Math.max(2, Math.min(120, (n.rlipp / maxRlipp) * 120));
        const c = rlippColor(n.rlipp);
        return `<tr onclick="showDetail('${{n.id}}')" data-id="${{n.id}}">
            <td>${{n.id}}</td>
            <td style="color:${{c}}">${{n.rlipp.toFixed(3)}}</td>
            <td>${{(n.p_rho||0).toFixed(3)}}</td>
            <td>${{(n.p_pval||0).toExponential(1)}}</td>
            <td>${{(n.c_rho||0).toFixed(3)}}</td>
            <td>${{(n.c_pval||0).toExponential(1)}}</td>
            <td><span class="bar" style="width:${{w}}px;background:${{c}}"></span></td>
        </tr>`;
    }}).join('');
}}

function showDetail(termId) {{
    document.querySelectorAll('#table-body tr').forEach(tr => {{
        tr.classList.toggle('selected', tr.dataset.id === termId);
    }});

    const node = nodeMap[termId];
    const children = childrenOf[termId] || [];
    const parents = parentsOf[termId] || [];

    // Recursively collect all descendant genes
    function getAllDescendantGenes(tid, visited) {{
        if (visited.has(tid)) return [];
        visited.add(tid);
        const kids = childrenOf[tid] || [];
        let genes = [];
        kids.forEach(c => {{
            if (c.relation === 'gene') {{
                genes.push(c.id);
            }} else {{
                genes = genes.concat(getAllDescendantGenes(c.id, visited));
            }}
        }});
        return genes;
    }}

    let html = `<h2>${{termId}}</h2>`;
    html += `<div class="meta">`;
    html += `RLIPP: <strong style="color:${{rlippColor(node.rlipp)}}">${{node.rlipp.toFixed(4)}}</strong>`;
    if (node.p_rho !== undefined) html += ` &nbsp;|&nbsp; P_rho: ${{node.p_rho.toFixed(4)}}`;
    if (node.p_pval !== undefined) html += ` (p=${{node.p_pval.toExponential(2)}})`;
    if (node.c_rho !== undefined) html += ` &nbsp;|&nbsp; C_rho: ${{node.c_rho.toFixed(4)}}`;
    if (node.c_pval !== undefined) html += ` (p=${{node.c_pval.toExponential(2)}})`;
    html += `</div>`;

    if (parents.length > 0) {{
        html += `<div class="children"><strong>Parents (${{parents.length}}):</strong><br>`;
        parents.forEach(p => {{
            const pn = nodeMap[p];
            const rl = pn ? ` (RLIPP=${{pn.rlipp.toFixed(3)}})` : '';
            html += `<span class="child" onclick="showDetail('${{p}}')">${{p}}${{rl}}</span>`;
        }});
        html += `</div>`;
    }}

    const termChildren = children.filter(c => c.relation !== 'gene');
    const directGenes = children.filter(c => c.relation === 'gene');

    if (termChildren.length > 0) {{
        html += `<div class="children"><strong>Child systems (${{termChildren.length}}):</strong><br>`;
        termChildren.forEach(c => {{
            const cn = nodeMap[c.id];
            const rl = cn ? ` (RLIPP=${{cn.rlipp.toFixed(3)}})` : '';
            html += `<span class="child" onclick="showDetail('${{c.id}}')">${{c.id}}${{rl}}</span>`;
        }});
        html += `</div>`;
    }}

    // Direct genes
    if (directGenes.length > 0) {{
        html += `<div class="children" style="margin-top:12px"><strong>Direct genes (${{directGenes.length}}):</strong><br>`;
        directGenes.forEach(c => {{
            const gn = nodeMap[c.id];
            let info = '';
            if (gn) {{
                info = ` (ρ=${{gn.rho.toFixed(3)}}`;
                if (gn.p_val !== undefined) info += `, p=${{gn.p_val.toExponential(1)}}`;
                info += `)`;
            }}
            html += `<span class="child gene">${{c.id}}${{info}}</span>`;
        }});
        html += `</div>`;
    }}

    // All descendant genes (from child subsystems)
    const allGenes = [...new Set(getAllDescendantGenes(termId, new Set()))];
    const directGeneIds = new Set(directGenes.map(c => c.id));
    const inheritedGenes = allGenes.filter(g => !directGeneIds.has(g));

    if (inheritedGenes.length > 0) {{
        // Sort by |rho| descending
        inheritedGenes.sort((a, b) => {{
            const ra = nodeMap[a] ? Math.abs(nodeMap[a].rho || 0) : 0;
            const rb = nodeMap[b] ? Math.abs(nodeMap[b].rho || 0) : 0;
            return rb - ra;
        }});
        const showCount = Math.min(inheritedGenes.length, 100);
        const label = inheritedGenes.length > showCount
            ? `All descendant genes (${{inheritedGenes.length}}, showing top ${{showCount}} by |ρ|)`
            : `All descendant genes (${{inheritedGenes.length}})`;
        html += `<div class="children" style="margin-top:12px"><strong>${{label}}:</strong><br>`;
        inheritedGenes.slice(0, showCount).forEach(g => {{
            const gn = nodeMap[g];
            let info = '';
            if (gn) {{
                info = ` (ρ=${{gn.rho.toFixed(3)}}`;
                if (gn.p_val !== undefined) info += `, p=${{gn.p_val.toExponential(1)}}`;
                info += `)`;
            }} else {{
                info = '';
            }}
            html += `<span class="child gene">${{g}}${{info}}</span>`;
        }});
        html += `</div>`;
    }}

    document.getElementById('detail').innerHTML = html;
}}

document.getElementById('sort-select').addEventListener('change', renderTable);
document.getElementById('filter-input').addEventListener('input', renderTable);
document.getElementById('min-rlipp').addEventListener('input', renderTable);
renderTable();
</script>
</body>
</html>"""

    outpath.write_text(html)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='NeST-VNN Explainability: RLIPP + Annotated Hierarchy'
    )
    parser.add_argument('study_id', help='cBioPortal study ID (e.g. laml_tcga_pub)')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('-hidden', default=None, help='Hidden embeddings directory (override)')
    parser.add_argument('-predicted', default=None, help='Predicted values file (override)')
    parser.add_argument('-test', default=None, help='Test data file (override)')
    parser.add_argument('-label', default=None, help='Label column (for new-format test files)')
    parser.add_argument('-task', default='continuous', choices=['continuous', 'binary'])
    parser.add_argument('-cpu_count', type=int, default=1, help='CPU cores for parallel computation')
    parser.add_argument('-genotype_hiddens', type=int, default=4, help='Hidden dim per term')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    study_dir = data_dir / "output" / args.study_id
    input_dir = study_dir / "nest_vnn_input"
    model_dir = study_dir / "model"
    metrics_dir = study_dir / "metrics"
    annotation_dir = study_dir / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths from convention, allow overrides
    args.ontology = str(input_dir / "ontology.txt")
    args.gene2id = str(input_dir / "gene2ind.txt")
    args.cell2id = str(input_dir / "cell2ind.txt")
    args.outdir = str(annotation_dir)

    if args.hidden is None:
        args.hidden = str(metrics_dir / "hidden")
    if args.predicted is None:
        args.predicted = str(metrics_dir / "predict.txt")
    if args.test is None:
        args.test = str(input_dir / "training_data.txt")

    print("=" * 60)
    print("NeST-VNN Explainability Pipeline")
    print("=" * 60)
    print(f"Study:      {args.study_id}")
    print(f"Input:      {input_dir}")
    print(f"Model:      {model_dir}")
    print(f"Annotation: {annotation_dir}\n")

    # Compute RLIPP and gene scores
    calculator = ClinicalRLIPPCalculator(args)
    rlipp_df, gene_df = calculator.calc_scores()

    # Build annotated hierarchy
    print("\nBuilding annotated hierarchy ...")
    outdir = Path(args.outdir)
    G = build_annotated_hierarchy(args.ontology, rlipp_df, gene_df, outdir)

    print(f"\n{'='*60}")
    print(f"DONE. Outputs in: {outdir.resolve()}")
    print(f"{'='*60}")
    print(f"""
Output files:
  rlipp_scores.txt           — RLIPP scores for all ontology terms
  gene_scores.txt            — Gene-level Spearman correlations
  hierarchy_annotated.graphml — Annotated hierarchy (open in Cytoscape)
  hierarchy_annotated.cx2    — Annotated hierarchy in CX2 format (NDEx/Cytoscape Web)
  hierarchy_viz.html         — Interactive HTML visualization (open in browser)
  top_systems.txt            — Top 20 systems by RLIPP

The RLIPP score measures relative local improvement in predictive power:
  RLIPP > 1: the system's hidden representation adds information
             beyond its children (important system)
  RLIPP ≈ 1: the system doesn't add much beyond its children
  RLIPP < 1: children are more informative than the parent
""")


if __name__ == "__main__":
    main()