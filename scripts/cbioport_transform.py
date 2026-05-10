"""
Transform cBioPortal output into NeST-VNN input files.

Interactive CLI lets you pick which clinical attributes to use as endpoints.
Works with any cBioPortal study downloaded by cbioportal_download.py.

Usage:
    python transform_to_nest_vnn.py <cbioportal_dir>
    python transform_to_nest_vnn.py cbioportal_laml_tcga_pub
    python transform_to_nest_vnn.py cbioportal_breast_msk_2025

The script will:
  1. Load the downloaded CSVs
  2. Show all available clinical attributes
  3. Let you interactively pick which ones to use as label columns
  4. For status/categorical columns, show unique values and let you pick 0/1 mapping
  5. Generate NeST-VNN input files with a single training_data.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import json
from datetime import datetime


# ── Configuration ────────────────────────────────────────────────────────────

NEST_VNN_SAMPLE = Path("nest_vnn/sample")

CNV_DEEP_DELETION = -2
CNV_AMPLIFICATION = 2


# ── CLI helpers ──────────────────────────────────────────────────────────────

def prompt_choice(prompt: str, options: list[str], allow_multi: bool = False) -> list[int] | int:
    """Display numbered options and get user selection."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")

    if allow_multi:
        print(f"\nEnter numbers separated by commas (e.g. 0,3,5), or 'done' to finish:")
    else:
        print(f"\nEnter number:")

    while True:
        raw = input("> ").strip()
        if raw.lower() in ('done', 'q', 'quit', 'skip', 'none', ''):
            return []
        try:
            if allow_multi:
                indices = [int(x.strip()) for x in raw.split(",")]
            else:
                indices = [int(raw)]
            if all(0 <= i < len(options) for i in indices):
                return indices if allow_multi else indices[0]
        except ValueError:
            pass
        print(f"Invalid input. Enter number(s) 0-{len(options)-1}.")


def prompt_binary_mapping(col_name: str, unique_values: list[str]) -> dict[str, float] | None:
    """Let user pick which values map to 0 (event) and 1 (no event)."""
    print(f"\n  Column '{col_name}' has these unique values:")
    for i, v in enumerate(unique_values):
        print(f"    [{i}] {v}")

    print(f"\n  Which values should map to 0 (event / bad outcome)?")
    print(f"  Enter numbers separated by commas:")
    while True:
        raw = input("  event(0)> ").strip()
        if raw.lower() in ('skip', 'q', ''):
            return None
        try:
            event_indices = [int(x.strip()) for x in raw.split(",")]
            if all(0 <= i < len(unique_values) for i in event_indices):
                break
        except ValueError:
            pass
        print(f"  Invalid. Enter number(s) 0-{len(unique_values)-1}, or 'skip'.")

    print(f"\n  Which values should map to 1 (no event / good outcome)?")
    print(f"  Enter numbers separated by commas:")
    while True:
        raw = input("  no-event(1)> ").strip()
        if raw.lower() in ('skip', 'q', ''):
            return None
        try:
            noevent_indices = [int(x.strip()) for x in raw.split(",")]
            if all(0 <= i < len(unique_values) for i in noevent_indices):
                break
        except ValueError:
            pass
        print(f"  Invalid. Enter number(s) 0-{len(unique_values)-1}, or 'skip'.")

    mapping = {}
    for i in event_indices:
        mapping[unique_values[i]] = 0.0
    for i in noevent_indices:
        mapping[unique_values[i]] = 1.0

    print(f"  Mapping: { {k: int(v) for k, v in mapping.items()} }")
    return mapping


def classify_column(series: pd.Series) -> str:
    """Classify a clinical column as 'numeric', 'binary_candidate', or 'categorical'."""
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    try:
        pd.to_numeric(non_null)
        return "numeric"
    except (ValueError, TypeError):
        pass
    n_unique = non_null.nunique()
    if n_unique <= 10:
        return "binary_candidate"
    return "categorical"


def interactive_endpoint_selection(clinical_df: pd.DataFrame) -> list[dict]:
    """
    Interactive CLI to select clinical attributes as label columns.
    Returns list of endpoint dicts:
        {"label_name": str, "clinical_col": str, "mode": "continuous"|"binary",
         "mapping": dict|None}
    """
    if clinical_df.empty:
        print("⚠ No clinical data available for endpoint selection.")
        return []

    # Skip ID columns
    skip_cols = {"patientId", "sampleId", "PATIENT_ID", "SAMPLE_ID",
                 "studyId", "uniquePatientKey", "uniqueSampleKey"}
    available = [c for c in clinical_df.columns if c not in skip_cols]

    # Classify and summarize each column
    col_info = []
    for col in available:
        series = clinical_df[col].dropna()
        if series.empty:
            continue
        col_type = classify_column(series)
        if col_type == "empty":
            continue

        if col_type == "numeric":
            vals = pd.to_numeric(series)
            summary = f"numeric, n={len(vals)}, median={vals.median():.1f}, range=[{vals.min():.1f}, {vals.max():.1f}]"
        elif col_type == "binary_candidate":
            counts = series.value_counts()
            summary = f"categorical ({series.nunique()} values): " + ", ".join(
                f"{v}({c})" for v, c in counts.items()
            )
        else:
            summary = f"categorical ({series.nunique()} unique values), n={len(series)}"

        col_info.append((col, col_type, summary))

    # Display
    print("\n" + "=" * 70)
    print("AVAILABLE CLINICAL ATTRIBUTES")
    print("=" * 70)
    options = []
    for i, (col, ctype, summary) in enumerate(col_info):
        label = "📊" if ctype == "numeric" else "🏷️"
        print(f"  [{i:2d}] {label} {col:40s} {summary}")
        options.append(col)

    print("\n" + "-" * 70)
    print("Select which attributes to use as endpoints/labels.")
    print("You can add multiple. Enter numbers separated by commas, or 'done' when finished.")
    print("-" * 70)

    indices = prompt_choice(
        "Which clinical attributes should be endpoints?",
        [f"{col} — {summary}" for col, _, summary in col_info],
        allow_multi=True,
    )

    if not indices:
        print("No endpoints selected.")
        return []

    endpoints = []
    for idx in indices:
        col, col_type, _ = col_info[idx]
        series = clinical_df[col].dropna()

        if col_type == "numeric":
            # Offer both continuous and binary (thresholded)
            print(f"\n  '{col}' is numeric. Use as:")
            mode_idx = prompt_choice(
                f"  How to use '{col}'?",
                ["Continuous (regression)", "Binary (threshold)", "Both"],
            )
            if isinstance(mode_idx, list) and not mode_idx:
                continue

            if mode_idx in (0, 2):
                label_name = col.lower().replace(" ", "_").replace("-", "_")
                endpoints.append({
                    "label_name": label_name,
                    "clinical_col": col,
                    "mode": "continuous",
                    "mapping": None,
                })

            if mode_idx in (1, 2):
                vals = pd.to_numeric(series)
                print(f"    median={vals.median():.1f}, mean={vals.mean():.1f}")
                thresh = input(f"    Enter threshold (values > threshold = 1): ").strip()
                try:
                    thresh = float(thresh)
                    label_name = f"binary_{col.lower().replace(' ', '_').replace('-', '_')}"
                    endpoints.append({
                        "label_name": label_name,
                        "clinical_col": col,
                        "mode": "binary_threshold",
                        "mapping": {"threshold": thresh},
                    })
                except ValueError:
                    print("    Invalid threshold, skipping binary version.")

        elif col_type == "binary_candidate":
            unique_vals = sorted(series.unique().tolist(), key=str)
            mapping = prompt_binary_mapping(col, unique_vals)
            if mapping:
                label_name = f"binary_{col.lower().replace(' ', '_').replace('-', '_')}"
                endpoints.append({
                    "label_name": label_name,
                    "clinical_col": col,
                    "mode": "binary_status",
                    "mapping": mapping,
                })
        else:
            print(f"  '{col}' has {series.nunique()} unique values — too many for binary.")
            print(f"  Skipping (use a column with fewer categories).")

    # Summary
    print(f"\n{'─'*70}")
    print(f"Selected {len(endpoints)} endpoint(s):")
    for ep in endpoints:
        print(f"  {ep['label_name']:30s} ← {ep['clinical_col']} ({ep['mode']})")
    print(f"{'─'*70}")

    return endpoints


# ── Gene panel ───────────────────────────────────────────────────────────────

def load_gene_panel(gene2ind_path: Path) -> dict[str, int]:
    gene2ind = {}
    with open(gene2ind_path) as f:
        for line in f:
            idx, gene = line.strip().split("\t")
            gene2ind[gene] = int(idx)
    print(f"Loaded gene panel: {len(gene2ind)} genes")
    return gene2ind


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_gene_col(df: pd.DataFrame) -> str | None:
    for candidate in ["gene.hugoGeneSymbol", "hugoGeneSymbol"]:
        if candidate in df.columns:
            return candidate
    return None


def save_matrix(matrix: np.ndarray, path: Path):
    with open(path, "w") as f:
        for row in matrix:
            f.write(",".join(str(x) for x in row) + "\n")


def save_tsv(df: pd.DataFrame, path: Path, header: bool = False):
    df.to_csv(path, sep="\t", index=False, header=header)


def build_cell2ind(sample_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"idx": range(len(sample_ids)), "sample_id": sample_ids})


# ── Matrix builders ─────────────────────────────────────────────────────────

def build_mutation_matrix(mutations_df, sample_ids, gene2ind):
    n_samples, n_genes = len(sample_ids), len(gene2ind)
    matrix = np.zeros((n_samples, n_genes), dtype=int)
    sample2idx = {s: i for i, s in enumerate(sample_ids)}
    gene_col = find_gene_col(mutations_df)
    if gene_col is None:
        print("⚠ Cannot find gene symbol column in mutations.")
        return matrix
    mapped = 0
    for _, row in mutations_df.iterrows():
        s, g = row.get("sampleId"), row.get(gene_col)
        if s in sample2idx and g in gene2ind:
            matrix[sample2idx[s], gene2ind[g]] = 1
            mapped += 1
    panel_hit = int(np.sum(matrix.any(axis=0)))
    print(f"  Mutations: {mapped}/{len(mutations_df)} mapped ({panel_hit}/{n_genes} genes hit)")
    return matrix


def build_cnv_matrices(cnv_df, sample_ids, gene2ind):
    n_samples, n_genes = len(sample_ids), len(gene2ind)
    del_m = np.zeros((n_samples, n_genes), dtype=int)
    amp_m = np.zeros((n_samples, n_genes), dtype=int)
    if cnv_df.empty:
        print("  CNV: no data")
        return del_m, amp_m
    sample2idx = {s: i for i, s in enumerate(sample_ids)}
    gene_col = find_gene_col(cnv_df)
    if gene_col is None:
        print("⚠ Cannot find gene symbol column in CNV.")
        return del_m, amp_m
    val_col = "alteration" if "alteration" in cnv_df.columns else "value"
    if val_col not in cnv_df.columns:
        print("⚠ Cannot find alteration value column in CNV.")
        return del_m, amp_m
    dc, ac = 0, 0
    for _, row in cnv_df.iterrows():
        s, g, v = row.get("sampleId"), row.get(gene_col), row.get(val_col)
        if s not in sample2idx or g not in gene2ind:
            continue
        try:
            v = int(float(v))
        except (ValueError, TypeError):
            continue
        si, gi = sample2idx[s], gene2ind[g]
        if v <= CNV_DEEP_DELETION:
            del_m[si, gi] = 1; dc += 1
        if v >= CNV_AMPLIFICATION:
            amp_m[si, gi] = 1; ac += 1
    print(f"  CNV deletions: {dc}, amplifications: {ac}")
    return del_m, amp_m


def build_fusion_matrix(fusions_df, sample_ids, gene2ind):
    n_samples, n_genes = len(sample_ids), len(gene2ind)
    matrix = np.zeros((n_samples, n_genes), dtype=int)
    if fusions_df.empty:
        print("  Fusions: no data")
        return matrix
    sample2idx = {s: i for i, s in enumerate(sample_ids)}
    sv_cols = [c for c in ["site1HugoSymbol", "site2HugoSymbol",
                           "site1.hugoSymbol", "site2.hugoSymbol",
                           "gene1.hugoGeneSymbol", "gene2.hugoGeneSymbol"]
               if c in fusions_df.columns]
    has_sv = len(sv_cols) > 0
    ev, gm = 0, 0
    for _, row in fusions_df.iterrows():
        s = row.get("sampleId")
        if s not in sample2idx:
            continue
        si = sample2idx[s]
        genes = []
        if has_sv:
            for c in sv_cols:
                g = row.get(c)
                if isinstance(g, str) and g.strip():
                    genes.append(g.strip())
        else:
            gc = find_gene_col(fusions_df)
            if gc:
                g = row.get(gc)
                if isinstance(g, str) and g.strip():
                    genes.append(g.strip())
        hit = False
        for g in genes:
            if g in gene2ind:
                matrix[si, gene2ind[g]] = 1; gm += 1; hit = True
        if hit:
            ev += 1
    panel_hit = int(np.sum(matrix.any(axis=0)))
    print(f"  Fusions: {ev}/{len(fusions_df)} events ({gm} gene marks, {panel_hit}/{n_genes} genes)")
    return matrix


# ── Training data builder ────────────────────────────────────────────────────


def build_training_data(clinical_df, sample_ids, endpoints):
    if clinical_df.empty or not endpoints:
        return pd.DataFrame()

    # Build a lookup: try sampleId first, then patientId
    # The clinical data may have both columns, or just patientId
    clin_by_sample = {}
    clin_by_patient = {}

    if "sampleId" in clinical_df.columns:
        for _, row in clinical_df.iterrows():
            clin_by_sample[row["sampleId"]] = row
    if "patientId" in clinical_df.columns:
        for _, row in clinical_df.iterrows():
            pid = row["patientId"]
            if pid not in clin_by_patient:  # keep first occurrence
                clin_by_patient[pid] = row

    # Also build sampleId → patientId from genomic data if clinical has patientId
    # Many studies have sampleId = patientId + suffix, but the pattern varies.
    # We try: exact sampleId match, then check if sampleId starts with any patientId.
    sample_to_patient = {}
    if clin_by_patient:
        patient_ids = sorted(clin_by_patient.keys(), key=len, reverse=True)  # longest first
        for sid in sample_ids:
            if sid in clin_by_sample:
                continue  # direct match available
            for pid in patient_ids:
                if sid.startswith(pid):
                    sample_to_patient[sid] = pid
                    break

    def resolve(sid):
        if sid in clin_by_sample:
            return clin_by_sample[sid]
        if sid in sample_to_patient:
            return clin_by_patient[sample_to_patient[sid]]
        if sid in clin_by_patient:
            return clin_by_patient[sid]
        return None

    # Debug: report match rate
    matched = sum(1 for sid in sample_ids if resolve(sid) is not None)
    print(f"\n  Clinical match: {matched}/{len(sample_ids)} samples linked to clinical records")
    if matched == 0:
        # Show what IDs look like to help debug
        sample_examples = sample_ids[:3]
        patient_examples = list(clin_by_patient.keys())[:3] if clin_by_patient else []
        sample_ex = list(clin_by_sample.keys())[:3] if clin_by_sample else []
        print(f"    Sample IDs (mutations):  {sample_examples}")
        print(f"    Patient IDs (clinical):  {patient_examples}")
        print(f"    Sample IDs (clinical):   {sample_ex}")

    rows = []
    for sid in sample_ids:
        record = resolve(sid)
        if record is None:
            continue
        row = {"cell_line": sid}
        has_any = False

        for ep in endpoints:
            raw = record.get(ep["clinical_col"], None)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                row[ep["label_name"]] = None
                continue
            raw_str = str(raw).strip()
            if raw_str == "":
                row[ep["label_name"]] = None
                continue

            if ep["mode"] == "continuous":
                try:
                    row[ep["label_name"]] = float(raw_str)
                    has_any = True
                except (ValueError, TypeError):
                    row[ep["label_name"]] = None

            elif ep["mode"] == "binary_status":
                mapping = ep["mapping"]
                val = mapping.get(raw_str)
                if val is None:
                    # Try case-insensitive match
                    for k, v in mapping.items():
                        if k.upper() == raw_str.upper():
                            val = v
                            break
                row[ep["label_name"]] = val
                if val is not None:
                    has_any = True

            elif ep["mode"] == "binary_threshold":
                try:
                    num = float(raw_str)
                    row[ep["label_name"]] = 1.0 if num > ep["mapping"]["threshold"] else 0.0
                    has_any = True
                except (ValueError, TypeError):
                    row[ep["label_name"]] = None

        row["dataset"] = "cBioPortal"
        if has_any:
            rows.append(row)

    col_order = ["cell_line"] + [ep["label_name"] for ep in endpoints] + ["dataset"]
    df = pd.DataFrame(rows, columns=col_order)

    print(f"\n  Training data: {len(df)} samples")
    for ep in endpoints:
        valid = df[ep["label_name"]].dropna()
        if valid.empty:
            print(f"    {ep['label_name']:30s}  → no data")
        elif ep["mode"] == "continuous":
            print(f"    {ep['label_name']:30s}  → {len(valid)} values  "
                  f"range=[{valid.min():.1f}, {valid.max():.1f}]  median={valid.median():.1f}")
        else:
            n0 = int((valid == 0).sum())
            n1 = int((valid == 1).sum())
            print(f"    {ep['label_name']:30s}  → {len(valid)} values  event(0)={n0}  no-event(1)={n1}")

    return df


# ── README ───────────────────────────────────────────────────────────────────

def write_readme(output_dir, sample_ids, gene2ind, endpoints,
                 mut_matrix, del_matrix, amp_matrix, fus_matrix, training_df):
    lines = []
    w = lines.append
    n_s, n_g = len(sample_ids), len(gene2ind)

    w("# NeST-VNN Input Data")
    w("")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("")
    w(f"- **Samples:** {n_s}")
    w(f"- **Gene panel:** {n_g} genes")
    w("")

    any_alt = (mut_matrix | del_matrix | amp_matrix | fus_matrix)
    w("## Feature Matrices")
    w("")
    w("| Feature | Non-zero | Density |")
    w("|---|---|---|")
    for name, mat in [("Mutations", mut_matrix), ("CN Deletions", del_matrix),
                      ("CN Amplifications", amp_matrix), ("Fusions", fus_matrix)]:
        w(f"| {name} | {int(mat.sum())} | {100*mat.mean():.2f}% |")
    w(f"| **Any alteration** | {int(any_alt.sum())} | "
      f"samples={int(np.sum(any_alt.any(axis=1)))}/{n_s}, "
      f"genes={int(np.sum(any_alt.any(axis=0)))}/{n_g} |")
    w("")

    if not training_df.empty and endpoints:
        w("## Endpoints")
        w("")
        w("| Label | Mode | N | Summary |")
        w("|---|---|---|---|")
        for ep in endpoints:
            col = ep["label_name"]
            if col not in training_df.columns:
                continue
            valid = training_df[col].dropna()
            if valid.empty:
                w(f"| `{col}` | {ep['mode']} | 0 | no data |")
            elif ep["mode"] == "continuous":
                w(f"| `{col}` | continuous | {len(valid)} | "
                  f"median={valid.median():.1f}, [{valid.min():.1f}, {valid.max():.1f}] |")
            else:
                n0, n1 = int((valid==0).sum()), int((valid==1).sum())
                w(f"| `{col}` | binary | {len(valid)} | event(0)={n0}, no-event(1)={n1} |")
        w("")

        w("## Endpoint Configuration")
        w("")
        w("```json")
        w(json.dumps(endpoints, indent=2))
        w("```")
        w("")

    w("## Usage")
    w("")
    if endpoints:
        ep0 = endpoints[0]
        task = "binary" if "binary" in ep0["mode"] else "continuous"
        w("```bash")
        w(f"python src/train.py \\")
        w(f"  -train {output_dir}/training_data.txt \\")
        w(f"  -label {ep0['label_name']} -task {task} \\")
        w(f"  -mutations {output_dir}/cell2mutation.txt \\")
        w(f"  -cn_deletions {output_dir}/cell2cndeletion.txt \\")
        w(f"  -cn_amplifications {output_dir}/cell2cnamplification.txt \\")
        w(f"  -fusions {output_dir}/cell2fusion.txt \\")
        w(f"  -onto {output_dir}/ontology.txt \\")
        w(f"  -gene2id {output_dir}/gene2ind.txt \\")
        w(f"  -cell2id {output_dir}/cell2ind.txt \\")
        w(f"  -std MODEL/std.txt -model MODEL/ \\")
        w(f"  -cuda 0 -epoch 300 -batchsize 64")
        w("```")
    w("")

    (output_dir / "README.md").write_text("\n".join(lines))
    print(f"  ✓ README.md")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transform cBioPortal data → NeST-VNN format")
    parser.add_argument("study_id", help="cBioPortal study ID (e.g. laml_tcga_pub, breast_msk_2025)")
    parser.add_argument("--data-dir", help="Base data directory", default="data")
    parser.add_argument("--endpoints-json", help="JSON file with endpoint config (skip interactive)", default=None)
    args = parser.parse_args()

    study_id = args.study_id
    data_dir = Path(args.data_dir)
    study_dir = data_dir / "output" / study_id

    input_dir = study_dir / "cbioportal_output"
    output_dir = study_dir / "nest_vnn_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared gene panel and ontology at data/ level
    gene2ind_path = data_dir / "gene2ind.txt"
    ontology_path = data_dir / "ontology.txt"

    print("=" * 60)
    print("cBioPortal → NeST-VNN Format Converter")
    print("=" * 60)
    print(f"Study:    {study_id}")
    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Gene map: {gene2ind_path}\n")

    # Validate
    for p in [gene2ind_path, ontology_path]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            print(f"Copy gene2ind.txt and ontology.txt from nest_vnn/sample/ to {data_dir}/")
            sys.exit(1)

    mutations_path = input_dir / "mutations.csv"
    if not mutations_path.exists():
        print(f"ERROR: {mutations_path} not found. Run cbioportal_download.py {study_id} first.")
        sys.exit(1)

    # Load
    gene2ind = load_gene_panel(gene2ind_path)

    print("\nLoading data ...")
    mutations_df = pd.read_csv(mutations_path, low_memory=False)
    print(f"  mutations:  {len(mutations_df)} rows")

    cnv_path = input_dir / "cnv.csv"
    cnv_df = pd.read_csv(cnv_path, low_memory=False) if cnv_path.exists() else pd.DataFrame()
    print(f"  cnv:        {len(cnv_df)} rows")

    fusions_path = input_dir / "fusions.csv"
    fusions_df = pd.read_csv(fusions_path, low_memory=False) if fusions_path.exists() else pd.DataFrame()
    print(f"  fusions:    {len(fusions_df)} rows")

    clinical_path = input_dir / "clinical_outcomes.csv"
    if not clinical_path.exists():
        clinical_path = input_dir / "patient_clinical.csv"
    clinical_df = pd.read_csv(clinical_path, low_memory=False) if clinical_path.exists() else pd.DataFrame()
    print(f"  clinical:   {len(clinical_df)} rows ({clinical_path.name})")

    # Sample universe
    sample_ids = sorted(mutations_df["sampleId"].unique())
    print(f"\nTotal unique samples: {len(sample_ids)}")

    # Endpoint selection
    if args.endpoints_json:
        with open(args.endpoints_json) as f:
            endpoints = json.load(f)
        print(f"\nLoaded {len(endpoints)} endpoints from {args.endpoints_json}")
    else:
        endpoints = interactive_endpoint_selection(clinical_df)

    # Save endpoint config for reproducibility
    if endpoints:
        config_path = output_dir / "endpoints.json"
        with open(config_path, "w") as f:
            json.dump(endpoints, f, indent=2)
        print(f"\nEndpoint config saved to {config_path} (reuse with --endpoints-json)")

    # Build matrices
    print("\nBuilding NeST-VNN matrices ...")
    cell2ind_df = build_cell2ind(sample_ids)
    mut_matrix = build_mutation_matrix(mutations_df, sample_ids, gene2ind)
    del_matrix, amp_matrix = build_cnv_matrices(cnv_df, sample_ids, gene2ind)
    fus_matrix = build_fusion_matrix(fusions_df, sample_ids, gene2ind)
    training_df = build_training_data(clinical_df, sample_ids, endpoints)

    # Write output files
    print(f"\nWriting to {output_dir}/ ...")

    save_tsv(cell2ind_df, output_dir / "cell2ind.txt")
    print(f"  ✓ cell2ind.txt          ({len(cell2ind_df)} samples)")

    shutil.copy(gene2ind_path, output_dir / "gene2ind.txt")
    print(f"  ✓ gene2ind.txt          (from {gene2ind_path})")

    save_matrix(mut_matrix, output_dir / "cell2mutation.txt")
    print(f"  ✓ cell2mutation.txt     ({mut_matrix.shape})")

    save_matrix(del_matrix, output_dir / "cell2cndeletion.txt")
    print(f"  ✓ cell2cndeletion.txt   ({del_matrix.shape})")

    save_matrix(amp_matrix, output_dir / "cell2cnamplification.txt")
    print(f"  ✓ cell2cnamplification.txt ({amp_matrix.shape})")

    save_matrix(fus_matrix, output_dir / "cell2fusion.txt")
    print(f"  ✓ cell2fusion.txt       ({fus_matrix.shape})")

    shutil.copy(ontology_path, output_dir / "ontology.txt")
    print(f"  ✓ ontology.txt          (from {ontology_path})")

    if not training_df.empty:
        training_df.to_csv(output_dir / "training_data.txt", sep="\t", index=False)
        print(f"  ✓ training_data.txt     ({len(training_df)} rows)")

    std_df = pd.DataFrame({"source": ["cBioPortal"], "mean": [0.0], "std": [1.0]})
    save_tsv(std_df, output_dir / "std.txt")
    print(f"  ✓ std.txt")

    write_readme(output_dir, sample_ids, gene2ind, endpoints,
                 mut_matrix, del_matrix, amp_matrix, fus_matrix, training_df)

    print(f"\n{'='*60}")
    print(f"DONE. Output in: {output_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()