#!/usr/bin/env python3
"""
download_data.py -- Stage 1 data acquisition for the STARLING SAXS benchmark.

Downloads the experimental SAXS data and reference results published with the
STARLING paper (Novak, Lotthammer, Emenecker & Holehouse, bioRxiv
2025.02.14.638373) from the Holehouse lab supporting-data repository:

    github.com/holehouse-lab/supportingdata/tree/master/2026/starling_2026

Two datasets are fetched:

1. Rg benchmark (paper Fig. 3B equivalent)
   analysis/experimental_comparison/saxs_rg/
     - all_comparison_data.csv    (137 sequences w/ SAXS Rg + other-method Rg)
     - all_comparison_seqs.fasta  (sequences; filter <384 aa -> 133 sequences)

2. Full SAXS-curve benchmark (paper Fig. 3E / S13-S14 equivalent)
   analysis/experimental_comparison/saxs_scattering/
     - experiment/sequences.fasta         (53 proteins)
     - experiment/mff_analysis_all.csv    (MFF-derived experimental Rg values)
     - experiment/<name>/<name>.dat       (experimental scattering curves)
     - experiment/<name>/<name>_clean.dat (cleaned curves, preferred when present)
     - ensembles/<name>/average_curve.dat (reference STARLING/FoXS curves)

Usage
-----
    python download_data.py --out ./data
    python download_data.py --out ./data --only rg
    python download_data.py --out ./data --only scattering

Only the Python standard library + `requests` are needed.
"""

import argparse
import json
import os
import sys
import time

import requests

REPO = "holehouse-lab/supportingdata"
BRANCH = "master"
BASE_DIR = "2026/starling_2026/analysis/experimental_comparison"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{BASE_DIR}"
API_TREES = f"https://api.github.com/repos/{REPO}/git/trees"

# Fallback list of protein folders (used only if the GitHub tree API is
# unreachable). Mirrors saxs_scattering/experiment/ as of 2026-02.
FALLBACK_PROTEINS = [
    "ANAC013_161_274", "a1_lcd", "a1_lcd_aro_minus_martin_2020",
    "a1_lcd_aro_minus_minus_martin_2020", "a1_lcd_aro_plus_martin_2020",
    "a1_lcd_plus7k_plus12d_blocky", "a1_lcd_plus7r_plus12d_minus10f",
    "a1_lcd_wt_martin_2020", "alpha_syn", "anac046", "ash1", "atcp12",
    "bmal1_530_625", "cgas_ntd", "dss1", "e1a_36_146", "ebna1_381_455",
    "eif4f_p150", "fatz1_delta91", "fhua", "ghr_icd", "gon7", "heh_nls",
    "hev_pnt3", "hev_pnt3_200_314", "hev_pnt3_yyy_aaa", "hst5", "ibb",
    "laf1_rgg_ysg2max", "mbp", "n_cornid", "n_fatz1", "nhe6cmdd",
    "nup153_nul", "nup49", "nupr1", "nurs_red1", "p53", "pnt", "pol2_ctd",
    "prota", "rnasea", "rs", "serf", "sfafp", "sic1", "smad2_linker",
    "syndecan3_ed", "syndecan4_ed", "tau", "tau_504_758", "tir_ctd",
    "trf2_ntd", "ul11",
]


def robust_get(url, dest=None, tries=8, timeout=(10, 60)):
    """GET with retries + resume; optionally stream to file."""
    headers = {}
    mode = "wb"
    if dest and os.path.exists(dest):
        have = os.path.getsize(dest)
        if have > 0:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=bool(dest))
            if r.status_code in (200, 206):
                if dest is None:
                    return r.content
                # server may ignore Range -> restart file
                if r.status_code == 200:
                    mode = "wb"
                with open(dest, mode) as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
                return dest
            if r.status_code == 416 and dest:  # range not satisfiable == complete
                return dest
            print(f"  [http {r.status_code}] {url}")
        except Exception as e:
            print(f"  [retry {attempt + 1}/{tries}] {type(e).__name__}: {e}")
        time.sleep(2 + attempt)
    return None


def get_tree_paths():
    """Return list of blob paths under the experimental_comparison dir."""
    # tree sha for master:2026/starling_2026 -> walk two levels
    r = requests.get(f"{API_TREES}/{BRANCH}:2026", timeout=(10, 60))
    r.raise_for_status()
    sha = next(t["sha"] for t in r.json()["tree"] if t["path"] == "starling_2026")
    r = requests.get(f"{API_TREES}/{sha}?recursive=1", timeout=(10, 120))
    r.raise_for_status()
    prefix = "analysis/experimental_comparison/"
    return [
        t["path"][len(prefix):]
        for t in r.json()["tree"]
        if t["type"] == "blob" and t["path"].startswith(prefix)
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./data", help="output data directory")
    ap.add_argument("--only", choices=["rg", "scattering"], default=None,
                    help="download only one of the two datasets")
    args = ap.parse_args()

    want_rg = args.only in (None, "rg")
    want_sc = args.only in (None, "scattering")

    # ------------------------------------------------------------------ paths
    try:
        all_paths = get_tree_paths()
        print(f"[tree] {len(all_paths)} files discovered via GitHub API")
    except Exception as e:
        print(f"[tree] GitHub API unavailable ({type(e).__name__}); "
              f"using embedded file list")
        all_paths = None

    downloads = []  # (remote_path_relative_to_BASE_DIR, local_path)

    if want_rg:
        for f in ["all_comparison_data.csv", "all_comparison_seqs.fasta", "readme.md"]:
            downloads.append((f"saxs_rg/{f}", os.path.join(args.out, "saxs_rg", f)))

    if want_sc:
        sc = "saxs_scattering"
        downloads.append((f"{sc}/experiment/sequences.fasta",
                          os.path.join(args.out, sc, "experiment", "sequences.fasta")))
        downloads.append((f"{sc}/experiment/mff_analysis_all.csv",
                          os.path.join(args.out, sc, "experiment", "mff_analysis_all.csv")))
        downloads.append((f"{sc}/experiment/readme.md",
                          os.path.join(args.out, sc, "experiment", "readme.md")))

        if all_paths is not None:
            sc_paths = [p for p in all_paths if p.startswith(f"{sc}/")]
            proteins = sorted({
                p.split("/")[2] for p in sc_paths
                if p.startswith(f"{sc}/experiment/") and len(p.split("/")) > 3
            })
            # case-insensitive map of ensemble folders -> average_curve.dat
            # (upstream case quirks: experiment folder 'nhe6cmdd' contains
            # files 'nhE6cmdd.dat', and the ensembles folder is 'nhE6cmdd')
            ref_map = {}
            for p in sc_paths:
                parts = p.split("/")
                if (len(parts) == 4 and parts[1] == "ensembles"
                        and parts[3] == "average_curve.dat"):
                    ref_map[parts[2].lower()] = p
            for name in proteins:
                sub = [p for p in sc_paths if p.startswith(f"{sc}/experiment/{name}/")]
                for p in sub:
                    # grab every .dat in the folder (raw + cleaned variants)
                    if p.lower().endswith(".dat"):
                        downloads.append((p, os.path.join(args.out, p)))
                ref = ref_map.get(name.lower())
                if ref is not None:
                    downloads.append((ref, os.path.join(args.out, ref)))
        else:
            for name in FALLBACK_PROTEINS:
                for fn in (f"{name}.dat", f"{name}_clean.dat"):
                    p = f"{sc}/experiment/{name}/{fn}"
                    downloads.append((p, os.path.join(args.out, p)))
                p = f"{sc}/ensembles/{name}/average_curve.dat"
                downloads.append((p, os.path.join(args.out, p)))

    # -------------------------------------------------------------- download
    n_ok, n_fail = 0, 0
    for remote, local in downloads:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if os.path.exists(local) and os.path.getsize(local) > 0:
            # verify completeness by re-downloading header-free: cheap trick is
            # to just accept non-empty files; use --fresh semantics if desired
            n_ok += 1
            continue
        url = f"{RAW}/{remote}"
        res = robust_get(url, dest=local)
        if res and os.path.getsize(local) > 0:
            n_ok += 1
        else:
            # missing optional files (e.g. no _clean.dat) are fine
            if os.path.exists(local) and os.path.getsize(local) == 0:
                os.remove(local)
            n_fail += 1
            if "_clean.dat" not in local:
                print(f"  [missing] {remote}")

    print(f"\nDone: {n_ok} files present, {n_fail} unavailable (optional "
          f"_clean.dat files count as unavailable if absent upstream).")
    print(f"Data written under: {os.path.abspath(args.out)}")

    # ------------------------------------------------- completeness report
    if want_sc:
        exp_root = os.path.join(args.out, "saxs_scattering", "experiment")
        ref_root = os.path.join(args.out, "saxs_scattering", "ensembles")
        have_exp, have_ref, missing = 0, 0, []
        names = FALLBACK_PROTEINS
        sf = os.path.join(exp_root, "sequences.fasta")
        if os.path.isfile(sf):
            names, nm = [], None
            for line in open(sf):
                line = line.strip()
                if line.startswith(">"):
                    names.append(line[1:].split()[0])
        for nm in names:
            d = None
            for cand in os.listdir(exp_root) if os.path.isdir(exp_root) else []:
                if cand.lower() == nm.lower():
                    d = os.path.join(exp_root, cand)
                    break
            has = False
            if d:
                fs = {f.lower() for f in os.listdir(d)}
                has = (f"{nm.lower()}.dat" in fs
                       or f"{nm.lower()}_clean.dat" in fs
                       or any(f.endswith(".dat") for f in fs))
            if has:
                have_exp += 1
            else:
                missing.append(nm)
            if os.path.isdir(ref_root):
                for cand in os.listdir(ref_root):
                    if cand.lower() == nm.lower() and os.path.isfile(
                            os.path.join(ref_root, cand, "average_curve.dat")):
                        have_ref += 1
                        break
        print(f"[check] proteins with experimental curve: {have_exp}/{len(names)}")
        print(f"[check] proteins with reference FoXS curve: {have_ref}")
        if missing:
            print(f"[check] MISSING experimental curves for: {missing}")
            print("        -> re-run this script; it resumes and only fetches "
                  "what is missing.")


if __name__ == "__main__":
    main()
