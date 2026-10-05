# panel/ — prospective panel screen (OC23 + MS15 [+ TP16])

Screen-first prospective test of the noise controller across the same
two-state benchmarks used by Suzuki & Amagasa (JCIM 2026). Everything is
driven by `targets.csv`; the frozen protocol is in
`prospective_test_panel_protocol.md`.

## Quickstart

```bash
# 1. fetch construct sequences, download + renumber references, write yamls
./panel/run_panel.sh fetch                      # OC23 + MS15 (ready rows)

# 2. stock screen: 100 models x 3 seeds per target + frozen diagnosis
#    (first seed's MSA is frozen to panel_msa/ and reused by all arms)
./panel/run_panel.sh screen                     # or: --set oc23 / --only adk,rfah

# 3. frozen arms: diagnosed targets get the diagnosed AND contra-diagnosed
#    arm (sign-law control); up to 5 primary-stratum healthy targets get both (no-harm test)
./panel/run_panel.sh intervene

# 3b. MSA-depth-engineered boost: all healthy-control targets, depth-32 MSA (frozen);
#      runs only where the degraded STOCK re-diagnoses as boost (manipulation check)
./panel/run_panel.sh degrade

# 4. panel summary by stratum -> panel/panel_summary.csv
./panel/run_panel.sh analyze
```

TP16 (16 transporters, optional phase 2):
`python3 panel/fetch_panel.py --targets panel/targets.csv --include-optional`
then `./panel/run_panel.sh screen --set tp16` etc.

## Files

| file | role |
|------|------|
| `targets.csv` | target registry: refs, UniProt, labels, flags. One row per target. |
| `fetch_panel.py` | UniProt/RCSB download, chain extraction, Needleman–Wunsch renumbering onto the input sequence, yaml writer, sanity checks → `panel/fetch_report.csv` |
| `eval_panel.py` | panel variant of `eval_basins.py`: residue-number intersection matching (references with missing density OK). Same Kabsch/TM/basin rules. `--basin_rule tm` for fold-switchers; `--escape_rmsd auto` scales the basin threshold to min(3.5, 0.5 × ref-pair RMSD) for small-motion targets |
| `panel_diagnose.py` | frozen decision table incl. ceiling clause → `diagnosis_<name>.txt` + `panel/panel_diagnosis.csv` |
| `analyze_panel.py` | per-target success (3/3 seeds + seed-level bootstrap CI; pooled Fisher p descriptive), paired sign-law claim, D1 geometry/reachability diagnostics → `panel/panel_summary.csv` |
| `msa_subsample.py` | deterministic depth-32 subsample of a frozen .a3m (engineered boost arm) |
| `make_manifest.py` | write/verify `panel/MANIFEST.md5` (freeze pinning; on the GPU machine use `--verify`, never re-fetch) |
| `run_panel.sh` | batch driver: fetch / screen / intervene / degrade / analyze |
| `prospective_test_panel_protocol.md` | frozen pre-registration (v2.4, panel-level; git tag panel-v2.5; push tag + GitHub release -> Zenodo DOI recorded in unpinned REGISTRATION.md before screening) |

`eval_basins.py`, `diagnose_stock.py`, `analyze_mbp.py` in the parent
directory are NOT modified; the panel variants exist because panel
references have construct offsets and missing residues that the
byte-validated ADK pipeline does not handle.

## Filling the remaining MS15 rows

Five rows in `targets.csv` are `status=tbd` pending Suzuki Table 2 values
(JCIM 2026, doi:10.1021/acs.jcim.6c02094 — Table 2 is in the main text).
Fill `ref1_pdb/ref1_chain/ref1_label/ref2_pdb/ref2_chain/ref2_label`, set
`status=ready`, then `fetch` + `screen` as usual.
