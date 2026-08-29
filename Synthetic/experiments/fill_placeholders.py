"""
fill_placeholders.py
====================
Fills the @@TOKEN@@ result markers in ../paper/pnas.tex from
results/d128/seed0/metrics.json, results/d128/seed0/picard_boost.npz and
results/d32/k_independence/ck_final.npz.

Usage (from the repo root):
    python3 experiments/fill_placeholders.py

Re-run after changing the experiments; deterministic given the results.
"""

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
from diffusion_control import gmm_control as gc

PNAS = os.path.abspath(os.path.join(ROOT, "..", "paper", "pnas.tex"))


def main():
    mj = json.load(open(os.path.join(RES, "d128", "seed0", "metrics.json")))
    e = mj["entries"]
    Hp0 = mj["H_p0"]
    bu, bs, br = e["boost_unc"], e["boost_star"], e["reduced"]
    du, ds = e["damp_unc"], e["damp_star"]

    z = np.load(os.path.join(RES, "d128", "seed0", "picard_boost.npz"))
    tg, nu2h, res = z["tg"], z["nu2"], z["res"]
    ratio = np.sqrt(nu2h[-1]) / gc.g(tg)
    peak = ratio[0]
    below = np.where(np.abs(ratio - 1.0) < 0.02)[0]
    decay = float(tg[below[0]]) if len(below) else float("nan")
    kappa = float(np.exp(np.mean(np.diff(np.log(res)))))

    zc = np.load(os.path.join(RES, "d32", "k_independence", "ck_final.npz"))
    C = zc["Chat"]
    cvar = float(np.abs(C - C.mean()).max() / C.mean() * 100)

    gapfrac = (bu["gap"] - bs["gap"]) / bu["gap"] * 100
    redfrac = (br["J"] - bu["J"]) / (bs["J"] - bu["J"]) * 100

    def h(x):
        return f"{x:.2f}"

    def gaprow(star, unc):
        return f"{star['gap']:.2f} ({star['gap'] - unc['gap']:+.2f})"

    def jrow(star, unc):
        return f"{star['J']:.2f} ({star['J'] - unc['J']:+.2f})"

    def sw2row(star, unc):
        return f"{star['sw2']:.3f} ({(star['sw2']/unc['sw2']-1)*100:+.0f}\\%)"

    vals = {
        "HP0": h(Hp0),
        "ALPHA": f"{br['alpha']:g}",
        "TW": f"{br['tw']:g}",
        "BOOSTPEAK": f"{peak:.2f}",
        "BOOSTDECAY": f"{decay:.1f}",
        "BOOSTGAPFRAC": f"{gapfrac:.0f}",
        "REDUCEDFRAC": f"{redfrac:.0f}",
        "OCCSTAR": f"{bs['occ'][7]:.4f}",
        "OCCUNC": f"{bu['occ'][7]:.4f}",
        "KAPPA": f"{kappa:.2f}",
        "CVAR": f"{cvar:.1f}",
        # boost rows
        "BUNCH": h(bu["H"]), "BUNCGAP": f"{bu['gap']:.2f}",
        "BUNCSW2": f"{bu['sw2']:.3f}", "BUNCJ": h(bu["J"]),
        "BREDH": h(br["H"]), "BREDGAP": gaprow(br, bu),
        "BREDSW2": sw2row(br, bu), "BREDJ": jrow(br, bu),
        "BSTARH": h(bs["H"]), "BSTARGAP": gaprow(bs, bu),
        "BSTARSW2": sw2row(bs, bu), "BSTARJ": jrow(bs, bu),
        # damp rows
        "DUNCH": h(du["H"]), "DUNCGAP": f"{du['gap']:.2f}",
        "DUNCSW2": f"{du['sw2']:.3f}", "DUNCJ": h(du["J"]),
        "DSTARH": h(ds["H"]), "DSTARGAP": gaprow(ds, du),
        "DSTARSW2": sw2row(ds, du), "DSTARJ": jrow(ds, du),
    }

    src = open(PNAS).read()
    missing = []
    for tok, v in vals.items():
        marker = f"@@{tok}@@"
        if marker not in src:
            missing.append(tok)
        src = src.replace(marker, v)
    open(PNAS, "w").write(src)

    left = sorted(set(re.findall(r"@@[A-Za-z0-9.-]+@@", src)))
    print(f"filled {len(vals) - len(missing)} tokens in {PNAS}")
    if missing:
        print("tokens with no marker in tex:", missing)
    if left:
        print("WARNING: markers remaining:", left)
    else:
        print("no markers remaining.")


if __name__ == "__main__":
    main()
