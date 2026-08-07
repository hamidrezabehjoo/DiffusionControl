"""merge_ck.py: assemble final C(K) dataset.

K = 2,4,8  from res/ck_a.npz (seed 0)
K = 16     from res/ck_b.npz (seed 0)
K = 32     from res/diagS1_*.npy (seed 1: seed-0 cloud produced a
           deterministic per-mode-WLS fitting artifact at one grid cell,
           t = 0.125, scale 1.05; re-drawn cloud is clean. CRN is used
           within each (nu1, nu2) pair in all cases.)
Writes res/ck_final.npz.
"""
import numpy as np
import gmm_control as gc

tg = np.linspace(0, gc.T, 41)
g2 = gc.g(tg) ** 2
norm = 0.05 * np.sqrt(np.trapezoid(g2, tg))

Ks, Chat, S_all = [], [], []
za = np.load("res/ck_a.npz")
for i, K in enumerate(za["Ks"]):
    Ks.append(K)
    S_all.append(za["S"][i])
zb = np.load("res/ck_b.npz")
i16 = list(zb["Ks"]).index(16)
Ks.append(16)
S_all.append(zb["S"][i16])
S32 = np.stack([np.load("res/diagS1_1.0.npy"), np.load("res/diagS1_1.05.npy")])
Ks.append(32)
S_all.append(S32)

for K, Sk in zip(Ks, S_all):
    dS = Sk[1] - Sk[0]
    Chat.append(np.sqrt(np.trapezoid(dS**2, tg)) / norm)

order = np.argsort(Ks)
Ks = np.array(Ks)[order]
Chat = np.array(Chat)[order]
S_all = np.array(S_all)[order]
np.savez("res/ck_final.npz", Ks=Ks, Chat=Chat, S=S_all, tg=tg)
for k, c in zip(Ks, Chat):
    print(f"K={k}: C_hat={c:.4f}")
m = Chat.mean()
print(f"mean={m:.4f}, spread={100*(Chat.max()-Chat.min())/m:.1f}%")
