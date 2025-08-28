import pickle as pkl
import numpy as np
from scipy.io import savemat

# ---- CONFIG ----
pkl_file = "./data/mr.train.modularity_adj"   # your existing pickle file
npy_file = "./data/mr_modularity.npy"
npz_file = "./data/mr_modularity.npz"
mat_file = "./data/mr_modularity.mat"

# ---- LOAD (in old env where it works) ----
with open(pkl_file, "rb") as f:
    sim_matrix = pkl.load(f, encoding="latin1")  # latin1 ensures cross-Py2/3 compatibility

print("Loaded sim_matrix:", type(sim_matrix), getattr(sim_matrix, "shape", None))

# ---- SAVE in safe formats ----
np.save(npy_file, sim_matrix)
np.savez(npz_file, matrix=sim_matrix)
savemat(mat_file, {"matrix": sim_matrix})

print(f"✅ Saved as {npy_file}, {npz_file}, {mat_file}")
