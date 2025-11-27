import numpy as np, pandas as pd

# Our 4 signals (paper’s left-limb subset)
CANDIDATE_SETS = [
    ["acc_ls_x","acc_ls_z","gyro_ls_y","gyro_lf_y"],  # long names
    ["ls_ax","ls_az","ls_gy","lf_gy"],               # short aliases
]

def read_and_select(path):
    dfH = pd.read_csv(path, sep=r"[\s,]+", engine="python", header=0, comment="#")
    dfH.columns = [c.strip().lower() for c in dfH.columns]
    cols = set(dfH.columns)
    for cand in CANDIDATE_SETS:
        if all(c in cols for c in cand):
            out = dfH[cand].astype("float32").dropna()
            if cand == ["ls_ax","ls_az","ls_gy","lf_gy"]:
                out.columns = ["acc_ls_x","acc_ls_z","gyro_ls_y","gyro_lf_y"]
            return out
    raise RuntimeError(f"Required columns not found in {path}")

def moving_average_df(df, w=3):
    out = df.copy()
    k = np.ones(w, dtype=np.float32)/float(w)
    for c in out.columns:
        x = out[c].astype("float32").values
        out[c] = np.convolve(x, k, mode="same")
    return out
