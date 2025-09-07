import argparse
import numpy as np
import pandas as pd

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def accuracy(y_true, y_pred):
    return np.mean((y_pred >= 0.5) == y_true)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data.csv")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--save", type=str, default="weights.npy")
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df[["y_norm","vy_norm","dist_norm","delta_gap_norm"]].values.astype(np.float32)
    y = df["action"].values.astype(np.float32).reshape(-1,1)

    # normalização simples (já estão razoavelmente normalizadas, mas padronizamos vy_norm e delta_gap_norm)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mean) / std

    # split treino/val
    n = len(Xn)
    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(n*(1.0-args.val_split))
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr = Xn[tr], y[tr]
    Xva, yva = Xn[va], y[va]

    # pesos (w, b)
    w = np.zeros((Xtr.shape[1], 1), dtype=np.float32)
    b = 0.0

    for ep in range(1, args.epochs+1):
        # forward
        z = Xtr @ w + b
        p = sigmoid(z)
        # loss BCE
        eps = 1e-8
        loss = -(ytr*np.log(p+eps) + (1-ytr)*np.log(1-p+eps)).mean()
        # grad
        dz = (p - ytr) / len(Xtr)
        dw = Xtr.T @ dz
        db = dz.sum()
        # step GD
        w -= args.lr * dw
        b -= args.lr * db

        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            p_tr = sigmoid(Xtr @ w + b)
            p_va = sigmoid(Xva @ w + b)
            acc_tr = accuracy(ytr, p_tr)
            acc_va = accuracy(yva, p_va)
            print(f"[{ep:03d}] loss={loss:.4f} acc_tr={acc_tr:.3f} acc_va={acc_va:.3f}")

    # salvar pesos + normalização (para uso na inferência)
    np.save(args.save, {"w":w, "b":b, "mean":mean, "std":std}, allow_pickle=True)
    print(f"pesos salvos em {args.save}")

if __name__ == "__main__":
    main()
