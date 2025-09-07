import os, csv, math, time, argparse, itertools, random
import numpy as np
from typing import Tuple
from game_env import FlappyEnv, Config
from expert_policy import expert_action

# ---------- util ----------
def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def accuracy(y_true, y_prob, thr=0.5): return np.mean((y_prob >= thr) == y_true)

def poly_features(X: np.ndarray, degree: int) -> np.ndarray:
    if degree <= 1: return X
    # features: x0..x3 -> adiciona termos quadráticos e interações de ordem 2
    n, d = X.shape
    feats = [X]
    # quadráticos
    feats += [X[:, [i]] * X[:, [i]] for i in range(d)]
    # interações
    for i in range(d):
        for j in range(i+1, d):
            feats.append((X[:, [i]] * X[:, [j]]))
    return np.concatenate(feats, axis=1)

def train_logreg_numpy(X, y, lr=0.1, epochs=60) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    # normaliza
    mean = X.mean(axis=0, keepdims=True); std = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mean)/std
    # split train/val
    idx = np.arange(len(Xn)); np.random.shuffle(idx)
    cut = int(0.8*len(Xn))
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr = Xn[tr], y[tr].reshape(-1,1)
    Xva, yva = Xn[va], y[va].reshape(-1,1)
    # pesos
    w = np.zeros((Xtr.shape[1],1), dtype=np.float32); b = 0.0
    for ep in range(1, epochs+1):
        z = Xtr @ w + b; p = sigmoid(z)
        dz = (p - ytr)/len(Xtr)
        dw = Xtr.T @ dz; db = dz.sum()
        w -= lr*dw; b -= lr*db
    # aval
    p_va = sigmoid((Xva @ w + b))
    acc_va = accuracy(yva, p_va)
    return w, b, mean, std, acc_va

def collect_array(episodes=80, gap=150, epsilon=0.1, seed=42):
    random.seed(seed); np.random.seed(seed)
    env = FlappyEnv(Config(pipe_gap=gap, seed=seed))
    X_list = []; y_list = []
    for _ in range(episodes):
        obs, _ = env.reset(); done = False
        while not done:
            a = expert_action(obs)
            if np.random.rand() < epsilon: a = np.random.randint(0,2)
            X_list.append(obs.copy()); y_list.append(a)
            obs, r, done, info = env.step(a)
    X = np.array(X_list, dtype=np.float32); y = np.array(y_list, dtype=np.float32)
    return X, y

# ---------- grid ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # hiperparâmetros (grid leve e eficiente)
    EPISODES = [60, 120, 200]
    GAPS     = [170, 150, 130]        # 170 = fácil; 130 = mais difícil
    EPSILONS = [0.05, 0.15]           # ruído na política (generalização)
    LRS      = [0.05, 0.1]
    EPOCHS   = [60, 100]
    POLY     = [1, 2]                 # grau das features

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run_id","episodes","gap","epsilon","lr","epochs","poly","val_acc","weights_path"])

    best = (-1.0, None)  # (acc, path)
    run_id = 0

    for episodes, gap, epsilon, lr, epochs, poly in itertools.product(
            EPISODES, GAPS, EPSILONS, LRS, EPOCHS, POLY
    ):
        run_id += 1
        print(f"\n[RUN {run_id}] ep={episodes} gap={gap} eps={epsilon} lr={lr} epc={epochs} poly={poly}")
        # dados
        X, y = collect_array(episodes=episodes, gap=gap, epsilon=epsilon, seed=args.seed+run_id)
        if poly > 1:
            X = poly_features(X, degree=poly)
        # treino
        w, b, mean, std, acc_va = train_logreg_numpy(X, y, lr=lr, epochs=epochs)
        weights = {"w":w, "b":b, "mean":mean, "std":std}
        out_path = os.path.join(args.out_dir, f"run_{run_id}_weights.npy")
        np.save(out_path, weights, allow_pickle=True)
        # log
        with open(summary_path, "a", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([run_id, episodes, gap, epsilon, lr, epochs, poly, f"{acc_va:.4f}", out_path])
        print(f"→ val_acc={acc_va:.4f} | weights: {out_path}")

        if acc_va > best[0]:
            best = (acc_va, out_path)

    # salva melhor em nome fixo
    if best[1] is not None:
        best_copy = os.path.join(args.out_dir, "best_weights.npy")
        W = np.load(best[1], allow_pickle=True)
        np.save(best_copy, W, allow_pickle=True)
        print(f"\n✓ Melhor modelo: acc={best[0]:.4f} | salvo em {best_copy}")
        print(f"Resumo dos runs: {summary_path}")
    else:
        print("Nenhum modelo treinado? Verifique o grid.")

if __name__ == "__main__":
    main()
