"""2-D illustration (Figure 1): count-based revival removes the centres of small far groups.

python illustration.py OUT.pdf OUT.json   (plots seed 0; reports far groups kept over seeds 0-9)
The size threshold is forced to 0.1 (the e-means paper's rule; FAISS 1.16 disables revival at d<=2).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FAR = np.array([[26.0, 5.0], [5.0, 26.0], [24.0, 24.0]])


def make(seed, blobs=20, per=2000, far_n=80):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0, 10, (blobs, 2))
    X = mu[np.repeat(np.arange(blobs), per)] + 0.4 * rng.normal(size=(blobs * per, 2))
    F = np.repeat(FAR, far_n, axis=0) + 0.3 * rng.normal(size=(len(FAR) * far_n, 2))
    X = np.vstack([X, F]).astype(np.float32)
    return np.ascontiguousarray(X[rng.permutation(len(X))]), blobs + len(FAR)


def run(X, k, seed, rule):
    from sklearn.cluster import kmeans_plusplus
    from distance_revival import DistanceAwareRevivalEmeans
    init = kmeans_plusplus(X, k, random_state=seed)[0].astype(np.float32)
    m = DistanceAwareRevivalEmeans(2, k, revival=rule, nepochs=5, seed=seed)
    m.train(X, init_centroids=init)
    return init, m.centroids


def kept(C):
    return int(sum(((C - f) ** 2).sum(1).min() < 4.0 for f in FAR))


def main():
    out_pdf, out_json = sys.argv[1], sys.argv[2]
    stats = {"count": [], "near": [], "init": []}
    for seed in range(10):
        X, k = make(seed)
        for rule in ("count", "near"):
            init, C = run(X, k, seed, rule)
            stats[rule].append(kept(C))
        stats["init"].append(kept(init))
    json.dump(stats, open(out_json, "w"))
    print(stats)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, k = make(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for ax, rule, title in zip(axes, ("count", "near"), ("Count-based revival", "Distance-aware revival (ours)")):
        init, C = run(X, k, 0, rule)
        ax.scatter(X[:, 0], X[:, 1], s=0.3, c="0.75", rasterized=True)
        ax.scatter(init[:, 0], init[:, 1], s=22, facecolors="none", edgecolors="C1", lw=0.9,
                   label="k-means++ initialization")
        ax.scatter(C[:, 0], C[:, 1], s=16, c="C0" if rule == "near" else "C3", marker="x", lw=1.3,
                   label="Centres after 5 epochs")
        for f in FAR:
            ax.add_patch(plt.Circle(f, 2.0, fill=False, ls=":", color="k", lw=0.8))
        ax.set_title(f"{title}: {kept(C)} of 3 far groups kept", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    from matplotlib.lines import Line2D
    handles = axes[0].get_legend_handles_labels()[0] + [
        Line2D([], [], ls=":", color="k", lw=0.8, label="Far groups")]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_pdf, dpi=200)


if __name__ == "__main__":
    main()
