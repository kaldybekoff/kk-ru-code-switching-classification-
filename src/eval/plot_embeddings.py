"""Plot an exploratory PCA projection of frozen XLM-R review embeddings."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.eval.density_groups import COUNTS, load_groups
from src.eval.human_review import holdout, references
from src.eval.sprint5 import ROOT
from src.features.representations import TransformerDoc


def main():
    rows = holdout(*references())
    groups = load_groups(COUNTS)
    ids = [cid for cid in sorted(rows) if groups.get(cid)]
    embeddings = TransformerDoc().fit().transform([rows[cid]["text"] for cid in ids])
    pca = PCA(n_components=2, svd_solver="full")
    points = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), layout="constrained")
    colors = {"mono": "#214666", "low": "#008075", "med": "#b55d12", "high": "#8a3855"}
    markers = {"mono": "o", "low": "s", "med": "^", "high": "D"}
    for group, color in colors.items():
        mask = [groups[cid] == group for cid in ids]
        ax.scatter(points[mask, 0], points[mask, 1], color=color, marker=markers[group],
                   label=f"{group} (n={sum(mask)})", s=44, alpha=0.82,
                   edgecolors="white", linewidths=0.5)
    ax.set_title("PCA of frozen XLM-R embeddings", loc="left", fontweight="bold", pad=14)
    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(title="Code-switch density", frameon=False)
    path = ROOT / "reports" / "embedding_pca.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"{len(ids)} comments. Explained variance {explained.sum():.1%}. {path}")


if __name__ == "__main__":
    main()
