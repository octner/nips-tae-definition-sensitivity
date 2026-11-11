"""Draw the two panels of the figure on label shares in Section 4.

The code names a model "arm", a concept "construct" and a prompt
"instrument". The ID impl_0 marks the original wording of a definition.

The script reads the label shares in tables/t1_verdict_matrix.csv and writes
these two PDF files to figs/:

    - f2_include_by_definition.pdf  the INCLUDE share of each protest
                                    definition, for each model
    - f3_ccc_2021_2024.pdf          the change of the INCLUDE share from
                                    CCC 2021 to CCC 2024, for each model

Run from the repository root.

    python3 code/an_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _paths import FIGS, MODELS, TABLES

MODEL_STYLE = {
    "gemma4_31b":     dict(color="#2a78d6", marker="o", label="Gemma"),
    "qwen36_27b":     dict(color="#eb6834", marker="s", label="Qwen"),
    "claude_sonnet5": dict(color="#1baf7a", marker="^", label="Sonnet"),
}
DEF_LABEL = {"acled": "ACLED", "ccc_2021": "CCC 2021", "ccc_2024": "CCC 2024",
             "cnts_agd": "CNTS", "mm": "MMPD"}

plt.rcParams.update({
    "font.size": 8.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "figure.dpi": 200,
})


def _grid(ax):
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def f2(t1):
    """The INCLUDE share of each protest definition, for each model."""
    sub = t1[(t1.construct == "protest")
             & t1.instrument.str.endswith("impl_0")].copy()
    sub["def_id"] = sub.instrument.str.replace("__impl_0", "", regex=False)
    order = (sub.groupby("def_id").INCLUDE.mean()
             .sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(3.15, 2.4))
    for model in MODELS:
        st = MODEL_STYLE[model]
        d = sub[sub.arm == model].set_index("def_id").reindex(order)
        ax.plot(d.INCLUDE, range(len(order)), st["marker"],
                color=st["color"], ms=5, mew=0.8, mec="white",
                label=st["label"], zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DEF_LABEL[d] for d in order])
    ax.set_xlabel("INCLUDE share, same 3,249 texts")
    ax.set_xlim(0, None)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right",
              handletextpad=0.2, borderaxespad=0.2)
    _grid(ax)
    fig.tight_layout()
    return fig


def f3(t1):
    """The change of the INCLUDE share from CCC 2021 to CCC 2024."""
    sub = t1[(t1.construct == "protest")
             & t1.instrument.isin(["ccc_2021__impl_0", "ccc_2024__impl_0"])]
    fig, ax = plt.subplots(figsize=(3.15, 1.9))
    for k, model in enumerate(MODELS):
        st = MODEL_STYLE[model]
        d = sub[sub.arm == model].set_index("instrument")
        a = d.loc["ccc_2021__impl_0", "INCLUDE"]
        b = d.loc["ccc_2024__impl_0", "INCLUDE"]
        y = len(MODELS) - 1 - k
        ax.annotate("", xy=(b, y), xytext=(a, y),
                    arrowprops=dict(arrowstyle="-|>", color=st["color"],
                                    lw=1.4, mutation_scale=9))
        ax.plot(a, y, st["marker"], color=st["color"], ms=5,
                mec="white", mew=0.8)
        ax.text(max(a, b) + 0.012, y, f"{100 * (b - a):+.1f} pts",
                va="center", fontsize=7.5, color="#444444")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_STYLE[m]["label"] for m in reversed(MODELS)])
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.set_xlim(sub.INCLUDE.min() - 0.04, sub.INCLUDE.max() + 0.09)
    ax.set_xlabel("INCLUDE share, CCC 2021 → 2024", fontsize=8)
    _grid(ax)
    fig.tight_layout()
    return fig


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    t1 = pd.read_csv(TABLES / "t1_verdict_matrix.csv")
    for name, fig in (("f2_include_by_definition", f2(t1)),
                      ("f3_ccc_2021_2024", f3(t1))):
        fig.savefig(FIGS / f"{name}.pdf")
        plt.close(fig)
        print(f"[fig] {FIGS / name}.pdf")


if __name__ == "__main__":
    main()
