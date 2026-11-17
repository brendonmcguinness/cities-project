"""Shared matplotlib style for the paper figures: matplotlib defaults, Arial/Helvetica, inward ticks."""
import matplotlib as mpl
import matplotlib.pyplot as plt


def apply():
    mpl.rcdefaults()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 10, "legend.fontsize": 8.5, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True, "savefig.dpi": 300, "pdf.fonttype": 42})
