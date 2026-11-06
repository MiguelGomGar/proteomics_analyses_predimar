"""Plot reporting the SHAP explainability analysis of the best models.

Follows the same contract as the other visualization modules: the function
builds and returns a matplotlib Figure, saving is left to the caller
(utils.io.save_figure). It delegates to shap's own beeswarm renderer, which
draws onto the active figure and returns nothing, so it is called with
show=False and the current figure is captured afterwards.
"""

import matplotlib

# Select a non-interactive backend before pyplot is imported, for the same
# reason as in the other visualization modules: figures are rendered in a loop.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

from src.config import MODEL_BAR_COLOR
from src.visualization.plot_utils import _style_axes_v


def plot_ranking_comparison(ranking_table: pd.DataFrame, figsize: tuple = (8, 14)) -> plt.Figure:
    """Show where the models agree and disagree on predictor importance.

    Heatmap of ranks (1 = most important) for every predictor, ordered from
    most to least important on average. A row of similar shades means the
    models concur; a row that jumps from dark to light means one model relies
    on a predictor the others ignore.

    Parameters
    ----------
    ranking_table : pd.DataFrame
        Table from model_explainability.build_ranking_comparison_table.
    figsize : tuple, default (8, 14)
        Figure size in inches. Tall by default since every predictor is shown.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rank_columns = [column for column in ranking_table.columns if column.startswith("Rank_")]
    matrix = ranking_table.set_index("Feature")[rank_columns]
    matrix.columns = [column.replace("Rank_", "") for column in rank_columns]

    figure, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="viridis_r",
        cbar_kws={"label": "Importance rank (1 = most important)"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("SHAP importance rank by model", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    figure.tight_layout()
    return figure


def plot_shap_summary(explanation, title: str, max_display: int | None = None) -> plt.Figure:
    """Draw the SHAP beeswarm summary for one model.

    Each point is one patient. Horizontal position is that patient's SHAP value
    for the predictor, and color encodes whether their value of the predictor
    was high or low, so the plot shows magnitude *and* direction of effect at a
    glance. shap.plots.beeswarm ranks predictors by mean absolute SHAP value
    and keeps only the top ``max_display``.

    Parameters
    ----------
    explanation : shap.Explanation
        Explanation from model_explainability.explain_model.
    title : str
        Figure title.
    max_display : int or None, default None
        Number of top predictors shown, most important first. None shows
        every predictor, matching shap's own convention for disabling the
        display cap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    figure = plt.gcf()
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.tight_layout()
    return figure


def plot_shap_bar(explanation, title: str, max_display: int | None = None) -> plt.Figure:
    """Draw the mean absolute SHAP value barplot for one model.

    Each bar is one predictor's mean absolute SHAP value across every patient
    in the external validation set, i.e. its average contribution magnitude
    regardless of direction. Unlike the beeswarm, this collapses each
    predictor to a single global importance number, which is what
    build_global_importance_table also reports as 'Mean_Abs_SHAP'.
    shap.plots.bar ranks predictors by this value and keeps only the top
    ``max_display``.

    Parameters
    ----------
    explanation : shap.Explanation
        Explanation from model_explainability.explain_model.
    title : str
        Figure title.
    max_display : int or None, default None
        Number of top predictors shown, most important first. None shows
        every predictor, matching shap's own convention for disabling the
        display cap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    shap.plots.bar(explanation, max_display=max_display, show=False)
    figure = plt.gcf()
    ax = plt.gca()

    # shap draws every bar and its value label in its own red/blue palette;
    # override both to the project's neutral clinical color and unify the
    # grid/spine style with the rest of the project's horizontal barplots.
    for patch in ax.patches:
        patch.set_facecolor(MODEL_BAR_COLOR)
        patch.set_edgecolor("#2c3e50")
        patch.set_linewidth(0.5)
    for text in ax.texts:
        text.set_color("#2c3e50")

    _style_axes_v(ax)
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.tight_layout()
    return figure
