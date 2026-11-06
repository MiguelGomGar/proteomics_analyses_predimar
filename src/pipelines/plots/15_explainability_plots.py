"""Draw the SHAP figures computed by pipeline 15.

Each beeswarm needs the whole SHAP explanation object pipeline 15 serialized,
which is reloaded here rather than recomputed. The ranking comparison heatmap
instead reads the cross-model ranking table pipeline 15 already summarized to
CSV.
"""

from pathlib import Path

from src.config import BEST_MODELS, SHAP_TOP_FEATURES
from src.models.model_explainability import load_explanation
from src.utils.filenames import (
    SHAP_BAR_FIGURE,
    SHAP_RANKING_FIGURE,
    SHAP_RANKING_FILE,
    SHAP_SUMMARY_FIGURE,
)
from src.utils.io import read_csv, save_figure
from src.utils.logging_utils import setup_logger
from src.utils.paths import EXPLAINABILITY_DIR
from src.visualization.model_explainability import (
    plot_ranking_comparison,
    plot_shap_bar,
    plot_shap_summary,
)

INPUT_DIR = EXPLAINABILITY_DIR
OUTPUT_DIR = EXPLAINABILITY_DIR

logger = setup_logger(Path(__file__).stem)


# %% Main function
def main() -> None:
    """Draw the SHAP beeswarm summary of each best model and their ranking comparison."""
    show_all_features = SHAP_TOP_FEATURES == -1
    title_suffix = "all predictors" if show_all_features else f"top {SHAP_TOP_FEATURES - 1} predictors"

    for abbreviation in BEST_MODELS:
        logger.info(f"Plotting the SHAP beeswarm of {abbreviation}...")
        explanation = load_explanation(INPUT_DIR, abbreviation)
        save_figure(
            plot_shap_summary(
                explanation,
                title=f"{abbreviation} — SHAP summary ({title_suffix})",
                max_display=None if show_all_features else SHAP_TOP_FEATURES,
            ),
            OUTPUT_DIR / SHAP_SUMMARY_FIGURE.format(model=abbreviation),
        )

        logger.info(f"Plotting the mean |SHAP| barplot of {abbreviation}...")
        save_figure(
            plot_shap_bar(
                explanation,
                title=f"{abbreviation} — mean |SHAP| value ({title_suffix})",
                max_display=None if show_all_features else SHAP_TOP_FEATURES,
            ),
            OUTPUT_DIR / SHAP_BAR_FIGURE.format(model=abbreviation),
        )

    logger.info("Plotting the SHAP importance ranking comparison...")
    ranking_table = read_csv(INPUT_DIR / SHAP_RANKING_FILE)
    save_figure(
        plot_ranking_comparison(ranking_table),
        OUTPUT_DIR / SHAP_RANKING_FIGURE,
    )

    logger.info(f"SHAP figures saved to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
