"""Phase 3e: SHAP explainability of the three best models (EN, SVM, MLP).

Every earlier phase reports how well the models discriminate, never why they
predict what they predict. This phase derives Shapley additive explanations
from the fitted multimodal pipelines of the same three best-performing models
identified in pipeline 14 (see BEST_MODELS), one attribution matrix per model
over its external validation set.

The fitted pipelines are reloaded as-is from disk; nothing is retrained. The
beeswarm figures are drawn from the explanations saved here by
src/pipelines/plots/15_explainability_plots.py.
"""

from pathlib import Path

from src.config import BEST_MODELS, TARGET_VARIABLE
from src.models.data_preprocessing import encode_target_variable
from src.models.model_explainability import (
    build_global_importance_table,
    build_ranking_comparison_table,
    explain_model,
    save_explanation,
)
from src.utils.filenames import SHAP_RANKING_FILE
from src.utils.io import read_parquet, save_csv
from src.utils.logging_utils import setup_logger
from src.utils.paths import (
    CLEANED_MULTIMODAL_DATA_PATH,
    EXPLAINABILITY_DIR,
    MULTIMODAL_MODELS_DIR,
)

INPUT_FILE = CLEANED_MULTIMODAL_DATA_PATH
MODEL_SOURCE_DIR = MULTIMODAL_MODELS_DIR
OUTPUT_DIR = EXPLAINABILITY_DIR

logger = setup_logger(Path(__file__).stem)


# %% Main function
def main() -> None:
    """Compute and save the SHAP explanation of each of the best models."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading the multimodal dataset from {INPUT_FILE}...")
    df = read_parquet(INPUT_FILE)
    y = encode_target_variable(df, TARGET_VARIABLE)

    importance_by_model = {}
    for abbreviation in BEST_MODELS:
        explanation = explain_model(abbreviation, MODEL_SOURCE_DIR, df, y, logger=logger)
        save_explanation(explanation, OUTPUT_DIR, abbreviation)
        importance_by_model[abbreviation] = build_global_importance_table(explanation, abbreviation)

    save_csv(
        build_ranking_comparison_table(importance_by_model),
        OUTPUT_DIR / SHAP_RANKING_FILE,
    )

    logger.info(f"SHAP explanations saved to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
