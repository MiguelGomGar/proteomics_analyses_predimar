"""Consolidate the modelling results into publication-ready Excel workbooks.

Each workbook groups one family of tables as multiple sheets, with every
table itself a multi-indexed, richly formatted pandas table (not a plain
CSV) so several metrics/partitions/parameters can be read at a glance:

- performance_tables.xlsx: one sheet per modelling phase, rows = models,
  columns = (Metric, Partition).
- comparison_tables.xlsx: one sheet per modality/filtering comparison, rows =
  models, columns = (Metric, {baseline, comparison, Delta}).
- hyperparameters_tables.xlsx: a single sheet comparing every modelling
  phase, rows = (Model, Parameter), columns = (Value, Phase).
- clinical_coefficients.xlsx / protein_coefficients.xlsx: a single sheet
  each comparing the Elastic Net coefficients across the phases that fit
  clinical predictors / proteins, rows = Feature, columns = (Coefficient,
  Phase).
- table1.xlsx: the clinical descriptive Table 1 (pipeline 06), rows =
  (Variable, Level), columns = (AF recurrence group, {Overall, no, yes,
  P-Value}).
"""

from pathlib import Path

from src.config import (
    CLINICAL_COEFFICIENT_PHASES,
    COMPARISON_PHASES,
    PERFORMANCE_PHASES,
    PROTEIN_COEFFICIENT_PHASES,
    PUBLICATION_METRICS,
    PUBLICATION_TABLES_DECIMALS,
)
from src.models.feature_selection import load_previously_kept_features
from src.utils.logging_utils import setup_logger
from src.utils.paths import (
    CLINICAL_MODELS_FILTERED_DIR,
    PROTEOMIC_MODELS_DIR,
    PUBLICATION_TABLES_DIR,
    TABLE1_PATH,
)
from src.models.results_saving import (
    build_coefficients_tables,
    build_comparison_tables,
    build_hyperparameters_tables,
    build_performance_tables,
    build_table1_workbook,
)

logger = setup_logger(Path(__file__).stem)

PERFORMANCE_WORKBOOK = PUBLICATION_TABLES_DIR / "performance_tables.xlsx"
COMPARISON_WORKBOOK = PUBLICATION_TABLES_DIR / "comparison_tables.xlsx"
HYPERPARAMETERS_WORKBOOK = PUBLICATION_TABLES_DIR / "hyperparameters_tables.xlsx"
CLINICAL_COEFFICIENTS_WORKBOOK = PUBLICATION_TABLES_DIR / "clinical_coefficients.xlsx"
PROTEIN_COEFFICIENTS_WORKBOOK = PUBLICATION_TABLES_DIR / "protein_coefficients.xlsx"
TABLE1_WORKBOOK = PUBLICATION_TABLES_DIR / "table1.xlsx"


# %% Main function
def main() -> None:
    """Write the three publication-ready Excel workbooks."""
    PUBLICATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    build_performance_tables(
        PERFORMANCE_PHASES,
        PUBLICATION_METRICS,
        PERFORMANCE_WORKBOOK,
        decimals=PUBLICATION_TABLES_DECIMALS,
        logger=logger,
    )
    build_comparison_tables(
        COMPARISON_PHASES,
        PUBLICATION_METRICS,
        COMPARISON_WORKBOOK,
        decimals=PUBLICATION_TABLES_DECIMALS,
        logger=logger,
    )
    build_hyperparameters_tables(
        PERFORMANCE_PHASES, HYPERPARAMETERS_WORKBOOK, logger=logger
    )

    clinical_feature_universe = load_previously_kept_features(
        CLINICAL_MODELS_FILTERED_DIR
    )
    build_coefficients_tables(
        CLINICAL_COEFFICIENT_PHASES,
        clinical_feature_universe,
        CLINICAL_COEFFICIENTS_WORKBOOK,
        sheet_name="Clinical coefficients",
        decimals=PUBLICATION_TABLES_DECIMALS,
        order_by_phase="Clinical filtered",
        logger=logger,
    )

    protein_feature_universe = load_previously_kept_features(PROTEOMIC_MODELS_DIR)
    build_coefficients_tables(
        PROTEIN_COEFFICIENT_PHASES,
        protein_feature_universe,
        PROTEIN_COEFFICIENTS_WORKBOOK,
        sheet_name="Protein coefficients",
        decimals=PUBLICATION_TABLES_DECIMALS,
        logger=logger,
    )

    build_table1_workbook(TABLE1_PATH, TABLE1_WORKBOOK, logger=logger)

    logger.info(f"Publication tables saved to {PUBLICATION_TABLES_DIR}.")


if __name__ == "__main__":
    main()
