"""SHAP explainability for the project's best-performing models.

Every modelling phase elsewhere reports *how well* a model discriminates
(ROC-AUC, PR-AUC, hard metrics) but never *why* it predicts what it predicts.
This module derives Shapley additive explanations from the already-fitted
multimodal pipelines of the three best-performing models (BEST_MODELS), one
patient-level attribution matrix per model, which the plotting pipeline turns
into a beeswarm summary of the top predictors.

Two design decisions shape the whole module.

**Only the classifier is explained, not the whole pipeline.** The preprocessor
is applied once and the SHAP analysis runs on the transformed matrix. Feeding
raw data to the explainers would re-run the IterativeImputer for every
background/foreground combination, which is both slow and stochastic. This is
sound here because the preprocessor encodes categorical variables with an
OrdinalEncoder (one ColumnTransformer lane per column), never one-hot, so
transformed features map one-to-one onto the original predictors.

**Each model is explained in its own native decision score** — log-odds for
the Elastic Net, margin for the SVM, logit(p) for the MLP. Since a
StandardScaler is monotone per feature, the sign and the ranking of the SHAP
values carry over to the original scale, but the magnitudes are not
comparable across models.
"""

import time
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.config import SEED, SHAP_BACKGROUND_SIZE, TEST_SIZE
from src.models.best_model import get_model_feature_columns, load_fitted_pipeline
from src.models.results_saving import _clean_feature_names
from src.utils.filenames import SHAP_EXPLANATION_FILE
from src.utils.io import read_joblib, save_joblib

# Guard for the logit transform of the MLP probabilities: a probability of
# exactly 0 or 1 would map to +/- infinity and poison every SHAP value.
_PROBABILITY_EPSILON = 1e-12


def get_transformed_matrices(fitted_pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Apply a fitted preprocessor to both partitions and name the columns.

    Parameters
    ----------
    fitted_pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with 'preprocessor' and 'clf' steps.
    X_train : pd.DataFrame
        Training predictors, used to build the background set.
    X_test : pd.DataFrame
        External validation predictors, the patients being explained.

    Returns
    -------
    tuple
        (X_train_transformed, X_test_transformed, feature_names), where the
        arrays are 2D numpy matrices and feature_names holds the cleaned
        predictor names in the preprocessor's own output order.

    Notes
    -----
    The output order is *not* the input column order: the ColumnTransformer
    emits the numeric lane first and then one lane per categorical variable.
    Always pair the SHAP values with these names, never with X.columns.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    feature_names = _clean_feature_names(preprocessor.get_feature_names_out())
    return (
        np.asarray(preprocessor.transform(X_train)),
        np.asarray(preprocessor.transform(X_test)),
        feature_names,
    )


def build_background_set(
    X_train_transformed: np.ndarray,
    size: int = SHAP_BACKGROUND_SIZE,
    seed: int = SEED,
    logger: Logger | None = None,
) -> np.ndarray:
    """Draw the reference set every SHAP contribution is measured against.

    A SHAP value answers "how much did this variable move the prediction away
    from what the model would have said knowing nothing about it?", and the
    background set is what defines that "knowing nothing" baseline.

    Parameters
    ----------
    X_train_transformed : np.ndarray
        Transformed training matrix.
    size : int, default SHAP_BACKGROUND_SIZE
        Number of training patients to sample.
    seed : int, default SEED
        Random seed, making the sample reproducible across runs.
    logger : logging.Logger, optional
        Logger for the size report.

    Returns
    -------
    np.ndarray
        Background matrix of shape (min(size, n_train), n_features).
    """
    n_train = X_train_transformed.shape[0]
    if size >= n_train:
        if logger:
            logger.warning(
                f"Requested a background set of {size} patients but only {n_train} "
                f"training patients are available; using all of them."
            )
        return X_train_transformed
    return np.asarray(shap.sample(X_train_transformed, size, random_state=seed))


def _mlp_logit_output(clf):
    """Wrap an MLP's predicted probability as a log-odds function.

    The MLP is explained on the logit scale rather than on the probability
    scale, consistent with the linear models being explained in log-odds /
    margin space rather than in probability space.

    Parameters
    ----------
    clf : sklearn.neural_network.MLPClassifier
        Fitted classifier.

    Returns
    -------
    callable
        Function mapping a transformed matrix to a 1D array of log-odds.
        Probabilities are clipped away from 0 and 1 before the transform.
    """

    def logit_output(X):
        probability = clf.predict_proba(X)[:, 1]
        probability = np.clip(probability, _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
        return np.log(probability / (1 - probability))

    return logit_output


def get_explained_output(fitted_pipeline, abbreviation: str):
    """Return the model output function the SHAP values will decompose.

    Parameters
    ----------
    fitted_pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with a 'clf' step.
    abbreviation : str
        Model abbreviation ('EN', 'SVM' or 'MLP').

    Returns
    -------
    tuple
        (output_function, output_label). The label names the score space
        ('log-odds', 'SVM margin' or 'logit(p)') for logs.
    """
    clf = fitted_pipeline.named_steps["clf"]
    if abbreviation == "MLP":
        return _mlp_logit_output(clf), "logit(p)"
    label = "SVM margin" if abbreviation == "SVM" else "log-odds"
    return clf.decision_function, label


def build_explainer(fitted_pipeline, abbreviation: str, background: np.ndarray, logger: Logger | None = None):
    """Build the most precise SHAP explainer available for a fitted model.

    The choice is driven by what each model actually is:

    - **EN** (LogisticRegression) and **SVM** *when its tuned kernel is
      linear*: the decision function is linear in the transformed features, so
      LinearExplainer gives the exact Shapley values in closed form, instantly.
      The coefficient/intercept pair is passed explicitly rather than the
      estimator, so the result does not depend on shap's introspection of
      sklearn internals.
    - **SVM with a non-linear kernel**: falls back to KernelExplainer. Not the
      current case (the tuned multimodal SVM uses a linear kernel) but kept so
      a future re-tuning that selects RBF still works.
    - **MLP**: KernelExplainer. DeepExplainer is *not* an option: it implements
      DeepLIFT-style backpropagation through the graph of an automatic
      differentiation framework, whereas sklearn's MLPClassifier is plain
      NumPy with no such graph and no shap bridge. KernelExplainer is
      model-agnostic and, at this problem size, cheap.

    Parameters
    ----------
    fitted_pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with 'preprocessor' and 'clf' steps.
    abbreviation : str
        Model abbreviation ('EN', 'SVM' or 'MLP').
    background : np.ndarray
        Reference set from build_background_set.
    logger : logging.Logger, optional
        Logger for the explainer choice.

    Returns
    -------
    shap.Explainer
        The chosen explainer, already bound to the background set.
    """
    clf = fitted_pipeline.named_steps["clf"]
    is_linear = abbreviation == "EN" or (abbreviation == "SVM" and clf.kernel == "linear")

    if is_linear:
        coefficients = np.asarray(clf.coef_).ravel()
        intercept = float(np.asarray(clf.intercept_).ravel()[0])
        if logger:
            logger.info(
                f"[{abbreviation}] Using LinearExplainer (exact) over a background of "
                f"{background.shape[0]} patients."
            )
        # An Independent masker is the current spelling of the interventional
        # perturbation: features are masked by substituting values drawn from
        # the background independently of the rest of the row. It matches what
        # KernelExplainer does for the MLP, keeping the models comparable.
        masker = shap.maskers.Independent(background)
        return shap.LinearExplainer((coefficients, intercept), masker)

    output_function, output_label = get_explained_output(fitted_pipeline, abbreviation)
    if logger:
        reason = (
            f"kernel='{clf.kernel}' is not linear"
            if abbreviation == "SVM"
            else "sklearn's MLPClassifier has no autodiff graph for DeepExplainer"
        )
        logger.info(
            f"[{abbreviation}] Using KernelExplainer (approximate) on {output_label}: {reason}."
        )
    return shap.KernelExplainer(output_function, background)


def compute_shap_values(
    explainer, X_test_transformed: np.ndarray, feature_names: list[str], logger: Logger | None = None
) -> shap.Explanation:
    """Compute the SHAP values and wrap them in a shap.Explanation.

    Normalizes the two output conventions shap uses: LinearExplainer returns a
    single 2D array, while KernelExplainer may return a list of one array per
    class for a classifier-like output. Anything other than a
    (n_samples, n_features) matrix is reduced to one before returning.

    KernelExplainer's L1 feature selection is switched off. Its default,
    ``l1_reg="num_features(10)"``, keeps only the ten features it deems most
    relevant *per patient* and assigns exactly zero to the rest, which would
    silently truncate the beeswarm's ranking.

    Parameters
    ----------
    explainer : shap.Explainer
        Explainer from build_explainer.
    X_test_transformed : np.ndarray
        Transformed external validation matrix.
    feature_names : list of str
        Predictor names, aligned to the columns of X_test_transformed.
    logger : logging.Logger, optional
        Logger for the elapsed time.

    Returns
    -------
    shap.Explanation
        Explanation carrying .values, .base_values, .data and .feature_names,
        ready for shap.plots.beeswarm.
    """
    keyword_arguments = {"l1_reg": 0} if isinstance(explainer, shap.KernelExplainer) else {}

    start = time.perf_counter()
    values = explainer.shap_values(X_test_transformed, **keyword_arguments)
    elapsed = time.perf_counter() - start

    values = np.asarray(values)
    if values.ndim == 3:
        # (n_samples, n_features, n_outputs): keep the positive class.
        values = values[:, :, -1]

    base_value = float(np.asarray(explainer.expected_value).ravel()[-1])
    if logger:
        logger.info(
            f"Computed SHAP values for {values.shape[0]} patients x "
            f"{values.shape[1]} features in {elapsed:.1f}s (base value {base_value:.4f})."
        )

    return shap.Explanation(
        values=values,
        base_values=np.full(values.shape[0], base_value),
        data=X_test_transformed,
        feature_names=list(feature_names),
    )


def explain_model(
    abbreviation: str,
    model_dir: str | Path,
    df: pd.DataFrame,
    y: pd.Series,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    background_size: int = SHAP_BACKGROUND_SIZE,
    logger: Logger | None = None,
) -> shap.Explanation:
    """Compute the SHAP explanation of one fitted pipeline's external validation set.

    Parameters
    ----------
    abbreviation : str
        Model abbreviation, matching the ``optimized_{abbreviation}.joblib``
        filename saved by ``results_saving.save_model``.
    model_dir : str or Path
        Output directory of the modelling phase the pipeline was fitted in.
    df : pd.DataFrame
        Cleaned dataset (predictors + target columns).
    y : pd.Series
        Encoded binary target, aligned to df's index.
    seed : int, default SEED
        Random seed for the external validation split and the background sample.
    test_size : float, default TEST_SIZE
        Size of the external validation split.
    background_size : int, default SHAP_BACKGROUND_SIZE
        Number of training patients sampled as the SHAP reference set.
    logger : logging.Logger, optional
        Logger for progress reporting.

    Returns
    -------
    shap.Explanation
        Per-patient SHAP values for every predictor of the external validation
        set, ready for shap.plots.beeswarm.
    """
    if logger:
        logger.info(f"Loading the fitted {abbreviation} pipeline from {model_dir}...")
    model = load_fitted_pipeline(model_dir, abbreviation)
    feature_columns = get_model_feature_columns(model)

    X = df[feature_columns]
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y
    )

    X_train_transformed, X_test_transformed, feature_names = get_transformed_matrices(
        model, X_train, X_test
    )
    background = build_background_set(
        X_train_transformed, size=background_size, seed=seed, logger=logger
    )

    explainer = build_explainer(model, abbreviation, background, logger=logger)
    return compute_shap_values(explainer, X_test_transformed, feature_names, logger=logger)


def build_global_importance_table(explanation: shap.Explanation, abbreviation: str) -> pd.DataFrame:
    """Rank predictors by mean absolute SHAP value for one model.

    Parameters
    ----------
    explanation : shap.Explanation
        Explanation from compute_shap_values.
    abbreviation : str
        Model abbreviation, recorded in the 'Model' column.

    Returns
    -------
    pd.DataFrame
        Columns ['Model', 'Feature', 'Mean_Abs_SHAP', 'Mean_SHAP',
        'Importance_Share', 'Rank'], sorted by importance. 'Mean_SHAP' keeps
        the sign, showing the average direction of the effect, while
        'Importance_Share' normalizes the magnitude to sum to 1 and is the
        quantity that can be compared across models.
    """
    values = np.asarray(explanation.values)
    mean_absolute = np.abs(values).mean(axis=0)
    total = mean_absolute.sum()

    table = pd.DataFrame(
        {
            "Model": abbreviation,
            "Feature": list(explanation.feature_names),
            "Mean_Abs_SHAP": mean_absolute,
            "Mean_SHAP": values.mean(axis=0),
            "Importance_Share": mean_absolute / total if total else 0.0,
        }
    )
    table = table.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    table["Rank"] = np.arange(1, len(table) + 1)
    return table


def build_ranking_comparison_table(importance_by_model: dict) -> pd.DataFrame:
    """Put the per-model importance rankings side by side.

    Parameters
    ----------
    importance_by_model : dict
        Mapping of model abbreviation to its global importance table, from
        build_global_importance_table.

    Returns
    -------
    pd.DataFrame
        One row per predictor with a 'Rank_{model}' and a 'Share_{model}'
        column per model, plus 'Mean_Rank', sorted by 'Mean_Rank'.
    """
    merged = None
    for abbreviation, table in importance_by_model.items():
        columns = table[["Feature", "Rank", "Importance_Share"]].rename(
            columns={"Rank": f"Rank_{abbreviation}", "Importance_Share": f"Share_{abbreviation}"}
        )
        merged = columns if merged is None else merged.merge(columns, on="Feature", how="outer")

    rank_columns = [column for column in merged.columns if column.startswith("Rank_")]
    merged["Mean_Rank"] = merged[rank_columns].mean(axis=1)
    return merged.sort_values("Mean_Rank").reset_index(drop=True)


def save_explanation(explanation: shap.Explanation, output_dir: str | Path, abbreviation: str) -> Path:
    """Persist a SHAP explanation so the figures can be redrawn without recomputing it.

    Parameters
    ----------
    explanation : shap.Explanation
        Explanation returned by explain_model.
    output_dir : str or Path
        Directory where shap_explanation_{abbreviation}.joblib is written.
    abbreviation : str
        Model abbreviation, e.g. 'EN'.

    Returns
    -------
    Path
        Path where the explanation was saved.
    """
    file_path = Path(output_dir) / SHAP_EXPLANATION_FILE.format(model=abbreviation)
    return save_joblib(explanation, file_path)


def load_explanation(input_dir: str | Path, abbreviation: str) -> shap.Explanation:
    """Read back the SHAP explanation saved by save_explanation.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing shap_explanation_{abbreviation}.joblib.
    abbreviation : str
        Model abbreviation, e.g. 'EN'.

    Returns
    -------
    shap.Explanation
        The deserialized explanation.
    """
    file_path = Path(input_dir) / SHAP_EXPLANATION_FILE.format(model=abbreviation)
    return read_joblib(file_path)
