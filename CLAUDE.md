# Instrucciones

## Contexto

This project is my master's thesis about predicting atrial fibrillation (AF) recurrence with machine learning (ML) models within the framework of the PREDIMAR trial.

## Hipótesis y objetivos

Hypothesis: ML models trained in clinical and proteomic data should predict FA recurrence better than clinical risk scores in literature.

Overall aim: train, tune and evaluate ML models to predict AF recurrence.

Specific aims:

- To determine whether ML models can improve the performance of published AF recurrence risk scores based on the same clinical and lifestyle information.
- To investigate whether baseline circulating proteome adds predictive value to clinical information in the prediction of AF recurrence.
- To identify proteins that may be considered biomarkers for AF recurrence risk.

## Especificaciones

Python code run in conda environment located in the following absolute path: `C:/Users/Miguel/miniconda3/envs/tfm`.

Modules are located within the `/src/` subdirectory, grouped by domain:

- `/src/data/`: data cleaning and transformation.
- `/src/visualization/`: plots generation.
- `/src/models/`: training, optimization and evaluation of ML models.
- `/src/utils/`: generic helpers (filepaths definitions, functions to load and save results...).
- `/src/pipelines/`: modules that run data analyses and models training.
- `/src/config.py`: general configuration (constants).
