# Critical Node Detection in Aerial Swarms

AE5510 course project (Problem 2), Haresh M (AE23B114), IIT Madras.
A criticality-score predictor for aerial swarms that uses only agent position and motion:
two-part LightGBM + GraphSAGE (radius graph built from positions) + a small stacker, with an online
one-step-at-a-time pipeline (about 1.7 ms per step).

## Setup
    python -m pip install -r requirements.txt

The dataset (about 880 MB) is not in this repository. Download it from the link in the course project
description and put `swarm_criticality_dataset_100_scenarios.csv` in the project folder.

## Run order
    jupyter nbconvert --execute preprocessing.ipynb          # writes processed/
    python ensemble_kinematic_baselines.py                   # two-part LightGBM score
    python graphsage_critical_node.py --aggregators pool --range 55 --out results_score_graphsage
    python graphsage_critical_node.py --aggregators mean --range 25 --out results_score_graphsage_r25
    python linear_blend.py --gs-dir results_score_graphsage_r25   # stacker, offline metrics
    python deployable_model.py                               # live demo on one test case

| File | Purpose |
|---|---|
| eda.ipynb | exploratory analysis |
| preprocessing.ipynb | cleaning, pruning, scaling, scenario-level 60/20/20 split |
| ml_model_comparison.ipynb | Phase 1 baselines and feature-set ablation |
| swarm_common.py | shared code: loading, history features, two-part LightGBM, metrics |
| ensemble_kinematic_baselines.py | trains the LightGBM score |
| graphsage_critical_node.py | trains the GraphSAGE score |
| linear_blend.py | stacker definition and offline evaluation |
| deployable_model.py | online pipeline on one case; prints and saves the ranked list of critical agents |
