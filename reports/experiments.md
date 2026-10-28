# Experiments

Generated with `make experiments` (`dvc exp show`). Chosen on **val** (2024); test (2025) shown for reference only.

| Experiment                 | best_iteration   | n_features   | val.model.mae_mw   | val.model.mape_pct   | test.model.mae_mw   | features.weather   | features.holidays   | train.learning_rate   | train.num_leaves   | train.min_child_samples   |
|----------------------------|------------------|--------------|--------------------|----------------------|---------------------|--------------------|---------------------|-----------------------|--------------------|---------------------------|
| workspace                  | 1700             | 15           | 1431.4             | 2.761                | 1445.5              | True               | True                | 0.05                  | 31                 | 20                        |
| main                       | 1700             | 15           | 1431.4             | 2.761                | 1445.5              | True               | True                | 0.05                  | 31                 | 20                        |
| ├── [bigger-trees] | 1262             | 15           | 1435               | 2.769                | 1426                | True               | True                | 0.03                  | 127                | 50                        |
| ├── [no-holidays]  | 691              | 12           | 1702.8             | 3.353                | 1682.4              | True               | False               | 0.05                  | 31                 | 20                        |
| └── [no-weather]   | 884              | 11           | 1524.4             | 2.94                 | 1525.5              | False              | True                | 0.05                  | 31                 | 20                        |

