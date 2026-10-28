# Random Forest Exercise

Random forest classification of the 2-D `make_moons` dataset (1500 points, noise 0.23, 50/50 train/test split).

## Scripts

- `exercise_rf.py`: trains a single forest and plots its decision boundary.
  ```
  python exercise_rf.py --num_trees 32 --tree_depth 5
  ```
- `grid_search_rf.py`: grid search over number of trees (1–128) and max. depth (1–None) with 5-fold cross-validation on the training data. Plots training/CV accuracy heatmaps and the decision boundaries of two forests:
  - **Best**: highest mean CV accuracy.
  - **Shallowest within 1 SE**: shallowest forest (with at least `--min_trees` trees) whose CV accuracy is within one standard error of the best (one-standard-error rule).
  ```
  python grid_search_rf.py --seed 0 --min_trees 32
  ```

Plots are saved to `random_forest_plots/`.

## Results

| Forest | Trees | Depth | CV acc. | Train acc. | Test acc. |
|---|---|---|---|---|---|
| Best | 128 | 8 | 94.1% | 98.8% | 95.3% |
| Shallowest within 1 SE | 32 | 5 | 93.6% | 96.8% | 94.5% |

(seed 0)

**Chosen hyper-parameters: 32 trees, max. depth 5.** Shallow trees underfit (depth ≤ 3: ~91%), deep trees overfit (training accuracy → 100%, CV stays ~94%). More trees reduce variance but don't cause overfitting; beyond ~32 trees the gain is negligible. Depth 5 reaches nearly the best accuracy with a smaller train/test gap and is the consistent choice across seeds (test accuracy 94–95%).
