"""Grid search for the decision forest toy example.

Searches over the number of trees and the maximum tree depth with cross-validation
on the training data, visualizes the results as heatmaps and evaluates the best
forest on the held-out test data.

Besides the forest with the highest cross-validation accuracy, the forest with the
shallowest trees whose accuracy lies within one standard error of the best one is
selected (one-standard-error rule). Its accuracy is statistically indistinguishable
from the best one, but it is less prone to fitting noise. Only the tree depth counts
as complexity, since more trees reduce variance instead of causing overfitting. A
minimum number of trees keeps the selected forest stable.
"""

import argparse
import datetime
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.ensemble as sk_ensemble
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


def plot_heatmap(ax, scores: np.ndarray, num_trees: list, tree_depths: list, title: str, vmin: float):
    """Plots a (depth x trees) accuracy matrix as an annotated heatmap."""
    im = ax.imshow(scores, cmap="viridis", vmin=vmin, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(num_trees)))
    ax.set_xticklabels(num_trees)
    ax.set_yticks(range(len(tree_depths)))
    ax.set_yticklabels(["None" if d is None else d for d in tree_depths])
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Max. tree depth")
    ax.set_title(title)

    # annotate each cell with its accuracy
    threshold = vmin + (1.0 - vmin) / 2
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            color = "black" if scores[i, j] > threshold else "white"
            ax.text(j, i, "{0:.1%}".format(scores[i, j]), ha="center", va="center", color=color, fontsize=8)

    return im


def select_one_standard_error(cv_results: dict, cv_folds: int, min_trees: int):
    """Selects the shallowest forest whose mean CV accuracy is within one standard error of the best.

    Only forests with at least min_trees trees are considered. Among equally deep forests,
    the one with the fewest trees is selected.
    """
    means = cv_results["mean_test_score"]
    std_errors = cv_results["std_test_score"] / np.sqrt(cv_folds)
    best_idx = int(np.argmax(means))
    threshold = means[best_idx] - std_errors[best_idx]

    params = cv_results["params"]
    candidates = [idx for idx in range(len(means))
                  if means[idx] >= threshold and params[idx]["n_estimators"] >= min_trees]
    if not candidates:
        raise ValueError("No forest with at least {0:d} trees lies within one standard error of the best".format(
            min_trees))

    # shallowest trees first (unlimited depth counts as deepest), then fewest trees
    simplest_idx = min(candidates, key=lambda idx: (
        np.inf if params[idx]["max_depth"] is None else params[idx]["max_depth"],
        params[idx]["n_estimators"]))

    return simplest_idx, threshold, std_errors[best_idx]


def mark_cell(ax, num_trees: list, tree_depths: list, params: dict, color: str, label: str):
    """Draws a frame around the heatmap cell of the given parameters."""
    x = num_trees.index(params["n_estimators"])
    y = tree_depths.index(params["max_depth"])
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=color, linewidth=3, label=label))


def plot_decision_boundary(ax, forest, features, feat_train, labels_train, feat_test, labels_test, title: str):
    """Plots the decision boundary of a fitted forest together with the data points."""
    h = 0.02  # step size in the mesh
    x_min, x_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
    y_min, y_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    prob_boundary = forest.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, prob_boundary, cmap=plt.get_cmap("RdBu"), alpha=0.8)

    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax.scatter(feat_train[:, 0], feat_train[:, 1], c=labels_train, cmap=cm_bright, edgecolors="k", label="Train")
    ax.scatter(feat_test[:, 0], feat_test[:, 1], c=labels_test, cmap=cm_bright, alpha=0.6, edgecolors="k",
               marker="P", label="Test")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)
    ax.legend(loc="lower left")


def main(save_fig: bool, show_fig: bool, result_dir: str, seed: int, cv_folds: int, min_trees: int):
    """Runs a grid search over the forest hyper-parameters on the moons dataset."""

    # generate result directory
    os.makedirs(result_dir, exist_ok=True)

    # same data as in exercise_rf.py, but with a fixed seed to make runs comparable
    features, labels = make_moons(n_samples=1500, noise=0.23, random_state=seed)

    # split into training and testing data
    feat_train, feat_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.5, random_state=42
    )

    # hyper-parameter grid
    num_trees = [1, 2, 4, 8, 16, 32, 64, 128]
    tree_depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, None] # None: no limit, splitting until all nodes are pure or too small
    param_grid = {"n_estimators": num_trees, "max_depth": tree_depths}

    forest = sk_ensemble.RandomForestClassifier(max_features=feat_train.shape[1], random_state=seed)

    # cross-validated grid search on the training data only, the test data stays untouched
    print("Grid search over {0:d} combinations with {1:d}-fold cross-validation...".format(
        len(num_trees) * len(tree_depths), cv_folds))
    search = GridSearchCV(forest, param_grid, cv=cv_folds, scoring="accuracy", return_train_score=True, n_jobs=-1)
    search.fit(feat_train, labels_train)

    # reshape the results into (depth x trees) matrices for the heatmaps
    shape = (len(tree_depths), len(num_trees))
    train_scores = np.zeros(shape)
    cv_scores = np.zeros(shape)
    for params, train_score, cv_score in zip(search.cv_results_["params"],
                                             search.cv_results_["mean_train_score"],
                                             search.cv_results_["mean_test_score"]):
        i = tree_depths.index(params["max_depth"])
        j = num_trees.index(params["n_estimators"])
        train_scores[i, j] = train_score
        cv_scores[i, j] = cv_score

    # select the shallowest forest within one standard error of the best one
    simple_idx, threshold, best_std_error = select_one_standard_error(search.cv_results_, cv_folds, min_trees)
    simple_params = search.cv_results_["params"][simple_idx]

    # GridSearchCV only refits the best forest, so refit the simple one on the full training data
    simple_forest = sk_ensemble.RandomForestClassifier(
        max_features=feat_train.shape[1], random_state=seed, **simple_params
    ).fit(feat_train, labels_train)

    # evaluate both forests on the test data
    models = [
        ("Best forest", search.best_estimator_, search.best_params_, search.best_score_),
        ("Shallowest forest within 1 SE", simple_forest, simple_params, search.cv_results_["mean_test_score"][simple_idx]),
    ]
    results = []
    for name, forest_model, params, cv_acc in models:
        train_acc = accuracy_score(labels_train, forest_model.predict(feat_train))
        test_acc = accuracy_score(labels_test, forest_model.predict(feat_test))
        results.append((train_acc, test_acc))

        print("\n{0}: num_trees={1}, tree_depth={2}".format(name, params["n_estimators"], params["max_depth"]))
        print("  Cross-validation accuracy: {0:.2%}".format(cv_acc))
        print("  Training accuracy: {0:.2%}".format(train_acc))
        print("  Testing accuracy: {0:.2%}".format(test_acc))
        print("  Feature importance: {0}".format(forest_model.feature_importances_))

    print("\nOne-standard-error threshold: {0:.2%} (best {1:.2%} - SE {2:.2%})".format(
        threshold, search.best_score_, best_std_error))

    # plot heatmaps (top) and the decision boundaries of both forests (bottom)
    # both heatmaps share the same color scale to make them comparable
    figure, axes = plt.subplots(2, 2, figsize=(20, 18), layout="constrained")
    vmin = min(train_scores.min(), cv_scores.min())
    plot_heatmap(axes[0, 0], train_scores, num_trees, tree_depths, "Mean training accuracy", vmin)
    im = plot_heatmap(axes[0, 1], cv_scores, num_trees, tree_depths, "Mean cross-validation accuracy", vmin)
    figure.colorbar(im, ax=axes[0, 1], label="Accuracy")

    # mark both selected combinations
    mark_cell(axes[0, 1], num_trees, tree_depths, search.best_params_, "red", "Best")
    mark_cell(axes[0, 1], num_trees, tree_depths, simple_params, "orange", "Shallowest within 1 SE")
    axes[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

    for ax, (name, forest_model, params, _), (train_acc, test_acc) in zip(axes[1], models, results):
        plot_decision_boundary(
            ax, forest_model, features, feat_train, labels_train, feat_test, labels_test,
            "{0}: {1} trees, depth {2}\nTrain {3:.2%} / Test {4:.2%}".format(
                name, params["n_estimators"], params["max_depth"], train_acc, test_acc)
        )
    figure.suptitle("Random Forest Grid Search", fontsize=16)

    # save figure if flag is set
    if save_fig:
        t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_path = os.path.join(result_dir, "rf_grid_search_{}.png".format(t))
        figure.savefig(file_path, bbox_inches="tight")
        print("Plot saved as " + file_path)

    if show_fig:
        plt.show()


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(
        description="Grid search over decision forest hyper-parameters for 2-dimensional point classification"
    )

    parser.add_argument(
        "--save_fig",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save plot to result_dir (use --no-save_fig to disable).",
    )

    parser.add_argument(
        "--show_fig",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show plot window (use --no-show_fig to disable).",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default=os.path.normpath(os.path.join(script_dir, "random_forest_plots")),
        help="Directory for results.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for data generation and the forest.",
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )

    parser.add_argument(
        "--min_trees",
        type=int,
        default=32,
        help="Minimum number of trees of the forest selected by the one-standard-error rule.",
    )

    args = parser.parse_args()
    main(args.save_fig, args.show_fig, args.result_dir, args.seed, args.cv_folds, args.min_trees)
