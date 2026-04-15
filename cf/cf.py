import numpy as np
import pandas as pd

from collections import defaultdict
from multiprocessing import Pool
from numpy import nan

def eval_user(args):
    cf, user, feat, metric, k = args
    return cf.evaluate_user(user, feat, metric, k)

class CollaborativeFiltering:
    """
    Generalized algorithm for running user-user collaborative filtering. Scales item variables to allow different feature value ranges. 
    Includes functionality for validation and evaluation as well as running predictions on a singular target user. 
    """

    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.X = data.to_numpy()
        self.X_scaled = None
        self.X_centered = None
        self.item_means = None
        self.item_stds = None
        self.user_means = None

        np.random.seed(42)

        self.preprocess()

    def preprocess(self) -> None:
        # scale by item standard deviation first because of variance in value ranges
        self.item_means = np.nanmean(self.X, axis=0)
        self.item_stds = np.nanstd(self.X, axis=0) + 1e-8
        self.X_scaled = (self.X - self.item_means) / self.item_stds

        # center and scale by user means
        self.user_means = np.nanmean(self.X_scaled, axis=1, keepdims=True)
        X_centered = self.X_scaled - self.user_means
        X_centered = np.where(np.isnan(X_centered), 0, X_centered)
        self.X_centered = X_centered

    def run(self, target: str, metric: str, k: int, mask_idx: int = None) -> dict[str, float]:

        # qc checks
        if target not in self.df.index:
            print(f"Error: {target} not found in the dataset.")
            return None

        if sum(self.df.loc[target].isna()) == 0:
            print(f"Error: {target} has no missing scores")
            return None
        
        # get index of target user
        target_idx = self.df.index.get_loc(target)

        target_row_scaled = self.X_scaled[target_idx].copy()
        X_centered = np.delete(self.X_centered, target_idx, axis=0)

        user_mean = self.user_means[target_idx]
        target_preds = np.where(np.isnan(self.X[target_idx]))[0]

        # optionally mask an index (used for loo validation)
        if mask_idx is not None:
            target_row_scaled[mask_idx] = nan
            target_preds = np.array([mask_idx])

        target_row = target_row_scaled - user_mean
        target_row = np.where(np.isnan(target_row), 0, target_row)

        # calculate similarity scores
        eps = 1e-10
        if metric == "l2":
            sim = -np.linalg.norm(X_centered - target_row, ord=2, axis=1)
        elif metric == "cosine":
            sim = X_centered.dot(target_row) / (
                (np.linalg.norm(target_row, ord=2) + eps) * (np.linalg.norm(X_centered, ord=2, axis=1) + eps)
            )
        else:
            print(f"Error: invalid metric, please choose one of (l2, cosine)")

        # get most similar users
        sim_min_max = (sim - np.nanmin(sim)) / (np.nanmax(sim) - np.nanmin(sim) + eps)
        sim_users = np.argpartition(sim_min_max, -k)[-k:]

        # collect predictions
        preds = {}
        for target_pred in target_preds:
            pred_item = self.df.columns[target_pred]
            pred_scaled = np.sum(sim_min_max[sim_users] * X_centered[sim_users, target_pred]) / \
                np.sum(sim_min_max[sim_users])
            
            # rescale predictions
            pred_scaled += user_mean
            pred_rating = pred_scaled * self.item_stds[target_pred] + self.item_means[target_pred]
            preds[pred_item] = pred_rating.item()

        return preds

    def evaluate_user(self, target: str, feat: str, metric: str, k: int):
        # run prediction on a particular user and feature
        actual = self.df.loc[target, feat]
        preds = self.run(target=target, metric=metric, k=k, 
                         mask_idx=self.df.columns.get_loc(feat))
        pred = preds[feat]

        return target, feat, actual, pred, metric, k
        

    def validate(self, sample_size=500) -> pd.DataFrame:
        validation_users = np.random.choice(self.df.index, size=sample_size, replace=False)

        # test different values for similarity metrics and k
        grid = [
            (self, user, feat, sim, k)
            for user in validation_users
            for feat in self.df.columns
            for sim in ["l2", "cosine"]
            for k in [5, 7, 9, 11]
            if not pd.isna(self.df.loc[user, feat])
        ]

        with Pool() as pool:
            results = pool.map(eval_user, grid)

        # collect results
        records = []
        for result in results:
            if result is None:
                continue
            user, feat, true, pred, metric, k = result
            records.append({
                "user": user,
                "feature": feat,
                "actual": true,
                "pred": pred,
                "metric": metric,
                "k": k
            })

        return pd.DataFrame(records)
    
    def evaluate(self, metric: str, k: int, sample_size=500) -> pd.DataFrame:
        #evaluation_users = np.random.choice(self.df.index, size=sample_size, replace=False)

        # perform evaluation on sampled users
        tasks = [
            (self, user, feat, metric, k)
            for user in self.df.index
            for feat in self.df.columns
            if not pd.isna(self.df.loc[user, feat])
        ]

        with Pool() as pool:
            results = pool.map(eval_user, tasks)

        # collect results
        records = []
        for result in results:
            if result is None:
                continue
            user, feat, true, pred, metric, k = result
            records.append({
                "user": user,
                "feature": feat,
                "actual": true,
                "pred": pred,
                "metric": metric,
                "k": k
            })

        return pd.DataFrame(records)
