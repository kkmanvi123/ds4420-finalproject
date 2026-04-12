import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import build_merged_dataset
from dataset import create_dataloaders
from mlp import MLPClassifier, MLPRegressor
from pca import prepare_pca_features, PCATransformer
from trainer import MLPTrainer

import os
import json
import pickle
import torch


def split_dataframe(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split dataframe into train/val/test before PCA fitting.
    val_size is the fraction of the train+val pool after test split.
    """
    # Drop any cols where the target_col has no data
    df = df.dropna(subset=[target_col]).copy()

    stratify_labels = None
    if task == "classification":
        stratify_labels = df[target_col]

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )

    stratify_labels_train = None
    if task == "classification":
        stratify_labels_train = train_val_df[target_col]

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_labels_train
    )

    return train_df, val_df, test_df


def prepare_split_dataframes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    n_components: float | int,
):
    """
    Build PCA features from split dataframes with train-only fitting.
    """
    # Target arrays
    y_train = train_df[target_col].copy()
    y_val = val_df[target_col].copy()
    y_test = test_df[target_col].copy()

    # Clean numeric feature frames
    X_train_df = prepare_pca_features(train_df, target_col=target_col)
    X_val_df = prepare_pca_features(val_df, target_col=target_col)
    X_test_df = prepare_pca_features(test_df, target_col=target_col)

    # Align columns across splits to exactly the train feature columns
    train_cols = X_train_df.columns.tolist()
    X_val_df = X_val_df.reindex(columns=train_cols)
    X_test_df = X_test_df.reindex(columns=train_cols)

    # Fit PCA on train only
    pca_transformer = PCATransformer(n_components=n_components)
    X_train = pca_transformer.fit_transform(X_train_df)
    X_val = pca_transformer.transform(X_val_df)
    X_test = pca_transformer.transform(X_test_df)

    return X_train, X_val, X_test, y_train, y_val, y_test, pca_transformer

def convert_to_json(obj):
    """
    Convert numpy/pandas values into JSON Python types.
    """
    if isinstance(obj, dict):
        return {str(k): convert_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_to_json(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)

def main():
    parser = argparse.ArgumentParser(
        description="Run merged imaging + visitation data through PCA + MLP."
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Folder containing the zip files.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs",
        help="Directory to save checkpoints, metrics, and artifacts."
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save trained model checkpoint and metadata."
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="mri",
        choices=["mri", "fdg_pet", "amyloid_pet", "tau_pet"],
        help="Which imaging table to use."
    )
    parser.add_argument(
        "--visit_zip_name",
        type=str,
        default="NACC_visitation_data.zip",
        help="Zip file containing visitation data."
    )
    parser.add_argument(
        "--visit_csv_name",
        type=str,
        default="investigator_ftldlbd_nacc72.csv",
        help="Exact CSV name inside visitation zip."
    )
    parser.add_argument("--target_col", type=str, required=True, help="Target column name.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "regression"],
        help="Task type."
    )
    parser.add_argument(
        "--n_components",
        type=str,
        default="0.95",
        help="PCA components count (e.g. 10) or explained variance ratio (e.g. 0.95)."
    )
    parser.add_argument(
        "--match_tolerance_days",
        type=int,
        default=180,
        help="Maximum allowed imaging-to-visit date gap in days."
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer sizes, e.g. '128,64' or '256,128,64'"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for hidden layers."
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
        help="Activation function to use in the MLP."
    )
    parser.add_argument(
        "--use_batchnorm",
        action="store_true",
        help="Use batch normalization after each hidden linear layer."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for Adam optimizer."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random_state", type=int, default=42)
    
    # Get the args from the terminal input
    args = parser.parse_args()
    
    if "." in args.n_components:
        args.n_components = float(args.n_components)
    else:
        args.n_components = int(args.n_components)
        
    args.hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]

    # 1. Build merged dataset
    df = build_merged_dataset(
        data_dir=args.data_dir,
        modality=args.modality,
        visit_zip_name=args.visit_zip_name,
        visit_csv_name=args.visit_csv_name,
        tolerance_days=args.match_tolerance_days,
    )

    print(f"\nLoaded merged dataframe: {df.shape}")

    # 2. Get target/task
    target_col = args.target_col
    task = args.task

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in merged dataframe.")

    print(f"Using target column: {target_col}")
    print(f"Task: {task}")
    

    # 3. Split first
    print("\n=== Splitting Data ===\n")
    train_df, val_df, test_df = split_dataframe(
        df=df,
        target_col=target_col,
        task=task,
        random_state=args.random_state,
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    
    

    # 4. PCA preprocessing with train-only fit
    print("\n=== Perform PCA ===\n")
    X_train, X_val, X_test, y_train, y_val, y_test, pca_transformer = prepare_split_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=target_col,
        n_components=args.n_components,
    )

    print(f"Final train PCA shape: {X_train.shape}")
    print(f"Final val PCA shape:   {X_val.shape}")
    print(f"Final test PCA shape:  {X_test.shape}")

    # 5. Encode targets + build loaders
    if task == "classification":
        # Get the class names
        class_names = sorted(df[target_col].dropna().unique().tolist())
        category_map = {cat: i for i, cat in enumerate(class_names)}

        y_train_enc = pd.Series(train_df[target_col]).map(category_map).to_numpy()
        y_val_enc = pd.Series(val_df[target_col]).map(category_map).to_numpy()
        y_test_enc = pd.Series(test_df[target_col]).map(category_map).to_numpy()

        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, y_train_enc,
            X_val, y_val_enc,
            X_test, y_test_enc,
            batch_size=args.batch_size,
            task="classification"
        )

        model = MLPClassifier(
            input_dim=X_train.shape[1],
            num_classes=len(class_names),
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            activation=args.activation,
            use_batchnorm=args.use_batchnorm,
        )

    else:
        y_train_num = pd.to_numeric(y_train, errors="coerce").to_numpy()
        y_val_num = pd.to_numeric(y_val, errors="coerce").to_numpy()
        y_test_num = pd.to_numeric(y_test, errors="coerce").to_numpy()

        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, y_train_num,
            X_val, y_val_num,
            X_test, y_test_num,
            batch_size=args.batch_size,
            task="regression"
        )
        
        model = MLPRegressor(
            input_dim=X_train.shape[1],
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            activation=args.activation,
            use_batchnorm=args.use_batchnorm,
        )

    
    # 6. Train
    print("\n=== Running MLP ===\n")
    trainer = MLPTrainer(
        model=model,
        task=task,
        class_names=class_names if task == "classification" else None,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    history = trainer.fit(train_loader, val_loader=val_loader, epochs=args.epochs)

    # 7. Evaluate
    print("\n=== Validation Metrics ===")
    val_metrics = trainer.evaluate(val_loader)
    for k, v in val_metrics.items():
        print(f"{k}: {v}")

    print("\n=== Test Metrics ===")
    test_metrics = trainer.evaluate(test_loader)
    for k, v in test_metrics.items():
        print(f"{k}: {v}")
        
    # 8. Create plots and save to a plots folder
    experiment_name = f"{args.modality}_{args.target_col}"
    trainer.plot_training_history(
        history,
        save_dir="plots",
        experiment_name=experiment_name
    )

    if task == "classification":
        trainer.plot_confusion_matrix(
            val_metrics,
            class_names=class_names,
            save_dir="plots",
            experiment_name=experiment_name,
            split="val"
        )
        
        trainer.plot_confusion_matrix(
            test_metrics,
            class_names=class_names,
            save_dir="plots",
            experiment_name=experiment_name,
            split="test"
        )
        
    print("Plots saved to: plots!")
    
    # 9. Save outputs / checkpoint
    os.makedirs(args.save_dir, exist_ok=True)

    experiment_name = f"{args.modality}_{args.target_col}"
    experiment_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save metrics and history
    results = {
        "experiment_name": experiment_name,
        "task": task,
        "target_col": target_col,
        "modality": args.modality,
        "args": vars(args),
        "history": convert_to_json(history),
        "val_metrics": convert_to_json(val_metrics),
        "test_metrics": convert_to_json(test_metrics),
        "class_names": convert_to_json(class_names) if task == "classification" else None,
    }

    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save PCA transformer
    with open(os.path.join(experiment_dir, "pca_transformer.pkl"), "wb") as f:
        pickle.dump(pca_transformer, f)

    # Save model checkpoint
    if args.save_model:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "task": task,
            "target_col": target_col,
            "modality": args.modality,
            "input_dim": X_train.shape[1],
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "activation": args.activation,
            "use_batchnorm": args.use_batchnorm,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "n_components": args.n_components,
            "class_names": class_names if task == "classification" else None,
            "category_map": category_map if task == "classification" else None,
        }

        if task == "classification":
            checkpoint["num_classes"] = len(class_names)

        torch.save(checkpoint, os.path.join(experiment_dir, "model_checkpoint.pt"))

    print(f"Saved outputs to: {experiment_dir}!")


if __name__ == "__main__":
    main()