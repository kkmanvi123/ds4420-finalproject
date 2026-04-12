import torch
import numpy as np
from metrics import classification_metrics, regression_metrics


class MLPTrainer:
    def __init__(self, model, task, lr=1e-3, weight_decay=0.0, device=None):
        """
        model: the MLP model (classifier or regressor)
        task: 'classification' or 'regression'
        """

        self.task = task

        # Use GPU if available
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Loss function comes from model (MSE or CrossEntropy)
        self.loss_fn = model.get_loss_fn()

        # Use Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train_one_epoch(self, dataloader):
        """
        Runs one full pass over the training dataset.
        """
        self.model.train()  # enable dropout, batchnorm updates
        total_loss = 0.0

        for X, y in dataloader:
            # Move batch to device
            X = X.to(self.device)
            y = self.model.prepare_targets(y.to(self.device))

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X)

            # Compute loss
            loss = self.loss_fn(outputs, y)

            # Backpropagation
            loss.backward()

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()

        # Return average loss across batches
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """
        Evaluate model on validation/test set.
        Returns loss + metrics depending on task.
        """
        self.model.eval()  # disable dropout, freeze batchnorm
        total_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():  # no gradients during eval
            for X, y in dataloader:
                X = X.to(self.device)
                y_device = y.to(self.device)
                y_prepped = self.model.prepare_targets(y_device)

                # Forward pass
                outputs = self.model(X)

                # Compute loss
                loss = self.loss_fn(outputs, y_prepped)
                total_loss += loss.item()

                if self.task == "classification":
                    # Convert logits to predicted class
                    preds = torch.argmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_device.cpu().numpy())

                else:
                    # Flatten outputs for regression metrics
                    preds = outputs.view(-1)
                    targets = y_device.view(-1)

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

        # Base loss
        results = {"loss": total_loss / len(dataloader)}

        # Add task-specific metrics
        if self.task == "classification":
            results.update(
                classification_metrics(
                    np.array(all_targets),
                    np.array(all_preds)
                )
            )
        else:
            results.update(
                regression_metrics(
                    np.array(all_targets),
                    np.array(all_preds)
                )
            )

        return results

    def fit(self, train_loader, val_loader=None, epochs=20):
        """
        Full training loop across epochs.
        """
        # Track the metrics
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

        for epoch in range(epochs):
            # Train step
            train_loss = self.train_one_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Evaluate on train
            train_metrics = self.evaluate(train_loader)
            history["train_metrics"].append(train_metrics)

            # Print training metrics
            if self.task == "classification":
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Train F1: {train_metrics['f1']:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train MAE: {train_metrics['mae']:.4f} | "
                    f"Train RMSE: {train_metrics['rmse']:.4f} | "
                    f"Train R2: {train_metrics['r2']:.4f}"
                )

            # Validation step
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_metrics"].append(val_metrics)

                # Print validation metrics
                if self.task == "classification":
                    print(
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val Acc: {val_metrics['accuracy']:.4f} | "
                        f"Val F1: {val_metrics['f1']:.4f}"
                    )
                else:
                    print(
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val MAE: {val_metrics['mae']:.4f} | "
                        f"Val RMSE: {val_metrics['rmse']:.4f} | "
                        f"Val R2: {val_metrics['r2']:.4f}"
                    )

        return history