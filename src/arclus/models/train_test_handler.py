import logging
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from arclus.utils import set_random_seed


class TrainTestHandler:
    """
    A training and test handler.
    Handles batch-wise training on a train set, early stopping on a validation set
    and provides function for prediction
    """

    def __init__(
        self,
        base_model: nn.Module,
        optimizer,
        criterion,
        epochs: int = 2000,
        device: Optional[torch.device] = None,
        random_seed: Optional[int] = 0,
        patience: int = 3,
        min_delta: float = 1.0e-05,
    ) -> None:
        """Initialize the train handler."""
        set_random_seed(random_seed)
        # max number of epochs
        self.epochs = epochs

        # Initialize the device
        self.device = device
        # early stopping params
        self.patience = patience
        self.min_delta = min_delta

        self.history = None
        self.reset_history()

        # Move model to device
        self.model = base_model.to(device=self.device)
        self.optimizer = torch.optim.Adam(params=(p for p in base_model.parameters() if p.requires_grad))
        self.loss = criterion

    def reset_history(self):
        self.history = []

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """
        Fit the model batch wise on train set and perform early stopping on validation set.
        :param train_loader: train set
        :param val_loader: validation set
        :return: None
        """
        self.reset_history()
        for epoch in range(self.epochs):
            self.model.train()

            for iteration, (x, y_true) in enumerate(train_loader, 0):
                # torch *accumulates* the gradient; hence we need to zero it before computing new gradients
                self.optimizer.zero_grad()

                # predict probability for each class
                y_pred = self.model(x.to(self.device))
                # get loss
                loss_value = self.loss(y_pred, y_true.to(self.device))

                # compute gradients
                loss_value.backward()

                # update parameters
                self.optimizer.step()

            # Validation phase
            result = self.evaluate(val_loader)
            logging.info(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
            self.history.append(result["val_loss"])

            # Early stopping on validation set
            if len(self.history) > self.patience:
                if result["val_loss"] >= (1 - self.min_delta) * max(self.history[-self.patience:]):
                    logging.info('Early stopping')
                    break

    def predict(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict probability for each label.
        :param x: the data
        :return: tensor containing probability for class 0 and 1 for each entry
        """
        self.model.eval()
        with torch.no_grad():
            y_pred_prob = self.model.forward(x.to(self.device)).cpu()
        return y_pred_prob

    def evaluate(
        self,
        val_loader: DataLoader
    ) -> dict:
        """
        Evaluation on validation set.
        :param val_loader:
        :return: dict: Return loss and accuracy for this epoch
        """
        self.model.eval()
        with torch.no_grad():
            val_loss = correct = count = 0
            for x, y_true in val_loader:
                y_pred = self.model(x.to(self.device))
                y_true = y_true.to(self.device)
                val_loss += (x.shape[0] * self.loss(y_pred, y_true)).item()
                correct += ((y_pred > 0) == (y_true > 0)).sum().item()
                count += y_true.numel()
            val_loss /= count
            acc = correct / count
        return dict(
            val_loss=val_loss,
            val_acc=acc,
        )
