import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import logging


class TFTTrainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logging.getLogger('tft_trainer')

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")

        # Loss function (quantile loss)
        self.quantiles = config['model']['quantiles']

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['model']['learning_rate'],
            weight_decay=config['model'].get('weight_decay', 0.01)
        )

        # Learning rate scheduler (fixed the verbose parameter)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['model'].get('scheduler_patience', 5)
        )

        # Early stopping
        self.early_stopping_patience = config['model'].get('early_stopping_patience', 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Checkpoint dir
        self.checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'models/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger.info("TFT Trainer initialized")

    def quantile_loss(self, y_pred, y_true):
        """
        Calculate quantile loss
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            if i < y_pred.shape[-1]:  # Make sure we have this quantile
                errors = y_true - y_pred[:, :, i]
                losses.append(torch.max((q - 1) * errors, q * errors).mean())

        if losses:
            return torch.mean(torch.stack(losses))
        else:
            # Fallback to MSE if quantile dimensions don't match
            return nn.MSELoss()(y_pred.mean(dim=-1), y_true)

    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        try:
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
                # Move data to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                try:
                    outputs = self.model(batch)
                    target = batch['target'].to(self.device)

                    # Handle output dimensions
                    if len(outputs.shape) == 2:
                        outputs = outputs.unsqueeze(-1)  # Add quantile dimension

                    # Make sure target has the right shape
                    if len(target.shape) == 2:
                        # target is [batch, sequence], outputs is [batch, sequence, quantiles]
                        loss = self.quantile_loss(outputs, target)
                    else:
                        loss = nn.MSELoss()(outputs.mean(dim=-1), target)

                    # Add L1 regularization if configured
                    l1_reg = self.config['model'].get('l1_regularization', 0)
                    if l1_reg > 0:
                        l1_penalty = 0
                        for param in self.model.parameters():
                            l1_penalty += torch.norm(param, 1)
                        loss += l1_reg * l1_penalty

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    grad_clip = self.config['model'].get('gradient_clip_val', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                    self.optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in training epoch: {e}")

        return epoch_loss / max(num_batches, 1)

    def validate(self):
        """
        Validate the model
        """
        self.model.eval()
        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            try:
                for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                    # Move data to device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)

                    try:
                        # Forward pass
                        outputs = self.model(batch)
                        target = batch['target'].to(self.device)

                        # Handle output dimensions
                        if len(outputs.shape) == 2:
                            outputs = outputs.unsqueeze(-1)

                        # Calculate loss
                        if len(target.shape) == 2:
                            loss = self.quantile_loss(outputs, target)
                        else:
                            loss = nn.MSELoss()(outputs.mean(dim=-1), target)

                        val_loss += loss.item()
                        num_batches += 1

                    except Exception as e:
                        self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                        continue

            except Exception as e:
                self.logger.error(f"Error in validation: {e}")

        return val_loss / max(num_batches, 1)

    def train(self):
        """
        Main training loop
        """
        self.logger.info("Starting training...")

        num_epochs = self.config['model'].get('num_epochs', 100)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={current_lr:.8f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                self._save_checkpoint(epoch, val_loss, is_best=True)
                self.logger.info(f"New best model saved with validation loss: {val_loss:.6f}")

            else:
                patience_counter += 1
                self.logger.info(f"No improvement. Patience: {patience_counter}/{self.early_stopping_patience}")

                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)

        # Test evaluation if test loader provided
        if self.test_loader:
            self.logger.info("Evaluating on test set...")
            test_loss = self._evaluate_test()
            self.logger.info(f"Test Loss: {test_loss:.6f}")
            return {
                'train_loss': train_loss,
                'val_loss': best_val_loss,
                'test_loss': test_loss,
                'epochs_trained': epoch + 1
            }
        else:
            return {
                'train_loss': train_loss,
                'val_loss': best_val_loss,
                'epochs_trained': epoch + 1
            }

    def _evaluate_test(self):
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.test_loader:
                # Move data to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                try:
                    # Forward pass
                    outputs = self.model(batch)
                    target = batch['target'].to(self.device)

                    # Handle output dimensions
                    if len(outputs.shape) == 2:
                        outputs = outputs.unsqueeze(-1)

                    # Calculate loss
                    if len(target.shape) == 2:
                        loss = self.quantile_loss(outputs, target)
                    else:
                        loss = nn.MSELoss()(outputs.mean(dim=-1), target)

                    test_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    self.logger.error(f"Error in test evaluation: {e}")
                    continue

        return test_loss / max(num_batches, 1)

    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'config': self.config
            }

            if is_best:
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                self.logger.info(f"Best model saved to {best_path}")
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f"Checkpoint saved to {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def train_with_cross_validation(self, n_splits=5):
        """
        Train using cross-validation (simplified version)
        """
        self.logger.info(f"Cross-validation training not fully implemented. Running single training instead.")
        return self.train()