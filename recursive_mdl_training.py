"""
Recursive MDL Training Example

This program demonstrates using MDLLoss recursively to train both:
1. The main autoencoder with MDL loss
2. The NormalizerModel itself with MDL loss to prevent overfitting

The key insight is that the normalizer is also a model that can overfit,
so we apply the same MDL principle to its training.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple, Optional
import numpy as np

# Import our separated MDL components
from mdl_loss_separated import (
    MDLLoss, NormalizerTrainer, NormalizerModel,
    calculate_statistical_moments, moment_distance_loss
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RecursiveMDLTrainer:
    """
    Trainer that applies MDL principles recursively to both main model and normalizer.
    """
    
    def __init__(self, 
                 main_model: nn.Module,
                 num_features: int,
                 main_lr: float = 1e-3,
                 normalizer_lr: float = 1e-4,
                 validation_split: float = 0.2):
        """
        Initialize recursive MDL trainer.
        
        Args:
            main_model: The autoencoder or main model to train
            num_features: Number of features for normalization
            main_lr: Learning rate for main model
            normalizer_lr: Learning rate for normalizer
            validation_split: Fraction of data to hold out for validation
        """
        self.main_model = main_model
        self.num_features = num_features
        self.validation_split = validation_split
        
        # Primary components
        self.main_mdl_loss = MDLLoss()
        self.normalizer_trainer = NormalizerTrainer(num_features, lr=normalizer_lr)
        
        # Meta-MDL for normalizer training
        self.normalizer_mdl_loss = MDLLoss()
        
        # Optimizers
        self.main_optimizer = torch.optim.Adam(main_model.parameters(), lr=main_lr)
        
        # Validation tracking
        self.best_main_val_mdl = float('inf')
        self.best_normalizer_val_mdl = float('inf')
        self.patience_main = 0
        self.patience_normalizer = 0
        self.max_patience = 5
        
        # Metrics tracking
        self.training_history = {
            'main_mdl': [], 'main_val_mdl': [],
            'normalizer_moment': [], 'normalizer_val_mdl': [],
            'normalizer_updates_per_batch': []
        }
    
    def split_validation_data(self, dataset):
        """Split dataset into training and validation sets."""
        dataset_size = len(dataset.dataset)
        val_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset.dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=dataset.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=dataset.batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_normalizer_with_mdl(self, residuals: torch.Tensor, 
                                  max_updates: int = 10) -> Tuple[float, int]:
        """
        Train normalizer using MDL principle to prevent overfitting.
        
        Returns:
            (final_moment_loss, num_updates_performed)
        """
        best_normalizer_mdl = float('inf')
        no_improvement_count = 0
        
        for update_idx in range(max_updates):
            # Standard moment-based update
            moment_loss = self.normalizer_trainer.update(residuals)
            
            # Evaluate normalizer's own MDL cost
            normalizer_mdl = self.evaluate_normalizer_mdl(residuals)
            
            # Early stopping based on MDL
            if normalizer_mdl < best_normalizer_mdl:
                best_normalizer_mdl = normalizer_mdl
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
                # Stop if normalizer MDL isn't improving (overfitting protection)
                if no_improvement_count >= 3:
                    logger.debug(f"Stopping normalizer updates at {update_idx + 1} "
                               f"due to MDL plateau")
                    break
        
        return moment_loss, update_idx + 1
    
    def validate_models(self, val_loader) -> Tuple[float, float]:
        """Validate both main model and normalizer on held-out data."""
        self.main_model.eval()
        
        total_main_mdl = 0.0
        total_normalizer_mdl = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, in val_loader:
                if batch_data.dim() > 2:
                    batch_data = batch_data.view(batch_data.shape[0], -1)
                
                # Main model validation
                yhat = self.main_model(batch_data)
                normalize_fn = self.normalizer_trainer.get_normalize_function()
                main_mdl = self.main_mdl_loss(batch_data, yhat, self.main_model, normalize_fn)
                
                # Normalizer validation
                residuals = batch_data - yhat
                normalizer_mdl = self.evaluate_normalizer_mdl(residuals)
                
                total_main_mdl += main_mdl.item()
                total_normalizer_mdl += normalizer_mdl
                num_batches += 1
        
        self.main_model.train()
        return total_main_mdl / num_batches, total_normalizer_mdl / num_batches
    
    def train_epoch(self, train_loader, val_loader, epoch: int) -> Dict[str, float]:
        """Train one epoch with recursive MDL application."""
        self.main_model.train()
        
        epoch_main_mdl = 0.0
        epoch_moment_loss = 0.0
        epoch_normalizer_updates = 0.0
        num_batches = 0
        
        for batch_idx, (batch_data,) in enumerate(train_loader):
            if batch_data.dim() > 2:
                batch_data = batch_data.view(batch_data.shape[0], -1)
            
            # Forward pass
            yhat = self.main_model(batch_data)
            residuals = batch_data - yhat
            
            # Get current normalization function
            normalize_fn = self.normalizer_trainer.get_normalize_function()
            
            # Calculate main model MDL loss
            main_mdl_loss = self.main_mdl_loss(batch_data, yhat, self.main_model, normalize_fn)
            
            # Update main model
            self.main_optimizer.zero_grad()
            main_mdl_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), max_norm=1.0)
            self.main_optimizer.step()
            
            # Train normalizer with MDL-based early stopping
            moment_loss, num_updates = self.train_normalizer_with_mdl(
                residuals.detach(), max_updates=10
            )
            
            # Accumulate metrics
            epoch_main_mdl += main_mdl_loss.item()
            epoch_moment_loss += moment_loss
            epoch_normalizer_updates += num_updates
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Main MDL = {main_mdl_loss.item():.0f}, "
                          f"Moment = {moment_loss:.3f}, "
                          f"Normalizer Updates = {num_updates}")
        
        # Validation
        val_main_mdl, val_normalizer_mdl = self.validate_models(val_loader)
        
        # Early stopping logic
        if val_main_mdl < self.best_main_val_mdl:
            self.best_main_val_mdl = val_main_mdl
            self.patience_main = 0
        else:
            self.patience_main += 1
        
        if val_normalizer_mdl < self.best_normalizer_val_mdl:
            self.best_normalizer_val_mdl = val_normalizer_mdl
            self.patience_normalizer = 0
        else:
            self.patience_normalizer += 1
        
        # Store metrics
        metrics = {
            'train_main_mdl': epoch_main_mdl / num_batches,
            'val_main_mdl': val_main_mdl,
            'train_moment_loss': epoch_moment_loss / num_batches,
            'val_normalizer_mdl': val_normalizer_mdl,
            'avg_normalizer_updates': epoch_normalizer_updates / num_batches
        }
        
        # Update history
        self.training_history['main_mdl'].append(metrics['train_main_mdl'])
        self.training_history['main_val_mdl'].append(metrics['val_main_mdl'])
        self.training_history['normalizer_moment'].append(metrics['train_moment_loss'])
        self.training_history['normalizer_val_mdl'].append(metrics['val_normalizer_mdl'])
        self.training_history['normalizer_updates_per_batch'].append(metrics['avg_normalizer_updates'])
        
        return metrics
    
    def should_stop_training(self) -> bool:
        """Check if training should stop based on validation performance."""
        return (self.patience_main >= self.max_patience or 
                self.patience_normalizer >= self.max_patience)
    
    def train(self, dataset, max_epochs: int = 20) -> Dict:
        """Main training loop with recursive MDL."""
        logger.info("Starting Recursive MDL Training")
        logger.info("=" * 50)
        
        # Split data
        train_loader, val_loader = self.split_validation_data(dataset)
        logger.info(f"Training batches: {len(train_loader)}, "
                   f"Validation batches: {len(val_loader)}")
        
        for epoch in range(max_epochs):
            metrics = self.train_epoch(train_loader, val_loader, epoch)
            
            logger.info(f"Epoch {epoch}: "
                       f"Train MDL = {metrics['train_main_mdl']:.0f}, "
                       f"Val MDL = {metrics['val_main_mdl']:.0f}, "
                       f"Moment = {metrics['train_moment_loss']:.3f}, "
                       f"Norm Updates = {metrics['avg_normalizer_updates']:.1f}")
            
            if self.should_stop_training():
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        return self.training_history
    def evaluate_normalizer_mdl(self, residuals: torch.Tensor) -> float:
        """
   Calculate MDL cost of lambda parameters directly.
   Now we have exactly num_features parameters instead of a neural network.
   """
        # Simple parameter encoding for lambda values
        num_params = self.normalizer_trainer.lambdas.numel()  # Should be num_features
        
        # Calculate parameter bits assuming reasonable precision
        param_variance = self.normalizer_trainer.lambdas.var(unbiased=False).clamp_min(1e-12)
        
        # Differential entropy + quantization for lambda parameters
        diff_bits = num_params * (0.5 * math.log2(2 * math.pi * math.e) + 
                                 0.5 * torch.log2(param_variance))
        quant_bits = num_params * math.log2(1.0 / 1e-6)  # Parameter resolution
        
        total_normalizer_mdl = diff_bits + quant_bits
        
        return total_normalizer_mdl.item()
    def _estimate_optimal_lambdas(self, residuals: torch.Tensor) -> torch.Tensor:
        """
   This method is no longer needed since we optimize lambdas directly.
   Return current lambda values as placeholder.
   """
        return self.normalizer_trainer.lambdas.detach()


def create_autoencoder(input_dim: int) -> nn.Module:
    """Create a simple autoencoder architecture."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        # Decoder
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, input_dim)
    )


def create_diverse_dataset(num_samples: int = 2000, num_features: int = 784) -> torch.utils.data.DataLoader:
    """Create a dataset with diverse statistical distributions."""
    torch.manual_seed(42)
    
    # Create data with various distributions to make normalization meaningful
    data_chunks = [
        torch.randn(num_samples, num_features // 4) * 0.5,  # Normal
        torch.distributions.Exponential(2.0).sample((num_samples, num_features // 4)),  # Skewed
        torch.rand(num_samples, num_features // 4) * 10,  # Uniform
        torch.randn(num_samples, num_features // 4).abs() * 2  # Half-normal
    ]
    
    dummy_data = torch.cat(data_chunks, dim=1)
    
    # Add some correlation structure
    noise = torch.randn(num_samples, num_features) * 0.1
    dummy_data = dummy_data + noise
    
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_data),
        batch_size=64,
        shuffle=True
    )


def main():
    """Main execution with recursive MDL training."""
    
    # Configuration
    num_features = 784
    
    # Create components
    autoencoder = create_autoencoder(num_features)
    dataset = create_diverse_dataset(num_samples=2000, num_features=num_features)
    
    logger.info(f"Dataset: {len(dataset.dataset)} samples, {num_features} features")
    logger.info(f"Model parameters: {sum(p.numel() for p in autoencoder.parameters())}")
    
    # Initialize recursive trainer
    trainer = RecursiveMDLTrainer(
        main_model=autoencoder,
        num_features=num_features,
        main_lr=1e-3,
        normalizer_lr=1e-4,
        validation_split=0.2
    )
    
    # Train with recursive MDL
    training_history = trainer.train(dataset, max_epochs=15)
    
    # Final analysis
    logger.info("\n" + "=" * 50)
    logger.info("FINAL RECURSIVE MDL ANALYSIS")
    logger.info("=" * 50)
    
    # Test final performance
    test_batch = next(iter(dataset))[0]
    if test_batch.dim() > 2:
        test_batch = test_batch.view(test_batch.shape[0], -1)
    
    with torch.no_grad():
        yhat_final = trainer.main_model(test_batch)
        residuals = test_batch - yhat_final
        
        normalize_fn = trainer.normalizer_trainer.get_normalize_function()
        final_main_mdl = trainer.main_mdl_loss(test_batch, yhat_final, trainer.main_model, normalize_fn)
        final_normalizer_mdl = trainer.evaluate_normalizer_mdl(residuals)
        
        stats = trainer.normalizer_trainer.get_stats(residuals)
        
        logger.info(f"Final Main Model MDL: {final_main_mdl.item():.0f} bits")
        logger.info(f"Final Normalizer MDL: {final_normalizer_mdl:.0f} bits")
        logger.info(f"Normalization effectiveness:")
        logger.info(f"  Variance: {stats['orig_var_mean']:.3f} → {stats['norm_var_mean']:.3f}")
        logger.info(f"  Mean deviation: {stats['orig_mean_abs']:.3f} → {stats['norm_mean_abs']:.3f}")
        logger.info(f"  Lambda statistics: μ={stats['lambda_mean']:.3f}, σ={stats['lambda_std']:.3f}")
        
        # Training efficiency metrics
        avg_updates = np.mean(training_history['normalizer_updates_per_batch'])
        logger.info(f"Average normalizer updates per batch: {avg_updates:.2f}")
        logger.info(f"Training converged in {len(training_history['main_mdl'])} epochs")
        
        # Overfitting analysis
        final_train_mdl = training_history['main_mdl'][-1]
        final_val_mdl = training_history['main_val_mdl'][-1]
        overfitting_ratio = final_val_mdl / final_train_mdl
        logger.info(f"Overfitting analysis: Val/Train MDL ratio = {overfitting_ratio:.3f}")
        
        if overfitting_ratio < 1.1:
            logger.info("✓ Minimal overfitting detected")
        elif overfitting_ratio < 1.3:
            logger.info("⚠ Moderate overfitting detected")
        else:
            logger.info("✗ Significant overfitting detected")


if __name__ == "__main__":
    main()
