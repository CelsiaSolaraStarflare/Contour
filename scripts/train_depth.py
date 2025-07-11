#!/usr/bin/env python3
"""
Depth Estimation Training Script
Trains Depth Anything V2 for high-accuracy depth estimation
Target: 0.08m accuracy on NVIDIA Jetson Nano
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import logging
from pathlib import Path
import time
from tqdm import tqdm
import wandb

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.depth.depth_estimator import DepthEstimator, DepthLoss
from src.utils.jetson_utils import setup_jetson, get_optimal_batch_size
from src.utils.camera import Camera

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepthDataset(torch.utils.data.Dataset):
    """Dataset for depth estimation training"""
    
    def __init__(self, config: dict, split: str = 'train'):
        """Initialize dataset"""
        self.config = config
        self.split = split
        
        # Data paths
        self.image_dir = Path(config['data'][f'{split}_path'])
        self.depth_dir = Path(config['data'][f'{split}_path'].replace('images', 'depth'))
        
        # Get image files
        self.image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        
        # Augmentation settings
        self.augmentation = config['data']['augmentation']
        
        logger.info(f"Loaded {len(self.image_files)} images for {split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load depth map
        depth_path = self.depth_dir / f"{image_path.stem}_depth.npy"
        if depth_path.exists():
            depth_map = np.load(depth_path)
        else:
            # Create dummy depth map if not available
            depth_map = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 5.0
        
        # Apply augmentation
        if self.split == 'train' and self.augmentation:
            image, depth_map = self._apply_augmentation(image, depth_map)
        
        # Preprocess
        image = self._preprocess_image(image)
        depth_map = self._preprocess_depth(depth_map)
        
        return {
            'image': torch.from_numpy(image).float(),
            'depth': torch.from_numpy(depth_map).float(),
            'path': str(image_path)
        }
    
    def _apply_augmentation(self, image: np.ndarray, depth_map: np.ndarray):
        """Apply data augmentation"""
        # Horizontal flip
        if self.augmentation.get('horizontal_flip', False) and np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            depth_map = cv2.flip(depth_map, 1)
        
        # Vertical flip
        if self.augmentation.get('vertical_flip', False) and np.random.random() > 0.5:
            image = cv2.flip(image, 0)
            depth_map = cv2.flip(depth_map, 0)
        
        # Rotation
        if self.augmentation.get('rotation', 0) > 0:
            angle = np.random.uniform(-self.augmentation['rotation'], self.augmentation['rotation'])
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            image = cv2.warpAffine(image, M, (width, height))
            depth_map = cv2.warpAffine(depth_map, M, (width, height))
        
        # Color augmentation
        if self.augmentation.get('brightness', 0) > 0:
            brightness = np.random.uniform(1 - self.augmentation['brightness'], 
                                         1 + self.augmentation['brightness'])
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        if self.augmentation.get('contrast', 0) > 0:
            contrast = np.random.uniform(1 - self.augmentation['contrast'], 
                                       1 + self.augmentation['contrast'])
            image = np.clip(image * contrast, 0, 255).astype(np.uint8)
        
        return image, depth_map
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize
        target_size = self.config['model']['input_size']
        image_resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to CHW format
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        
        return image_chw
    
    def _preprocess_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Preprocess depth map"""
        # Resize
        target_size = self.config['model']['output_size']
        depth_resized = cv2.resize(depth_map, target_size)
        
        # Add channel dimension
        depth_chw = depth_resized[np.newaxis, :, :]
        
        return depth_chw


def train_depth_model(config_path: str):
    """Train depth estimation model"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup Jetson Nano
    if os.name == 'posix' and os.path.exists('/etc/nv_tegra_release'):
        setup_jetson()
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    
    # Initialize wandb
    if config['logging'].get('wandb_project'):
        wandb.init(project=config['logging']['wandb_project'], config=config)
    
    # Create datasets
    train_dataset = DepthDataset(config, 'train')
    val_dataset = DepthDataset(config, 'val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Initialize model
    depth_estimator = DepthEstimator(config)
    model = depth_estimator.model
    
    # Initialize loss function
    loss_weights = config['training']['loss_weights']
    criterion = DepthLoss(loss_weights)
    
    # Initialize optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['training']['learning_rate'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    
    # Initialize scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    elif config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        scheduler = None
    
    # Training loop
    best_val_loss = float('inf')
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting depth estimation training...")
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {}
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
        
        for batch in train_pbar:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_depths = model(images)
            
            # Calculate loss
            loss = criterion(pred_depths, depths)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Calculate additional metrics
            with torch.no_grad():
                metrics = depth_estimator.evaluate_accuracy(
                    pred_depths.cpu().numpy(), depths.cpu().numpy()
                )
                for key, value in metrics.items():
                    train_metrics[key] = train_metrics.get(key, 0) + value
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rmse': f"{metrics['rmse']:.4f}"
            })
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
            
            for batch in val_pbar:
                images = batch['image'].to(device)
                depths = batch['depth'].to(device)
                
                # Forward pass
                pred_depths = model(images)
                
                # Calculate loss
                loss = criterion(pred_depths, depths)
                val_loss += loss.item()
                
                # Calculate metrics
                metrics = depth_estimator.evaluate_accuracy(
                    pred_depths.cpu().numpy(), depths.cpu().numpy()
                )
                for key, value in metrics.items():
                    val_metrics[key] = val_metrics.get(key, 0) + value
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'rmse': f"{metrics['rmse']:.4f}"
                })
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Train RMSE: {train_metrics['rmse']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
        
        # Log to wandb
        if wandb.run:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_rmse': train_metrics['rmse'],
                'val_rmse': val_metrics['rmse'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = log_dir / 'best_depth_model.pth'
            depth_estimator.save_model(str(model_path))
            logger.info(f"New best model saved: {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_freq'] == 0:
            checkpoint_path = log_dir / f'depth_model_epoch_{epoch+1}.pth'
            depth_estimator.save_model(str(checkpoint_path))
        
        # Early stopping check
        if val_metrics['rmse'] <= config['evaluation']['target_rmse']:
            logger.info(f"Target RMSE achieved: {val_metrics['rmse']:.4f}")
            break
    
    # Final model optimization
    if config['optimization']['tensorrt']:
        logger.info("Optimizing model with TensorRT...")
        optimized_model = depth_estimator._optimize_tensorrt()
        
        # Save optimized model
        optimized_path = log_dir / 'depth_model_optimized.pth'
        torch.save({
            'model_state_dict': optimized_model.state_dict(),
            'config': config
        }, optimized_path)
        logger.info(f"Optimized model saved: {optimized_path}")
    
    logger.info("Depth estimation training completed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train Depth Anything V2 model")
    parser.add_argument('--config', type=str, default='configs/depth_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        train_depth_model(args.config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 