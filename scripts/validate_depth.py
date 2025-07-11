#!/usr/bin/env python3

"""
Standalone Depth Estimation Validation Script
Validates Depth Anything V2 model accuracy
Target: 0.08m accuracy on NVIDIA Jetson Nano
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import wandb
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.depth.depth_estimator import DepthEstimator, DepthLoss
from src.utils.jetson_utils import setup_jetson

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DepthDataset(torch.utils.data.Dataset):
    # (Copy the DepthDataset class from train_depth.py here)
    def __init__(self, config: dict, split: str = 'val'):
        self.config = config
        self.split = split
        
        self.image_dir = Path(config['data'][f'{split}_path'])
        self.depth_dir = Path(config['data'][f'{split}_path'].replace('images', 'depth'))
        
        self.image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        
        self.augmentation = config['data']['augmentation']
        
        logger.info(f"Loaded {len(self.image_files)} images for {split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        import cv2
        import numpy as np
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth_path = self.depth_dir / f"{image_path.stem}_depth.npy"
        if depth_path.exists():
            depth_map = np.load(depth_path)
        else:
            depth_map = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 5.0
        
        if self.split == 'train' and self.augmentation:
            image, depth_map = self._apply_augmentation(image, depth_map)
        
        image = self._preprocess_image(image)
        depth_map = self._preprocess_depth(depth_map)
        
        return {
            'image': torch.from_numpy(image).float(),
            'depth': torch.from_numpy(depth_map).float(),
            'path': str(image_path)
        }
    
    def _apply_augmentation(self, image: np.ndarray, depth_map: np.ndarray):
        import cv2
        import numpy as np
        if self.augmentation.get('horizontal_flip', False) and np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            depth_map = cv2.flip(depth_map, 1)
        
        if self.augmentation.get('vertical_flip', False) and np.random.random() > 0.5:
            image = cv2.flip(image, 0)
            depth_map = cv2.flip(depth_map, 0)
        
        if self.augmentation.get('rotation', 0) > 0:
            angle = np.random.uniform(-self.augmentation['rotation'], self.augmentation['rotation'])
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (width, height))
            depth_map = cv2.warpAffine(depth_map, M, (width, height))
        
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
        import cv2
        import numpy as np
        target_size = self.config['model']['input_size']
        image_resized = cv2.resize(image, target_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        return image_chw
    
    def _preprocess_depth(self, depth_map: np.ndarray) -> np.ndarray:
        import cv2
        import numpy as np
        target_size = self.config['model']['output_size']
        depth_resized = cv2.resize(depth_map, target_size)
        depth_chw = depth_resized[np.newaxis, :, :]
        return depth_chw

def validate_depth_model(config_path: str, model_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['hardware']['device'])
    
    if config['logging'].get('wandb_project'):
        wandb.init(project=config['logging']['wandb_project'] + '_validate', config=config)
    
    val_dataset = DepthDataset(config, 'val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    depth_estimator = DepthEstimator(config)
    depth_estimator.load_model(model_path)
    model = depth_estimator.model
    model.to(device)
    model.eval()
    
    loss_weights = config['training']['loss_weights']
    criterion = DepthLoss(loss_weights)
    
    val_loss = 0.0
    val_metrics = {}
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validation")
        for batch in val_pbar:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            
            pred_depths = model(images)
            
            loss = criterion(pred_depths, depths)
            val_loss += loss.item()
            
            metrics = depth_estimator.evaluate_accuracy(
                pred_depths.cpu().numpy(), depths.cpu().numpy()
            )
            for key, value in metrics.items():
                val_metrics[key] = val_metrics.get(key, 0) + value
            
            val_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rmse': f"{metrics['rmse']:.4f}"
            })
    
    val_loss /= len(val_loader)
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)
    
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")
    logger.info(f"Other metrics: {val_metrics}")
    
    if wandb.run:
        wandb.log({
            'val_loss': val_loss,
            'val_rmse': val_metrics['rmse'],
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })
    
    return val_metrics

def main():
    parser = argparse.ArgumentParser(description="Validate Depth Anything V2 model")
    parser.add_argument('--config', type=str, default='configs/depth_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file')
    
    args = parser.parse_args()
    
    try:
        validate_depth_model(args.config, args.model)
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    main() 