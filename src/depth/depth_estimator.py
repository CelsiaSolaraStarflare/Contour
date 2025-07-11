"""
Depth Estimation Module
Uses Depth Anything V2 for high-accuracy depth estimation
Target: 0.08m accuracy on NVIDIA Jetson Nano
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Depth estimation using Depth Anything V2"""
    
    def __init__(self, config: Dict):
        """Initialize depth estimator"""
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        self.input_size = tuple(config['model']['input_size'])
        self.output_size = tuple(config['model']['output_size'])
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Move to device
        self.model.to(self.device)
        
        # TensorRT optimization if enabled
        if config['optimization']['tensorrt']:
            self.model = self._optimize_tensorrt()
        
        logger.info(f"Depth estimator initialized on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load Depth Anything V2 model"""
        try:
            # Import Depth Anything V2
            from depth_anything import DepthAnything
            
            model = DepthAnything.from_pretrained(
                'LiheYoung/depth_anything_vitl14',
                trust_remote_code=True
            )
            
            logger.info("Depth Anything V2 model loaded successfully")
            return model
            
        except ImportError:
            logger.error("Depth Anything not found. Please install: pip install depth-anything")
            raise
        except Exception as e:
            logger.error(f"Error loading depth model: {e}")
            raise
    
    def _optimize_tensorrt(self) -> nn.Module:
        """Optimize model with TensorRT"""
        try:
            import torch_tensorrt
            
            # Create TensorRT model
            trt_model = torch_tensorrt.compile(
                self.model,
                inputs=[torch_tensorrt.Input(
                    min_shape=[1, 3, *self.input_size],
                    opt_shape=[1, 3, *self.input_size],
                    max_shape=[1, 3, *self.input_size]
                )],
                enabled_precisions=[torch.float16] if self.config['optimization']['fp16'] else [torch.float32]
            )
            
            logger.info("TensorRT optimization completed")
            return trt_model
            
        except ImportError:
            logger.warning("TensorRT not available, using original model")
            return self.model
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
            return self.model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for depth estimation"""
        # Resize image
        image_resized = cv2.resize(image, self.input_size)
        
        # Convert to RGB if needed
        if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_resized
        
        # Normalize to [0, 1]
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess(self, depth_tensor: torch.Tensor) -> np.ndarray:
        """Postprocess depth tensor to numpy array"""
        # Remove batch dimension and convert to numpy
        depth_np = depth_tensor.squeeze().cpu().numpy()
        
        # Resize to output size
        depth_resized = cv2.resize(depth_np, self.output_size)
        
        # Normalize depth values (assuming model outputs normalized depth)
        depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min())
        
        return depth_normalized
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from RGB image"""
        try:
            with torch.no_grad():
                # Preprocess
                input_tensor = self.preprocess(image)
                
                # Inference
                depth_tensor = self.model(input_tensor)
                
                # Postprocess
                depth_map = self.postprocess(depth_tensor)
                
                return depth_map
                
        except Exception as e:
            logger.error(f"Error in depth estimation: {e}")
            # Return zero depth map as fallback
            return np.zeros(self.output_size, dtype=np.float32)
    
    def estimate_batch(self, images: np.ndarray) -> np.ndarray:
        """Estimate depth for a batch of images"""
        batch_size = len(images)
        depth_maps = []
        
        for i in range(batch_size):
            depth_map = self.estimate(images[i])
            depth_maps.append(depth_map)
        
        return np.array(depth_maps)
    
    def evaluate_accuracy(self, pred_depth: np.ndarray, gt_depth: np.ndarray) -> Dict[str, float]:
        """Evaluate depth estimation accuracy"""
        # Calculate metrics
        rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))
        abs_rel = np.mean(np.abs(pred_depth - gt_depth) / gt_depth)
        sq_rel = np.mean(((pred_depth - gt_depth) ** 2) / gt_depth)
        
        # Calculate accuracy metrics
        thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()
        
        return {
            'rmse': rmse,
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'a1': a1,
            'a2': a2,
            'a3': a3
        }
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


class DepthLoss(nn.Module):
    """Custom loss function for depth estimation"""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
        """Compute depth estimation loss"""
        # Depth loss (L1)
        depth_loss = torch.mean(torch.abs(pred_depth - gt_depth))
        
        # Edge-aware loss
        edge_loss = self._edge_loss(pred_depth, gt_depth)
        
        # Smoothness loss
        smoothness_loss = self._smoothness_loss(pred_depth)
        
        # Total loss
        total_loss = (
            self.weights['depth_loss'] * depth_loss +
            self.weights['edge_loss'] * edge_loss +
            self.weights['smoothness_loss'] * smoothness_loss
        )
        
        return total_loss
    
    def _edge_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Edge-aware loss"""
        # Compute gradients
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        gt_grad_x = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        gt_grad_y = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
        
        # Edge loss
        edge_loss = torch.mean(torch.abs(pred_grad_x - gt_grad_x)) + \
                   torch.mean(torch.abs(pred_grad_y - gt_grad_y))
        
        return edge_loss
    
    def _smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Smoothness loss to encourage smooth depth maps"""
        grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        smoothness_loss = torch.mean(grad_x) + torch.mean(grad_y)
        
        return smoothness_loss 