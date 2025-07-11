"""
Object Detection Module
Uses SSD with MobileNetV3-Small for traffic signs and lane markings detection
Target: 20 FPS on NVIDIA Jetson Nano
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Object detection using SSD with MobileNetV3-Small"""
    
    def __init__(self, config: Dict):
        """Initialize object detector"""
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        self.input_size = tuple(config['model']['input_size'])
        self.num_classes = config['model']['num_classes']
        self.class_names = config['data']['classes']
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Move to device
        self.model.to(self.device)
        
        # TensorRT optimization if enabled
        if config['optimization']['tensorrt']:
            self.model = self._optimize_tensorrt()
        
        # Detection threshold
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        logger.info(f"Object detector initialized on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load SSD with MobileNetV3-Small model"""
        try:
            # Import SSD model
            from torchvision.models.detection import ssd300_v2
            from torchvision.models.detection.ssd import SSDHead
            
            # Load pretrained SSD model
            model = ssd300_v2(pretrained=True)
            
            # Modify for our classes (background, traffic_sign, lane_marking)
            in_channels = 256
            num_anchors = [4, 6, 6, 6, 4, 4]  # Number of anchors per feature map
            
            # Create new classification head
            cls_head = nn.ModuleList()
            for anchors in num_anchors:
                cls_head.append(nn.Conv2d(in_channels, anchors * self.num_classes, kernel_size=3, padding=1))
            
            # Create new regression head
            bbox_head = nn.ModuleList()
            for anchors in num_anchors:
                bbox_head.append(nn.Conv2d(in_channels, anchors * 4, kernel_size=3, padding=1))
            
            # Replace heads
            model.head.classification_head.module_list = cls_head
            model.head.regression_head.module_list = bbox_head
            
            logger.info("SSD with MobileNetV3-Small model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading detection model: {e}")
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
            
            logger.info("TensorRT optimization completed for detection model")
            return trt_model
            
        except ImportError:
            logger.warning("TensorRT not available, using original detection model")
            return self.model
        except Exception as e:
            logger.warning(f"TensorRT optimization failed for detection: {e}")
            return self.model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for object detection"""
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
    
    def postprocess(self, detections: List[Dict[str, torch.Tensor]], 
                   original_size: Tuple[int, int]) -> List[Dict[str, np.ndarray]]:
        """Postprocess detection results"""
        processed_detections = []
        
        for detection in detections:
            boxes = detection['boxes'].cpu().numpy()
            scores = detection['scores'].cpu().numpy()
            labels = detection['labels'].cpu().numpy()
            
            # Filter by confidence threshold
            mask = scores > self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Apply NMS
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(), scores.tolist(), 
                    self.confidence_threshold, self.nms_threshold
                )
                
                if len(indices) > 0:
                    indices = indices.flatten()
                    boxes = boxes[indices]
                    scores = scores[indices]
                    labels = labels[indices]
                    
                    # Scale boxes to original image size
                    h, w = original_size
                    input_h, input_w = self.input_size
                    
                    boxes[:, [0, 2]] *= w / input_w
                    boxes[:, [1, 3]] *= h / input_h
                    
                    # Convert to list of detections
                    for box, score, label in zip(boxes, scores, labels):
                        processed_detections.append({
                            'bbox': box.astype(np.float32),
                            'score': float(score),
                            'class_id': int(label),
                            'class_name': self.class_names.get(int(label), 'unknown')
                        })
        
        return processed_detections
    
    def detect(self, image: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Detect objects in image"""
        try:
            with torch.no_grad():
                # Preprocess
                input_tensor = self.preprocess(image)
                
                # Inference
                detections = self.model(input_tensor)
                
                # Postprocess
                processed_detections = self.postprocess(detections, image.shape[:2])
                
                return processed_detections
                
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    def detect_batch(self, images: np.ndarray) -> List[List[Dict[str, np.ndarray]]]:
        """Detect objects in a batch of images"""
        batch_detections = []
        
        for image in images:
            detections = self.detect(image)
            batch_detections.append(detections)
        
        return batch_detections
    
    def filter_by_class(self, detections: List[Dict], class_names: List[str]) -> List[Dict]:
        """Filter detections by class names"""
        filtered = []
        for detection in detections:
            if detection['class_name'] in class_names:
                filtered.append(detection)
        return filtered
    
    def get_traffic_signs(self, detections: List[Dict]) -> List[Dict]:
        """Get traffic sign detections"""
        return self.filter_by_class(detections, ['traffic_sign'])
    
    def get_lane_markings(self, detections: List[Dict]) -> List[Dict]:
        """Get lane marking detections"""
        return self.filter_by_class(detections, ['lane_marking'])
    
    def evaluate_accuracy(self, pred_detections: List[Dict], 
                         gt_detections: List[Dict], 
                         iou_threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate detection accuracy"""
        # Calculate metrics
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        # Match predictions to ground truth
        matched_gt = set()
        
        for pred in pred_detections:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_detections):
                if i in matched_gt:
                    continue
                
                if pred['class_id'] == gt['class_id']:
                    iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(gt_detections) - len(matched_gt)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mAP = precision  # Simplified mAP calculation
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mAP': mAP,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Detection model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Detection model loaded from {path}")


class DetectionLoss(nn.Module):
    """Custom loss function for object detection"""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred_cls: torch.Tensor, pred_bbox: torch.Tensor,
                gt_cls: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
        """Compute detection loss"""
        # Classification loss (Cross-entropy)
        cls_loss = nn.functional.cross_entropy(pred_cls, gt_cls)
        
        # Localization loss (Smooth L1)
        loc_loss = nn.functional.smooth_l1_loss(pred_bbox, gt_bbox)
        
        # Total loss
        total_loss = (
            self.weights['classification_loss'] * cls_loss +
            self.weights['localization_loss'] * loc_loss
        )
        
        return total_loss 