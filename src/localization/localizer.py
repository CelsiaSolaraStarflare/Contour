"""
Localization Module
Uses FAISS for GPS-based localization with high accuracy
Target: 95.1% accuracy within 1 meter on NVIDIA Jetson Nano
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import faiss
import exifread
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Localizer:
    """GPS-based localization using FAISS index"""
    
    def __init__(self, config: Dict):
        """Initialize localizer"""
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Feature extraction model
        self.feature_extractor = self._load_feature_extractor()
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        
        # FAISS index
        self.index = None
        self.gps_coordinates = []
        self.image_paths = []
        
        # Localization parameters
        self.search_k = config.get('search_k', 10)
        self.distance_threshold = config.get('distance_threshold', 1.0)  # 1 meter
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        
        # Load pre-built index if available
        self._load_index()
        
        logger.info(f"Localizer initialized on {self.device}")
    
    def _load_feature_extractor(self) -> nn.Module:
        """Load Vision Transformer for feature extraction"""
        try:
            import timm
            
            # Load ViT-B/16 model
            model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=0  # Remove classification head
            )
            
            logger.info("Vision Transformer feature extractor loaded successfully")
            return model
            
        except ImportError:
            logger.error("timm not found. Please install: pip install timm")
            raise
        except Exception as e:
            logger.error(f"Error loading feature extractor: {e}")
            raise
    
    def _load_index(self):
        """Load pre-built FAISS index"""
        index_path = self.config.get('index_path', 'models/localization_index.faiss')
        metadata_path = self.config.get('metadata_path', 'models/localization_metadata.pkl')
        
        if Path(index_path).exists() and Path(metadata_path).exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.gps_coordinates = metadata['gps_coordinates']
                    self.image_paths = metadata['image_paths']
                
                logger.info(f"Loaded FAISS index with {len(self.gps_coordinates)} entries")
                
            except Exception as e:
                logger.warning(f"Failed to load pre-built index: {e}")
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image using Vision Transformer"""
        try:
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            
            with torch.no_grad():
                # Extract features
                features = self.feature_extractor(input_tensor)
                
                # Normalize features
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                
                return features.cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros((1, 768), dtype=np.float32)  # ViT-B/16 feature dimension
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for Vision Transformer"""
        # Resize to 224x224 (ViT input size)
        image_resized = cv2.resize(image, (224, 224))
        
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
    
    def localize(self, features: np.ndarray) -> Dict:
        """Localize image using FAISS index"""
        if self.index is None:
            logger.warning("No FAISS index available for localization")
            return {
                'gps': None,
                'confidence': 0.0,
                'distance': float('inf'),
                'matches': []
            }
        
        try:
            # Search in FAISS index
            distances, indices = self.index.search(features, self.search_k)
            
            # Get GPS coordinates of matches
            matches = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.gps_coordinates):
                    gps = self.gps_coordinates[idx]
                    image_path = self.image_paths[idx] if idx < len(self.image_paths) else None
                    
                    matches.append({
                        'gps': gps,
                        'distance': float(distance),
                        'image_path': image_path,
                        'rank': i
                    })
            
            # Get best match
            if matches:
                best_match = matches[0]
                confidence = self._calculate_confidence(best_match['distance'])
                
                return {
                    'gps': best_match['gps'],
                    'confidence': confidence,
                    'distance': best_match['distance'],
                    'matches': matches
                }
            else:
                return {
                    'gps': None,
                    'confidence': 0.0,
                    'distance': float('inf'),
                    'matches': []
                }
                
        except Exception as e:
            logger.error(f"Error in localization: {e}")
            return {
                'gps': None,
                'confidence': 0.0,
                'distance': float('inf'),
                'matches': []
            }
    
    def _calculate_confidence(self, distance: float) -> float:
        """Calculate confidence based on feature distance"""
        # Exponential decay based on distance
        confidence = np.exp(-distance / 0.5)  # 0.5 is the decay factor
        return min(confidence, 1.0)
    
    def build_index(self, image_dir: str, gps_file: str):
        """Build FAISS index from training data"""
        logger.info("Building FAISS index for localization...")
        
        # Load GPS coordinates
        gps_data = self._load_gps_data(gps_file)
        
        # Extract features from all images
        features_list = []
        gps_list = []
        image_paths_list = []
        
        image_dir_path = Path(image_dir)
        image_files = list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png"))
        
        for image_file in tqdm(image_files, desc="Extracting features"):
            try:
                # Load image
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                # Get GPS coordinates
                gps = self._get_gps_from_image(image_file, gps_data)
                if gps is None:
                    continue
                
                # Extract features
                features = self.extract_features(image)
                
                features_list.append(features)
                gps_list.append(gps)
                image_paths_list.append(str(image_file))
                
            except Exception as e:
                logger.warning(f"Error processing {image_file}: {e}")
                continue
        
        if not features_list:
            logger.error("No valid features extracted")
            return
        
        # Stack features
        all_features = np.vstack(features_list)
        
        # Create FAISS index
        dimension = all_features.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add features to index
        self.index.add(all_features.astype(np.float32))
        
        # Store metadata
        self.gps_coordinates = gps_list
        self.image_paths = image_paths_list
        
        # Save index and metadata
        self._save_index()
        
        logger.info(f"FAISS index built with {len(features_list)} entries")
    
    def _load_gps_data(self, gps_file: str) -> Dict:
        """Load GPS data from file"""
        try:
            with open(gps_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading GPS data: {e}")
            return {}
    
    def _get_gps_from_image(self, image_path: Path, gps_data: Dict) -> Optional[Tuple[float, float]]:
        """Get GPS coordinates from image EXIF or GPS data file"""
        # Try to get from GPS data file first
        image_name = image_path.name
        if image_name in gps_data:
            return gps_data[image_name]
        
        # Try to extract from EXIF
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
            
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                lat = self._convert_to_degrees(tags['GPS GPSLatitude'].values)
                lon = self._convert_to_degrees(tags['GPS GPSLongitude'].values)
                
                # Apply hemisphere
                if 'GPS GPSLatitudeRef' in tags and tags['GPS GPSLatitudeRef'].values == 'S':
                    lat = -lat
                if 'GPS GPSLongitudeRef' in tags and tags['GPS GPSLongitudeRef'].values == 'W':
                    lon = -lon
                
                return (lat, lon)
                
        except Exception as e:
            logger.debug(f"Could not extract GPS from EXIF for {image_path}: {e}")
        
        return None
    
    def _convert_to_degrees(self, values) -> float:
        """Convert GPS coordinates to decimal degrees"""
        degrees = float(values[0].num) / float(values[0].den)
        minutes = float(values[1].num) / float(values[1].den)
        seconds = float(values[2].num) / float(values[2].den)
        
        return degrees + minutes / 60.0 + seconds / 3600.0
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            # Save FAISS index
            index_path = self.config.get('index_path', 'models/localization_index.faiss')
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = self.config.get('metadata_path', 'models/localization_metadata.pkl')
            metadata = {
                'gps_coordinates': self.gps_coordinates,
                'image_paths': self.image_paths
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index saved to {index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def evaluate_accuracy(self, test_images: List[str], 
                         test_gps: List[Tuple[float, float]]) -> Dict[str, float]:
        """Evaluate localization accuracy"""
        if self.index is None:
            return {'accuracy': 0.0, 'mean_distance': float('inf')}
        
        correct_predictions = 0
        distances = []
        
        for image_path, true_gps in zip(test_images, test_gps):
            try:
                # Load and process image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Extract features and localize
                features = self.extract_features(image)
                result = self.localize(features)
                
                if result['gps'] is not None:
                    # Calculate distance to true GPS
                    distance = self._calculate_gps_distance(result['gps'], true_gps)
                    distances.append(distance)
                    
                    # Check if within threshold
                    if distance <= self.distance_threshold:
                        correct_predictions += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating {image_path}: {e}")
                continue
        
        if not distances:
            return {'accuracy': 0.0, 'mean_distance': float('inf')}
        
        accuracy = correct_predictions / len(distances)
        mean_distance = np.mean(distances)
        
        return {
            'accuracy': accuracy,
            'mean_distance': mean_distance,
            'correct_predictions': correct_predictions,
            'total_predictions': len(distances)
        }
    
    def _calculate_gps_distance(self, gps1: Tuple[float, float], 
                               gps2: Tuple[float, float]) -> float:
        """Calculate distance between two GPS coordinates in meters"""
        # Haversine formula
        lat1, lon1 = np.radians(gps1)
        lat2, lon2 = np.radians(gps2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in meters
        r = 6371000
        
        return r * c
    
    def refine_pose(self, features: np.ndarray, 
                   initial_gps: Tuple[float, float]) -> Tuple[float, float]:
        """Refine camera pose using feature matching"""
        # This is a simplified pose refinement
        # In practice, you would use more sophisticated methods like PnP
        
        # For now, return the initial GPS coordinates
        return initial_gps


class LocalizationLoss(nn.Module):
    """Loss function for localization training"""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
    
    def forward(self, pred_features: torch.Tensor, 
                gt_features: torch.Tensor) -> torch.Tensor:
        """Compute localization loss"""
        # Cosine similarity loss
        similarity_loss = 1.0 - torch.nn.functional.cosine_similarity(
            pred_features, gt_features, dim=1
        ).mean()
        
        # Triplet loss for better feature discrimination
        triplet_loss = self._triplet_loss(pred_features, gt_features)
        
        # Total loss
        total_loss = (
            self.weights.get('similarity', 1.0) * similarity_loss +
            self.weights.get('triplet', 0.1) * triplet_loss
        )
        
        return total_loss
    
    def _triplet_loss(self, pred_features: torch.Tensor, 
                     gt_features: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss for feature learning"""
        # Simplified triplet loss
        # In practice, you would use proper triplet mining
        
        # Random negative samples
        batch_size = pred_features.shape[0]
        neg_indices = torch.randperm(batch_size)
        
        anchor = pred_features
        positive = gt_features
        negative = pred_features[neg_indices]
        
        # Triplet loss
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        margin = 1.0
        triplet_loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()
        
        return triplet_loss 