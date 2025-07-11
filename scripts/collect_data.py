#!/usr/bin/env python3
"""
Data Collection Script
Collects training data for Contour system with GPS metadata
Target: 3,000 road images with diverse conditions
"""

import argparse
import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import exifread
from typing import Dict, List, Optional, Tuple
import threading
import queue

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.camera import Camera, CameraRecorder
from src.utils.jetson_utils import setup_jetson

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Data collector for Contour training dataset"""
    
    def __init__(self, config: Dict):
        """Initialize data collector"""
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / 'images'
        self.metadata_dir = self.output_dir / 'metadata'
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Initialize camera
        self.camera = Camera(config['camera'])
        
        # Data collection parameters
        self.target_count = config.get('target_count', 3000)
        self.capture_interval = config.get('capture_interval', 2.0)  # seconds
        self.image_quality = config.get('image_quality', 95)
        
        # GPS simulation (in real deployment, use actual GPS)
        self.gps_enabled = config.get('gps_enabled', True)
        self.gps_data = {}
        
        # Collection state
        self.is_collecting = False
        self.collected_count = 0
        self.start_time = None
        
        logger.info(f"Data collector initialized. Target: {self.target_count} images")
    
    def start_collection(self):
        """Start data collection"""
        self.is_collecting = True
        self.start_time = time.time()
        
        logger.info("Starting data collection...")
        
        try:
            while self.is_collecting and self.collected_count < self.target_count:
                # Capture frame
                frame = self.camera.capture()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    time.sleep(1)
                    continue
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_filename = f"road_{timestamp}_{self.collected_count:06d}.jpg"
                image_path = self.images_dir / image_filename
                
                # Save image
                success = cv2.imwrite(str(image_path), frame, 
                                    [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
                
                if success:
                    # Generate metadata
                    metadata = self._generate_metadata(image_filename, frame)
                    
                    # Save metadata
                    metadata_filename = f"{image_filename.replace('.jpg', '.json')}"
                    metadata_path = self.metadata_dir / metadata_filename
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Update GPS data
                    if self.gps_enabled:
                        self.gps_data[image_filename] = metadata['gps']
                    
                    self.collected_count += 1
                    
                    # Log progress
                    if self.collected_count % 100 == 0:
                        elapsed_time = time.time() - self.start_time
                        rate = self.collected_count / elapsed_time
                        remaining = (self.target_count - self.collected_count) / rate
                        
                        logger.info(f"Collected {self.collected_count}/{self.target_count} images "
                                  f"({rate:.1f} images/sec, {remaining:.0f}s remaining)")
                
                # Wait for next capture
                time.sleep(self.capture_interval)
        
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user")
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
        finally:
            self.stop_collection()
    
    def stop_collection(self):
        """Stop data collection"""
        self.is_collecting = False
        logger.info(f"Data collection stopped. Total collected: {self.collected_count}")
        
        # Save GPS data summary
        if self.gps_data:
            gps_summary_path = self.output_dir / 'gps_data.json'
            with open(gps_summary_path, 'w') as f:
                json.dump(self.gps_data, f, indent=2)
            logger.info(f"GPS data saved to {gps_summary_path}")
    
    def _generate_metadata(self, filename: str, frame: np.ndarray) -> Dict:
        """Generate metadata for captured image"""
        metadata = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'image_size': {
                'width': frame.shape[1],
                'height': frame.shape[0],
                'channels': frame.shape[2]
            },
            'camera_info': {
                'type': self.camera.camera_type,
                'resolution': self.camera.resolution,
                'fps': self.camera.fps
            },
            'collection_info': {
                'count': self.collected_count,
                'target_count': self.target_count,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0
            }
        }
        
        # Add GPS data
        if self.gps_enabled:
            metadata['gps'] = self._simulate_gps()
        else:
            metadata['gps'] = None
        
        # Add image statistics
        metadata['image_stats'] = {
            'mean_brightness': float(np.mean(frame)),
            'std_brightness': float(np.std(frame)),
            'min_value': int(np.min(frame)),
            'max_value': int(np.max(frame))
        }
        
        return metadata
    
    def _simulate_gps(self) -> Tuple[float, float]:
        """Simulate GPS coordinates (replace with actual GPS in deployment)"""
        # Simulate movement along a road
        base_lat = 37.7749  # San Francisco
        base_lon = -122.4194
        
        # Add some variation based on collection count
        lat_offset = (self.collected_count * 0.0001) % 0.01  # ~10m increments
        lon_offset = (self.collected_count * 0.0001) % 0.01
        
        # Add some noise
        lat_noise = np.random.normal(0, 0.00001)  # ~1m noise
        lon_noise = np.random.normal(0, 0.00001)
        
        lat = base_lat + lat_offset + lat_noise
        lon = base_lon + lon_offset + lon_noise
        
        return (lat, lon)
    
    def collect_from_directory(self, input_dir: str):
        """Collect data from existing directory of images"""
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return
        
        # Get image files
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        
        for i, image_file in enumerate(image_files):
            if self.collected_count >= self.target_count:
                break
            
            try:
                # Load image
                frame = cv2.imread(str(image_file))
                if frame is None:
                    logger.warning(f"Could not load image: {image_file}")
                    continue
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                new_filename = f"road_{timestamp}_{self.collected_count:06d}.jpg"
                new_path = self.images_dir / new_filename
                
                # Copy image
                success = cv2.imwrite(str(new_path), frame, 
                                    [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
                
                if success:
                    # Generate metadata
                    metadata = self._generate_metadata(new_filename, frame)
                    
                    # Save metadata
                    metadata_filename = f"{new_filename.replace('.jpg', '.json')}"
                    metadata_path = self.metadata_dir / metadata_filename
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Update GPS data
                    if self.gps_enabled:
                        self.gps_data[new_filename] = metadata['gps']
                    
                    self.collected_count += 1
                    
                    if self.collected_count % 100 == 0:
                        logger.info(f"Processed {self.collected_count}/{min(len(image_files), self.target_count)} images")
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
        
        logger.info(f"Directory collection completed. Total collected: {self.collected_count}")
    
    def extract_gps_from_exif(self, image_path: str) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from image EXIF data"""
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
    
    def create_dataset_summary(self) -> Dict:
        """Create summary of collected dataset"""
        summary = {
            'total_images': self.collected_count,
            'target_count': self.target_count,
            'collection_time': time.time() - self.start_time if self.start_time else 0,
            'output_directory': str(self.output_dir),
            'image_directory': str(self.images_dir),
            'metadata_directory': str(self.metadata_dir),
            'gps_enabled': self.gps_enabled,
            'camera_info': {
                'type': self.camera.camera_type,
                'resolution': self.camera.resolution,
                'fps': self.camera.fps
            }
        }
        
        # Calculate collection rate
        if summary['collection_time'] > 0:
            summary['collection_rate'] = self.collected_count / summary['collection_time']
        
        # Add GPS statistics if available
        if self.gps_data:
            lats = [coord[0] for coord in self.gps_data.values()]
            lons = [coord[1] for coord in self.gps_data.values()]
            
            summary['gps_stats'] = {
                'min_lat': min(lats),
                'max_lat': max(lats),
                'min_lon': min(lons),
                'max_lon': max(lons),
                'center_lat': sum(lats) / len(lats),
                'center_lon': sum(lons) / len(lons)
            }
        
        return summary


def collect_from_iphone(config: Dict):
    """Collect data from iPhone (simulated)"""
    logger.info("iPhone data collection mode")
    
    # For now, we'll simulate iPhone collection
    # In practice, you would use AirPlay or similar for iPhone streaming
    
    collector = DataCollector(config)
    
    # Simulate iPhone collection by using webcam with different settings
    collector.camera.resolution = (1920, 1080)  # Higher resolution for iPhone
    collector.camera.fps = 30
    
    logger.info("Starting iPhone data collection (simulated)...")
    collector.start_collection()


def collect_from_webcam(config: Dict):
    """Collect data from webcam"""
    logger.info("Webcam data collection mode")
    
    collector = DataCollector(config)
    
    logger.info("Starting webcam data collection...")
    collector.start_collection()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Collect training data for Contour system")
    parser.add_argument('--camera', type=str, choices=['webcam', 'iphone'], default='webcam',
                       help='Camera source')
    parser.add_argument('--output', type=str, default='data/training',
                       help='Output directory for collected data')
    parser.add_argument('--target', type=int, default=3000,
                       help='Target number of images to collect')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Capture interval in seconds')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality (1-100)')
    parser.add_argument('--gps', action='store_true',
                       help='Enable GPS simulation')
    parser.add_argument('--from-dir', type=str,
                       help='Collect from existing directory instead of camera')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'camera': {
            'type': args.camera,
            'device_id': 0,
            'resolution': [640, 480],
            'fps': 30
        },
        'output_dir': args.output,
        'target_count': args.target,
        'capture_interval': args.interval,
        'image_quality': args.quality,
        'gps_enabled': args.gps
    }
    
    try:
        # Setup Jetson Nano
        setup_jetson()
        
        if args.from_dir:
            # Collect from existing directory
            collector = DataCollector(config)
            collector.collect_from_directory(args.from_dir)
        else:
            # Collect from camera
            if args.camera == 'iphone':
                collect_from_iphone(config)
            else:
                collect_from_webcam(config)
        
        # Create dataset summary
        collector = DataCollector(config)
        summary = collector.create_dataset_summary()
        
        # Save summary
        summary_path = Path(args.output) / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset summary saved to {summary_path}")
        logger.info(f"Collection completed: {summary['total_images']} images")
        
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise


if __name__ == "__main__":
    main() 