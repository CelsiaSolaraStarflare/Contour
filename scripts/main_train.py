"""
#!/usr/bin/env python3
"""
Main Training Script for Contour Project
Handles training for depth estimation and potentially other components
"""

import argparse
import logging
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Main training script for Contour")
    parser.add_argument('--config', type=str, default='configs/depth_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default='depth',
                        choices=['depth', 'detection', 'all'],
                        help='Model to train')
    
    args = parser.parse_args()
    
    try:
        if args.model == 'depth' or args.model == 'all':
            # Import and call depth training
            from train_depth import train_depth_model
            logger.info("Starting depth model training...")
            train_depth_model(args.config)
        
        # Add other models here in the future
        
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 