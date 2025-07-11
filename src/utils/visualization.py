"""
Visualization Module
Real-time visualization of depth maps, detections, and 3D reconstructions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Visualizer:
    """Real-time visualization for Contour system"""
    
    def __init__(self, config: Dict):
        """Initialize visualizer"""
        self.config = config
        self.window_size = tuple(config.get('window_size', [1280, 720]))
        self.font_scale = config.get('font_scale', 0.6)
        self.thickness = config.get('thickness', 2)
        
        # Color maps
        self.depth_colormap = cv2.COLORMAP_JET
        self.detection_colors = {
            'traffic_sign': (0, 255, 0),    # Green
            'lane_marking': (255, 0, 0),    # Blue
            'background': (128, 128, 128)   # Gray
        }
        
        # Initialize display
        self._setup_display()
        
        logger.info("Visualizer initialized")
    
    def _setup_display(self):
        """Setup display windows"""
        try:
            # Create main window
            cv2.namedWindow('Contour - 3D Road Mapping', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Contour - 3D Road Mapping', *self.window_size)
            
            # Create sub-windows
            cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
            cv2.namedWindow('3D Reconstruction', cv2.WINDOW_NORMAL)
            
        except Exception as e:
            logger.warning(f"Could not setup display windows: {e}")
    
    def visualize(self, results: Dict) -> np.ndarray:
        """Create comprehensive visualization from results"""
        try:
            # Extract results
            frame = results.get('frame', np.zeros((480, 640, 3), dtype=np.uint8))
            depth_map = results.get('depth_map', None)
            detections = results.get('detections', [])
            point_cloud = results.get('point_cloud', None)
            location = results.get('location', {})
            fps = results.get('fps', 0)
            processing_time = results.get('processing_time', 0)
            
            # Create main visualization
            main_vis = self._create_main_visualization(
                frame, depth_map, detections, location, fps, processing_time
            )
            
            # Create sub-visualizations
            if depth_map is not None:
                depth_vis = self._visualize_depth_map(depth_map)
                cv2.imshow('Depth Map', depth_vis)
            
            if detections:
                detection_vis = self._visualize_detections(frame, detections)
                cv2.imshow('Detections', detection_vis)
            
            if point_cloud is not None and len(point_cloud) > 0:
                reconstruction_vis = self._visualize_3d_reconstruction(point_cloud)
                cv2.imshow('3D Reconstruction', reconstruction_vis)
            
            return main_vis
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _create_main_visualization(self, frame: np.ndarray, depth_map: Optional[np.ndarray],
                                 detections: List[Dict], location: Dict, 
                                 fps: float, processing_time: float) -> np.ndarray:
        """Create main visualization with all components"""
        # Create a larger canvas
        canvas_height = frame.shape[0] * 2
        canvas_width = frame.shape[1] * 2
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Place original frame
        canvas[:frame.shape[0], :frame.shape[1]] = frame
        
        # Place depth map if available
        if depth_map is not None:
            depth_vis = self._visualize_depth_map(depth_map)
            canvas[:frame.shape[0], frame.shape[1]:frame.shape[1]*2] = depth_vis
        
        # Place detections
        detection_vis = self._visualize_detections(frame, detections)
        canvas[frame.shape[0]:frame.shape[0]*2, :frame.shape[1]] = detection_vis
        
        # Place 3D reconstruction preview
        reconstruction_vis = self._create_reconstruction_preview(location)
        canvas[frame.shape[0]:frame.shape[0]*2, frame.shape[1]:frame.shape[1]*2] = reconstruction_vis
        
        # Add overlays
        self._add_performance_overlay(canvas, fps, processing_time)
        self._add_location_overlay(canvas, location)
        
        return canvas
    
    def _visualize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Visualize depth map with color mapping"""
        try:
            # Normalize depth map to [0, 255]
            depth_normalized = ((depth_map - depth_map.min()) / 
                              (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            
            # Apply color map
            depth_colored = cv2.applyColorMap(depth_normalized, self.depth_colormap)
            
            # Add depth scale
            depth_colored = self._add_depth_scale(depth_colored, depth_map)
            
            return depth_colored
            
        except Exception as e:
            logger.error(f"Error visualizing depth map: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _add_depth_scale(self, depth_vis: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Add depth scale to visualization"""
        try:
            height, width = depth_vis.shape[:2]
            
            # Create scale bar
            scale_height = height - 40
            scale_width = 20
            scale_x = width - 30
            
            # Draw scale bar
            for i in range(scale_height):
                # Map position to depth value
                depth_val = depth_map.min() + (depth_map.max() - depth_map.min()) * (1 - i / scale_height)
                color_val = int((depth_val - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255)
                color = tuple(map(int, cv2.applyColorMap(np.array([[color_val]]), self.depth_colormap)[0, 0]))
                
                cv2.line(depth_vis, (scale_x, 20 + i), (scale_x + scale_width, 20 + i), color, 1)
            
            # Add labels
            cv2.putText(depth_vis, f"{depth_map.max():.1f}m", (scale_x + 25, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(depth_vis, f"{depth_map.min():.1f}m", (scale_x + 25, height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return depth_vis
            
        except Exception as e:
            logger.warning(f"Could not add depth scale: {e}")
            return depth_vis
    
    def _visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Visualize object detections"""
        vis_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            score = detection['score']
            
            # Get color for class
            color = self.detection_colors.get(class_name, (0, 255, 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness)[0]
            
            # Draw label background
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.thickness)
        
        return vis_frame
    
    def _create_reconstruction_preview(self, location: Dict) -> np.ndarray:
        """Create 3D reconstruction preview"""
        # Create a placeholder for 3D reconstruction
        preview = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add GPS coordinates if available
        if location.get('gps'):
            lat, lon = location['gps']
            confidence = location.get('confidence', 0)
            
            # Draw GPS info
            cv2.putText(preview, "GPS Location:", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(preview, f"Lat: {lat:.6f}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(preview, f"Lon: {lon:.6f}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(preview, f"Confidence: {confidence:.2f}", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add 3D reconstruction placeholder
        cv2.putText(preview, "3D Reconstruction", (20, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(preview, "Point Cloud & Mesh", (20, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return preview
    
    def _visualize_3d_reconstruction(self, point_cloud: np.ndarray) -> np.ndarray:
        """Visualize 3D reconstruction from point cloud"""
        try:
            # Create a 2D projection of the point cloud
            if len(point_cloud) == 0:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Extract 3D points and colors
            points_3d = point_cloud[:, :3]
            colors = point_cloud[:, 3:].astype(np.uint8)
            
            # Create 2D projection (top-down view)
            x = points_3d[:, 0]
            y = points_3d[:, 2]  # Use Z as Y for top-down view
            
            # Normalize to image coordinates
            x_norm = ((x - x.min()) / (x.max() - x.min()) * 640).astype(int)
            y_norm = ((y - y.min()) / (y.max() - y.min()) * 480).astype(int)
            
            # Create visualization
            vis = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw points
            for i in range(len(x_norm)):
                if 0 <= x_norm[i] < 640 and 0 <= y_norm[i] < 480:
                    color = tuple(map(int, colors[i]))
                    cv2.circle(vis, (x_norm[i], y_norm[i]), 1, color, -1)
            
            # Add title
            cv2.putText(vis, "3D Point Cloud (Top View)", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis, f"Points: {len(point_cloud)}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return vis
            
        except Exception as e:
            logger.error(f"Error visualizing 3D reconstruction: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _add_performance_overlay(self, canvas: np.ndarray, fps: float, processing_time: float):
        """Add performance metrics overlay"""
        # FPS counter
        cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Processing time
        cv2.putText(canvas, f"Time: {processing_time*1000:.1f}ms", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Performance indicator
        if fps >= 20:
            color = (0, 255, 0)  # Green
        elif fps >= 15:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(canvas, "Performance", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _add_location_overlay(self, canvas: np.ndarray, location: Dict):
        """Add location information overlay"""
        if location.get('gps'):
            lat, lon = location['gps']
            confidence = location.get('confidence', 0)
            
            # Draw location info
            cv2.putText(canvas, "Location:", (canvas.shape[1] - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, f"Lat: {lat:.4f}", (canvas.shape[1] - 200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(canvas, f"Lon: {lon:.4f}", (canvas.shape[1] - 200, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(canvas, f"Conf: {confidence:.2f}", (canvas.shape[1] - 200, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_visualization(self, visualization: np.ndarray, filepath: str):
        """Save visualization to file"""
        try:
            cv2.imwrite(filepath, visualization)
            logger.info(f"Visualization saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
    
    def create_comparison_visualization(self, original: np.ndarray, 
                                      depth_map: np.ndarray, 
                                      detections: List[Dict]) -> np.ndarray:
        """Create side-by-side comparison visualization"""
        # Create comparison layout
        height, width = original.shape[:2]
        comparison = np.zeros((height, width * 3, 3), dtype=np.uint8)
        
        # Original image
        comparison[:, :width] = original
        
        # Depth map
        depth_vis = self._visualize_depth_map(depth_map)
        comparison[:, width:width*2] = cv2.resize(depth_vis, (width, height))
        
        # Detections
        detection_vis = self._visualize_detections(original, detections)
        comparison[:, width*2:] = detection_vis
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(comparison, "Depth Map", (width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(comparison, "Detections", (width*2 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return comparison
    
    def create_metrics_dashboard(self, metrics: Dict) -> np.ndarray:
        """Create metrics dashboard visualization"""
        # Create dashboard
        dashboard = np.zeros((400, 600, 3), dtype=np.uint8)
        
        y_offset = 40
        for metric_name, metric_value in metrics.items():
            # Draw metric name
            cv2.putText(dashboard, f"{metric_name}:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw metric value
            if isinstance(metric_value, float):
                value_str = f"{metric_value:.3f}"
            else:
                value_str = str(metric_value)
            
            cv2.putText(dashboard, value_str, (300, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y_offset += 40
        
        # Add title
        cv2.putText(dashboard, "Contour System Metrics", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return dashboard 