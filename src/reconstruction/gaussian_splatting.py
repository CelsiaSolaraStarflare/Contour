"""
3D Reconstruction Module
Uses Gaussian Splatting for high-quality 3D reconstruction
Generates point clouds and meshes from depth maps
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

try:
    import gsplat
    from gsplat import rasterize_gaussians
    GS_AVAILABLE = True
except ImportError:
    GS_AVAILABLE = False
    logging.warning("gsplat not available. Install with: pip install gsplat")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    logging.warning("open3d not available. Install with: pip install open3d")

logger = logging.getLogger(__name__)


class GaussianSplatting:
    """3D reconstruction using Gaussian Splatting"""
    
    def __init__(self, config: Dict):
        """Initialize Gaussian Splatting reconstructor"""
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Check dependencies
        if not GS_AVAILABLE:
            raise ImportError("gsplat is required for 3D reconstruction")
        
        # Configuration parameters
        self.num_points = config.get('num_points', 100000)
        self.point_size = config.get('point_size', 0.01)
        self.splat_size = config.get('splat_size', 0.02)
        
        # Camera parameters
        self.fx = config.get('fx', 525.0)  # Focal length x
        self.fy = config.get('fy', 525.0)  # Focal length y
        self.cx = config.get('cx', 319.5)  # Principal point x
        self.cy = config.get('cy', 239.5)  # Principal point y
        
        logger.info(f"Gaussian Splatting reconstructor initialized on {self.device}")
    
    def depth_to_pointcloud(self, depth_map: np.ndarray, rgb_image: np.ndarray) -> np.ndarray:
        """Convert depth map to point cloud"""
        height, width = depth_map.shape
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Back-project to 3D
        z = depth_map
        x = (x - self.cx) * z / self.fx
        y = (y - self.cy) * z / self.fy
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=-1)
        
        # Filter valid points (non-zero depth)
        valid_mask = depth_map > 0
        valid_points = points[valid_mask]
        valid_colors = rgb_image[valid_mask]
        
        # Combine points and colors
        pointcloud = np.concatenate([valid_points, valid_colors], axis=1)
        
        return pointcloud
    
    def create_gaussians(self, pointcloud: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create Gaussian parameters from point cloud"""
        # Convert to tensor
        points = torch.from_numpy(pointcloud[:, :3]).float().to(self.device)
        colors = torch.from_numpy(pointcloud[:, 3:]).float().to(self.device) / 255.0
        
        # Initialize Gaussian parameters
        num_points = points.shape[0]
        
        # Positions
        positions = points
        
        # Colors (RGB)
        colors = colors
        
        # Opacities (alpha)
        opacities = torch.ones(num_points, 1, device=self.device) * 0.8
        
        # Scales (3D covariance)
        scales = torch.ones(num_points, 3, device=self.device) * self.splat_size
        
        # Rotations (quaternions)
        rotations = torch.zeros(num_points, 4, device=self.device)
        rotations[:, 0] = 1.0  # Identity quaternion
        
        return positions, colors, opacities, scales, rotations
    
    def reconstruct(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> Dict:
        """Reconstruct 3D scene from RGB image and depth map"""
        try:
            # Convert depth map to point cloud
            pointcloud = self.depth_to_pointcloud(depth_map, rgb_image)
            
            if len(pointcloud) == 0:
                logger.warning("No valid points in depth map")
                return {
                    'pointcloud': np.array([]),
                    'gaussians': None,
                    'mesh': None
                }
            
            # Sample points if too many
            if len(pointcloud) > self.num_points:
                indices = np.random.choice(len(pointcloud), self.num_points, replace=False)
                pointcloud = pointcloud[indices]
            
            # Create Gaussian parameters
            positions, colors, opacities, scales, rotations = self.create_gaussians(pointcloud)
            
            # Create mesh if open3d is available
            mesh = None
            if O3D_AVAILABLE:
                mesh = self._create_mesh(pointcloud)
            
            return {
                'pointcloud': pointcloud,
                'gaussians': {
                    'positions': positions,
                    'colors': colors,
                    'opacities': opacities,
                    'scales': scales,
                    'rotations': rotations
                },
                'mesh': mesh
            }
            
        except Exception as e:
            logger.error(f"Error in 3D reconstruction: {e}")
            return {
                'pointcloud': np.array([]),
                'gaussians': None,
                'mesh': None
            }
    
    def _create_mesh(self, pointcloud: np.ndarray) -> Optional[o3d.geometry.TriangleMesh]:
        """Create mesh from point cloud using Open3D"""
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:] / 255.0)
            
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # Create mesh using Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8, width=0, scale=1.1, linear_fit=False
            )
            
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Failed to create mesh: {e}")
            return None
    
    def render_gaussians(self, gaussians: Dict, camera_pose: np.ndarray, 
                        image_size: Tuple[int, int]) -> np.ndarray:
        """Render Gaussian splats to image"""
        try:
            # Extract Gaussian parameters
            positions = gaussians['positions']
            colors = gaussians['colors']
            opacities = gaussians['opacities']
            scales = gaussians['scales']
            rotations = gaussians['rotations']
            
            # Convert camera pose to tensor
            camera_pose = torch.from_numpy(camera_pose).float().to(self.device)
            
            # Rasterize Gaussians
            rendered_image = rasterize_gaussians(
                positions, colors, opacities, scales, rotations,
                camera_pose, image_size[0], image_size[1]
            )
            
            # Convert to numpy
            rendered_image = rendered_image.cpu().numpy()
            
            return rendered_image
            
        except Exception as e:
            logger.error(f"Error rendering Gaussians: {e}")
            return np.zeros((*image_size, 3), dtype=np.uint8)
    
    def save_pointcloud(self, pointcloud: np.ndarray, filepath: str):
        """Save point cloud to file"""
        try:
            if filepath.endswith('.ply'):
                self._save_ply(pointcloud, filepath)
            elif filepath.endswith('.pcd'):
                self._save_pcd(pointcloud, filepath)
            else:
                np.save(filepath, pointcloud)
            
            logger.info(f"Point cloud saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving point cloud: {e}")
    
    def _save_ply(self, pointcloud: np.ndarray, filepath: str):
        """Save point cloud as PLY file"""
        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:] / 255.0)
            o3d.io.write_point_cloud(filepath, pcd)
        else:
            # Manual PLY writing
            with open(filepath, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(pointcloud)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                for point in pointcloud:
                    f.write(f"{point[0]} {point[1]} {point[2]} {int(point[3])} {int(point[4])} {int(point[5])}\n")
    
    def _save_pcd(self, pointcloud: np.ndarray, filepath: str):
        """Save point cloud as PCD file"""
        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:] / 255.0)
            o3d.io.write_point_cloud(filepath, pcd)
    
    def save_mesh(self, mesh, filepath: str):
        """Save mesh to file"""
        try:
            if mesh is not None and O3D_AVAILABLE:
                o3d.io.write_triangle_mesh(filepath, mesh)
                logger.info(f"Mesh saved to {filepath}")
            else:
                logger.warning("No mesh available to save")
                
        except Exception as e:
            logger.error(f"Error saving mesh: {e}")
    
    def filter_pointcloud(self, pointcloud: np.ndarray, 
                         min_depth: float = 0.1, 
                         max_depth: float = 10.0) -> np.ndarray:
        """Filter point cloud by depth range"""
        depths = pointcloud[:, 2]
        mask = (depths >= min_depth) & (depths <= max_depth)
        return pointcloud[mask]
    
    def downsample_pointcloud(self, pointcloud: np.ndarray, 
                             voxel_size: float = 0.05) -> np.ndarray:
        """Downsample point cloud using voxel grid"""
        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:] / 255.0)
            
            # Downsample
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            
            # Convert back to numpy
            points = np.asarray(downsampled_pcd.points)
            colors = np.asarray(downsampled_pcd.colors) * 255
            
            return np.concatenate([points, colors], axis=1)
        else:
            # Simple random sampling
            if len(pointcloud) > self.num_points:
                indices = np.random.choice(len(pointcloud), self.num_points, replace=False)
                return pointcloud[indices]
            return pointcloud


class ReconstructionLoss(nn.Module):
    """Loss function for 3D reconstruction"""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
    
    def forward(self, pred_pointcloud: torch.Tensor, 
                gt_pointcloud: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss"""
        # Chamfer distance
        chamfer_loss = self._chamfer_distance(pred_pointcloud, gt_pointcloud)
        
        # Color loss
        color_loss = torch.mean(torch.abs(
            pred_pointcloud[:, 3:] - gt_pointcloud[:, 3:]
        ))
        
        # Total loss
        total_loss = (
            self.weights.get('chamfer', 1.0) * chamfer_loss +
            self.weights.get('color', 0.1) * color_loss
        )
        
        return total_loss
    
    def _chamfer_distance(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute Chamfer distance between point clouds"""
        # Compute pairwise distances
        dist_matrix = torch.cdist(pred[:, :3], gt[:, :3])
        
        # Forward and backward distances
        forward_dist = torch.min(dist_matrix, dim=1)[0].mean()
        backward_dist = torch.min(dist_matrix, dim=0)[0].mean()
        
        return forward_dist + backward_dist 