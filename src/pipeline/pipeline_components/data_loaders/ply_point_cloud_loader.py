from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pathlib
from plyfile import PlyData
from .abstract_data_loader import AbstractDataLoader


class PLYPointCloudLoader(AbstractDataLoader):
    """
    Data loader component for PLY point cloud files.
    Loads 3D points and optional RGB colors from PLY files.
    
    Args:
        ply_path: Path to PLY file
        load_colors: Whether to load RGB colors if available (default: True)
        
    Returns:
        Dictionary containing point cloud data
        
    Raises:
        FileNotFoundError: If PLY file doesn't exist
        ValueError: If PLY file format is invalid
    """
    
    def __init__(self, ply_path: str, load_colors: bool = True) -> None:
        super().__init__()
        self.ply_path = pathlib.Path(ply_path)
        self.load_colors = load_colors
        
        # Load point cloud data during initialization
        self._load_point_cloud_data()
        
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component is a data loader and doesn't require inputs."""
        return []

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs point cloud data."""
        return ["point_cloud"]

    def _run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Load and return point cloud data.
        
        Args:
            **kwargs: Unused arguments
            
        Returns:
            Dictionary containing point cloud data
        """
        return {
            "point_cloud": self._create_point_cloud_entity()
        }

    def _load_point_cloud_data(self) -> None:
        """Load point cloud data from PLY file."""
        if not self.ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {self.ply_path}")
            
        print(f"Loading point cloud from: {self.ply_path}")
        
        # Parse PLY file
        points, colors = self._parse_ply_file()
        
        # Store as instance variables
        self.points = points
        self.colors = colors
        
        print(f"Loaded {len(points)} points with {'colors' if colors is not None else 'no colors'}")

    def _parse_ply_file(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load point cloud from PLY file.
        
        Returns:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (if available)
        """
        ply_data = PlyData.read(str(self.ply_path))
        vertex_element = ply_data['vertex']
        
        # Access the actual data array
        vertices = vertex_element.data
        
        # Extract 3D coordinates
        points = np.column_stack([
            vertices['x'].astype(np.float32),
            vertices['y'].astype(np.float32), 
            vertices['z'].astype(np.float32)
        ])
        
        # Extract colors if available and requested
        colors = None
        if self.load_colors:
            try:
                # Check if color properties exist in the dtype
                if hasattr(vertices.dtype, 'names') and vertices.dtype.names is not None:
                    if 'red' in vertices.dtype.names:
                        colors = np.column_stack([
                            vertices['red'].astype(np.uint8),
                            vertices['green'].astype(np.uint8),
                            vertices['blue'].astype(np.uint8)
                        ])
            except Exception as e:
                print(f"Warning: Could not extract colors from PLY file: {e}")
                colors = None
        
        # Validate point data
        self._validate_point_cloud_data(points, colors)
        
        return points, colors

    def _validate_point_cloud_data(self, points: np.ndarray, colors: Optional[np.ndarray]) -> None:
        """Validate loaded point cloud data."""
        # Check for valid point coordinates
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            raise ValueError("Invalid point coordinates found (NaN or Inf)")
        
        # Check point cloud bounds
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        print(f"Point cloud bounds: min={min_coords}, max={max_coords}")
        
        # Check for reasonable coordinate ranges (warn if very large or very small)
        coord_range = max_coords - min_coords
        if np.any(coord_range > 10000):
            print("Warning: Point cloud has very large coordinate range (>10km)")
        if np.any(coord_range < 0.001):
            print("Warning: Point cloud has very small coordinate range (<1mm)")
        
        # Validate colors if present
        if colors is not None:
            if len(colors) != len(points):
                raise ValueError(f"Color array length ({len(colors)}) doesn't match points ({len(points)})")
            
            if colors.shape[1] != 3:
                raise ValueError(f"Color array must have 3 channels (RGB), got {colors.shape[1]}")
            
            # Check color value ranges
            if colors.dtype == np.uint8:
                if np.any(colors < 0) or np.any(colors > 255):
                    print("Warning: Color values outside valid range [0, 255]")
            elif colors.dtype == np.float32 or colors.dtype == np.float64:
                if np.any(colors < 0) or np.any(colors > 1):
                    print("Warning: Float color values outside valid range [0, 1]")

    def _create_point_cloud_entity(self):
        """
        Create a point cloud entity compatible with the pipeline.
        
        This creates a simple object that mimics the structure expected by
        the pipeline components.
        """
        class PointCloudEntity:
            def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
                self.point_cloud_numpy = points
                self.rgb_numpy = colors
                # Add empty confidence scores for compatibility
                self.confidence_scores_numpy = None
                
            def as_numpy(self):
                return self.point_cloud_numpy
        
        return PointCloudEntity(self.points, self.colors)

    def __iter__(self):
        """Make this component iterable for pipeline usage."""
        # For point cloud data, we typically want to yield the complete cloud
        # rather than individual points, so we yield once
        yield {
            "point_cloud": self._create_point_cloud_entity()
        } 