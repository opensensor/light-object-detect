from pydantic import BaseModel, Field, validator
from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon


class Point2D(BaseModel):
    """2D point with normalized coordinates."""
    
    x: float = Field(..., description="Normalized x coordinate (0-1)", ge=0.0, le=1.0)
    y: float = Field(..., description="Normalized y coordinate (0-1)", ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "x": 0.5,
                "y": 0.5
            }
        }


class DetectionZone(BaseModel):
    """Detection zone definition with polygon coordinates."""
    
    id: str = Field(..., description="Unique identifier for the zone")
    name: str = Field(..., description="Human-readable name for the zone")
    polygon: List[Point2D] = Field(..., description="List of points defining the polygon (min 3 points)")
    enabled: bool = Field(True, description="Whether the zone is active")
    filter_classes: Optional[List[str]] = Field(
        None, 
        description="List of class labels to detect in this zone (None = all classes)"
    )
    min_confidence: Optional[float] = Field(
        None,
        description="Minimum confidence threshold for this zone (overrides global threshold)",
        ge=0.0,
        le=1.0
    )
    
    @validator('polygon')
    def validate_polygon(cls, v):
        """Validate that polygon has at least 3 points."""
        if len(v) < 3:
            raise ValueError('Polygon must have at least 3 points')
        return v
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is inside the zone polygon.
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
            
        Returns:
            True if point is inside the polygon
        """
        # Convert polygon points to shapely format
        polygon_coords = [(p.x, p.y) for p in self.polygon]
        polygon = ShapelyPolygon(polygon_coords)
        point = Point(x, y)
        
        return polygon.contains(point)
    
    def contains_box(self, x_min: float, y_min: float, x_max: float, y_max: float,
                    mode: str = "center") -> bool:
        """
        Check if a bounding box intersects with the zone.
        
        Args:
            x_min: Normalized minimum x coordinate
            y_min: Normalized minimum y coordinate
            x_max: Normalized maximum x coordinate
            y_max: Normalized maximum y coordinate
            mode: Detection mode - "center" (box center must be in zone),
                  "any" (any part of box in zone), "all" (entire box in zone)
            
        Returns:
            True if box meets the zone criteria based on mode
        """
        polygon_coords = [(p.x, p.y) for p in self.polygon]
        polygon = ShapelyPolygon(polygon_coords)
        
        if mode == "center":
            # Check if center of box is in zone
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            return polygon.contains(Point(center_x, center_y))
        
        elif mode == "any":
            # Check if any part of box intersects with zone
            box_polygon = ShapelyPolygon([
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max)
            ])
            return polygon.intersects(box_polygon)
        
        elif mode == "all":
            # Check if entire box is within zone
            box_polygon = ShapelyPolygon([
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max)
            ])
            return polygon.contains(box_polygon)
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'center', 'any', or 'all'")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "zone1",
                "name": "Entrance Area",
                "polygon": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.5, "y": 0.1},
                    {"x": 0.5, "y": 0.5},
                    {"x": 0.1, "y": 0.5}
                ],
                "enabled": True,
                "filter_classes": ["person", "car"],
                "min_confidence": 0.6
            }
        }


class ZoneConfiguration(BaseModel):
    """Configuration for multiple detection zones."""
    
    zones: List[DetectionZone] = Field(default_factory=list, description="List of detection zones")
    zone_mode: str = Field(
        "center",
        description="How to determine if detection is in zone: 'center', 'any', or 'all'"
    )
    
    @validator('zone_mode')
    def validate_zone_mode(cls, v):
        """Validate zone mode."""
        if v not in ["center", "any", "all"]:
            raise ValueError("zone_mode must be 'center', 'any', or 'all'")
        return v
    
    def get_zone_for_detection(self, x_min: float, y_min: float, 
                               x_max: float, y_max: float) -> Optional[DetectionZone]:
        """
        Find the first zone that contains the detection.
        
        Args:
            x_min: Normalized minimum x coordinate
            y_min: Normalized minimum y coordinate
            x_max: Normalized maximum x coordinate
            y_max: Normalized maximum y coordinate
            
        Returns:
            DetectionZone if found, None otherwise
        """
        for zone in self.zones:
            if not zone.enabled:
                continue
            
            if zone.contains_box(x_min, y_min, x_max, y_max, mode=self.zone_mode):
                return zone
        
        return None
    
    class Config:
        schema_extra = {
            "example": {
                "zones": [
                    {
                        "id": "zone1",
                        "name": "Entrance",
                        "polygon": [
                            {"x": 0.0, "y": 0.0},
                            {"x": 0.5, "y": 0.0},
                            {"x": 0.5, "y": 1.0},
                            {"x": 0.0, "y": 1.0}
                        ],
                        "enabled": True,
                        "filter_classes": ["person"]
                    }
                ],
                "zone_mode": "center"
            }
        }

