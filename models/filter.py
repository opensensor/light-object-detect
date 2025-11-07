from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class FilterOperator(str, Enum):
    """Operators for filtering rules."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"


class FilterRule(BaseModel):
    """A single filter rule for detections."""
    
    field: str = Field(..., description="Field to filter on (label, confidence, width, height)")
    operator: FilterOperator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    
    class Config:
        schema_extra = {
            "example": {
                "field": "label",
                "operator": "in",
                "value": ["person", "car"]
            }
        }


class FilterGroup(BaseModel):
    """Group of filter rules with AND/OR logic."""
    
    logic: str = Field("AND", description="Logic operator: 'AND' or 'OR'")
    rules: List[FilterRule] = Field(..., description="List of filter rules")
    
    @validator('logic')
    def validate_logic(cls, v):
        """Validate logic operator."""
        if v.upper() not in ["AND", "OR"]:
            raise ValueError("logic must be 'AND' or 'OR'")
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
                "logic": "AND",
                "rules": [
                    {
                        "field": "label",
                        "operator": "in",
                        "value": ["person", "car"]
                    },
                    {
                        "field": "confidence",
                        "operator": "greater_equal",
                        "value": 0.7
                    }
                ]
            }
        }


class EventFilter(BaseModel):
    """Complete event filter configuration."""
    
    enabled: bool = Field(True, description="Whether filtering is enabled")
    filter_groups: List[FilterGroup] = Field(
        default_factory=list,
        description="List of filter groups (combined with OR logic)"
    )
    
    # Predefined filter presets
    person_only: bool = Field(False, description="Only detect persons")
    vehicle_only: bool = Field(False, description="Only detect vehicles")
    animal_only: bool = Field(False, description="Only detect animals")
    
    # Common vehicle classes
    VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
    
    # Common animal classes
    ANIMAL_CLASSES = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
                     "bear", "zebra", "giraffe"]
    
    def get_allowed_classes(self) -> Optional[List[str]]:
        """
        Get the list of allowed classes based on presets.
        
        Returns:
            List of allowed class names, or None if no preset is active
        """
        if self.person_only:
            return ["person"]
        elif self.vehicle_only:
            return self.VEHICLE_CLASSES
        elif self.animal_only:
            return self.ANIMAL_CLASSES
        
        return None
    
    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "person_only": True,
                "filter_groups": []
            }
        }


class DetectionEvent(BaseModel):
    """Detection event with metadata."""
    
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: float = Field(..., description="Event timestamp (Unix time)")
    stream_name: Optional[str] = Field(None, description="Name of the stream")
    detections_count: int = Field(..., description="Number of detections in this event")
    labels: List[str] = Field(..., description="List of unique labels detected")
    max_confidence: float = Field(..., description="Maximum confidence score")
    zones_triggered: List[str] = Field(
        default_factory=list,
        description="List of zone IDs that were triggered"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "event_id": "evt_123456",
                "timestamp": 1699200000.0,
                "stream_name": "front_door",
                "detections_count": 2,
                "labels": ["person", "car"],
                "max_confidence": 0.95,
                "zones_triggered": ["entrance", "parking"]
            }
        }

