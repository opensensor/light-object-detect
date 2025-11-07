from typing import List, Optional
from models.detection import DetectionResult
from models.zone import DetectionZone, ZoneConfiguration


def filter_detections_by_zones(
    detections: List[DetectionResult],
    zone_config: ZoneConfiguration
) -> List[DetectionResult]:
    """
    Filter detections based on zone configuration.
    
    Args:
        detections: List of detection results
        zone_config: Zone configuration
        
    Returns:
        Filtered list of detections with zone information
    """
    if not zone_config.zones:
        # No zones defined, return all detections
        return detections
    
    filtered_detections = []
    
    for detection in detections:
        bbox = detection.bounding_box
        
        # Find which zone (if any) contains this detection
        zone = zone_config.get_zone_for_detection(
            bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
        )
        
        if zone is None:
            # Detection is not in any zone, skip it
            continue
        
        # Check if detection class is allowed in this zone
        if zone.filter_classes and detection.label not in zone.filter_classes:
            continue
        
        # Check if detection meets zone's minimum confidence
        if zone.min_confidence and detection.confidence < zone.min_confidence:
            continue
        
        # Add zone information to detection
        detection.zone_id = zone.id
        filtered_detections.append(detection)
    
    return filtered_detections


def apply_class_filter(
    detections: List[DetectionResult],
    allowed_classes: Optional[List[str]] = None,
    blocked_classes: Optional[List[str]] = None
) -> List[DetectionResult]:
    """
    Filter detections by class labels.
    
    Args:
        detections: List of detection results
        allowed_classes: List of allowed class labels (None = all allowed)
        blocked_classes: List of blocked class labels (None = none blocked)
        
    Returns:
        Filtered list of detections
    """
    filtered = detections
    
    # Apply allowed classes filter
    if allowed_classes:
        filtered = [d for d in filtered if d.label in allowed_classes]
    
    # Apply blocked classes filter
    if blocked_classes:
        filtered = [d for d in filtered if d.label not in blocked_classes]
    
    return filtered


def apply_confidence_filter(
    detections: List[DetectionResult],
    min_confidence: float
) -> List[DetectionResult]:
    """
    Filter detections by minimum confidence.
    
    Args:
        detections: List of detection results
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered list of detections
    """
    return [d for d in detections if d.confidence >= min_confidence]


def apply_size_filter(
    detections: List[DetectionResult],
    min_width: Optional[float] = None,
    min_height: Optional[float] = None,
    max_width: Optional[float] = None,
    max_height: Optional[float] = None
) -> List[DetectionResult]:
    """
    Filter detections by bounding box size.
    
    Args:
        detections: List of detection results
        min_width: Minimum normalized width (0-1)
        min_height: Minimum normalized height (0-1)
        max_width: Maximum normalized width (0-1)
        max_height: Maximum normalized height (0-1)
        
    Returns:
        Filtered list of detections
    """
    filtered = []
    
    for detection in detections:
        bbox = detection.bounding_box
        width = bbox.x_max - bbox.x_min
        height = bbox.y_max - bbox.y_min
        
        # Check minimum width
        if min_width is not None and width < min_width:
            continue
        
        # Check minimum height
        if min_height is not None and height < min_height:
            continue
        
        # Check maximum width
        if max_width is not None and width > max_width:
            continue
        
        # Check maximum height
        if max_height is not None and height > max_height:
            continue
        
        filtered.append(detection)
    
    return filtered

