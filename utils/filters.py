from typing import List, Any
from models.detection import DetectionResult
from models.filter import FilterRule, FilterGroup, FilterOperator, EventFilter


def apply_filter_rule(detection: DetectionResult, rule: FilterRule) -> bool:
    """
    Apply a single filter rule to a detection.
    
    Args:
        detection: Detection result to filter
        rule: Filter rule to apply
        
    Returns:
        True if detection passes the filter
    """
    # Get the field value from detection
    if rule.field == "label":
        field_value = detection.label
    elif rule.field == "confidence":
        field_value = detection.confidence
    elif rule.field == "width":
        field_value = detection.bounding_box.x_max - detection.bounding_box.x_min
    elif rule.field == "height":
        field_value = detection.bounding_box.y_max - detection.bounding_box.y_min
    elif rule.field == "zone_id":
        field_value = detection.zone_id
    elif rule.field == "track_id":
        field_value = detection.track_id
    else:
        # Unknown field, skip this rule
        return True
    
    # Apply operator
    if rule.operator == FilterOperator.EQUALS:
        return field_value == rule.value
    elif rule.operator == FilterOperator.NOT_EQUALS:
        return field_value != rule.value
    elif rule.operator == FilterOperator.IN:
        return field_value in rule.value
    elif rule.operator == FilterOperator.NOT_IN:
        return field_value not in rule.value
    elif rule.operator == FilterOperator.GREATER_THAN:
        return field_value > rule.value
    elif rule.operator == FilterOperator.LESS_THAN:
        return field_value < rule.value
    elif rule.operator == FilterOperator.GREATER_EQUAL:
        return field_value >= rule.value
    elif rule.operator == FilterOperator.LESS_EQUAL:
        return field_value <= rule.value
    else:
        return True


def apply_filter_group(detection: DetectionResult, group: FilterGroup) -> bool:
    """
    Apply a filter group to a detection.
    
    Args:
        detection: Detection result to filter
        group: Filter group to apply
        
    Returns:
        True if detection passes the filter group
    """
    if not group.rules:
        return True
    
    results = [apply_filter_rule(detection, rule) for rule in group.rules]
    
    if group.logic == "AND":
        return all(results)
    else:  # OR
        return any(results)


def apply_event_filter(detections: List[DetectionResult], event_filter: EventFilter) -> List[DetectionResult]:
    """
    Apply event filter to a list of detections.
    
    Args:
        detections: List of detection results
        event_filter: Event filter configuration
        
    Returns:
        Filtered list of detections
    """
    if not event_filter.enabled:
        return detections
    
    # Apply preset filters first
    allowed_classes = event_filter.get_allowed_classes()
    if allowed_classes:
        detections = [d for d in detections if d.label in allowed_classes]
    
    # Apply custom filter groups
    if event_filter.filter_groups:
        filtered = []
        for detection in detections:
            # Detection passes if it matches ANY filter group (OR logic between groups)
            if any(apply_filter_group(detection, group) for group in event_filter.filter_groups):
                filtered.append(detection)
        detections = filtered
    
    return detections


def create_person_filter() -> EventFilter:
    """Create a filter for person detection only."""
    return EventFilter(
        enabled=True,
        person_only=True
    )


def create_vehicle_filter() -> EventFilter:
    """Create a filter for vehicle detection only."""
    return EventFilter(
        enabled=True,
        vehicle_only=True
    )


def create_high_confidence_filter(min_confidence: float = 0.8) -> EventFilter:
    """
    Create a filter for high-confidence detections.
    
    Args:
        min_confidence: Minimum confidence threshold
        
    Returns:
        EventFilter configured for high confidence
    """
    return EventFilter(
        enabled=True,
        filter_groups=[
            FilterGroup(
                logic="AND",
                rules=[
                    FilterRule(
                        field="confidence",
                        operator=FilterOperator.GREATER_EQUAL,
                        value=min_confidence
                    )
                ]
            )
        ]
    )


def create_size_filter(min_width: float = 0.05, min_height: float = 0.05) -> EventFilter:
    """
    Create a filter for minimum object size.
    
    Args:
        min_width: Minimum normalized width
        min_height: Minimum normalized height
        
    Returns:
        EventFilter configured for size filtering
    """
    return EventFilter(
        enabled=True,
        filter_groups=[
            FilterGroup(
                logic="AND",
                rules=[
                    FilterRule(
                        field="width",
                        operator=FilterOperator.GREATER_EQUAL,
                        value=min_width
                    ),
                    FilterRule(
                        field="height",
                        operator=FilterOperator.GREATER_EQUAL,
                        value=min_height
                    )
                ]
            )
        ]
    )


def create_combined_filter(
    allowed_classes: List[str],
    min_confidence: float = 0.5,
    min_width: float = 0.0,
    min_height: float = 0.0
) -> EventFilter:
    """
    Create a combined filter with multiple criteria.
    
    Args:
        allowed_classes: List of allowed class labels
        min_confidence: Minimum confidence threshold
        min_width: Minimum normalized width
        min_height: Minimum normalized height
        
    Returns:
        EventFilter with combined criteria
    """
    rules = [
        FilterRule(
            field="label",
            operator=FilterOperator.IN,
            value=allowed_classes
        ),
        FilterRule(
            field="confidence",
            operator=FilterOperator.GREATER_EQUAL,
            value=min_confidence
        )
    ]
    
    if min_width > 0:
        rules.append(
            FilterRule(
                field="width",
                operator=FilterOperator.GREATER_EQUAL,
                value=min_width
            )
        )
    
    if min_height > 0:
        rules.append(
            FilterRule(
                field="height",
                operator=FilterOperator.GREATER_EQUAL,
                value=min_height
            )
        )
    
    return EventFilter(
        enabled=True,
        filter_groups=[
            FilterGroup(
                logic="AND",
                rules=rules
            )
        ]
    )

