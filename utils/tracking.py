import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

from models.detection import DetectionResult, BoundingBox


@dataclass
class Track:
    """Represents a tracked object across frames."""
    
    track_id: int
    label: str
    bounding_box: BoundingBox
    confidence: float
    last_seen: float = field(default_factory=time.time)
    age: int = 0  # Number of frames this track has existed
    hits: int = 1  # Number of times this track was matched
    misses: int = 0  # Number of consecutive frames without a match
    
    def update(self, detection: DetectionResult):
        """Update track with new detection."""
        self.bounding_box = detection.bounding_box
        self.confidence = detection.confidence
        self.last_seen = time.time()
        self.age += 1
        self.hits += 1
        self.misses = 0
    
    def mark_missed(self):
        """Mark that this track was not matched in current frame."""
        self.age += 1
        self.misses += 1


class ObjectTracker:
    """
    Simple object tracker using IoU-based matching.
    
    This is a lightweight alternative to DeepSORT that doesn't require
    deep learning features. It uses Intersection over Union (IoU) for
    matching detections to existing tracks.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_time_since_update: float = 1.0
    ):
        """
        Initialize the object tracker.
        
        Args:
            max_age: Maximum number of frames to keep a track without updates
            min_hits: Minimum number of hits before a track is confirmed
            iou_threshold: Minimum IoU for matching detections to tracks
            max_time_since_update: Maximum time (seconds) since last update
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_time_since_update = max_time_since_update
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box
            box2: Second bounding box
            
        Returns:
            IoU score (0-1)
        """
        # Calculate intersection
        x_min = max(box1.x_min, box2.x_min)
        y_min = max(box1.y_min, box2.y_min)
        x_max = min(box1.x_max, box2.x_max)
        y_max = min(box1.y_max, box2.y_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
        area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _match_detections_to_tracks(
        self,
        detections: List[DetectionResult]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.
        
        Args:
            detections: List of detections in current frame
            
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
            - matches: List of (detection_idx, track_idx) pairs
            - unmatched_detections: List of detection indices
            - unmatched_tracks: List of track indices
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for d_idx, detection in enumerate(detections):
            for t_idx, track in enumerate(self.tracks):
                # Only match same class
                if detection.label == track.label:
                    iou_matrix[d_idx, t_idx] = self._calculate_iou(
                        detection.bounding_box,
                        track.bounding_box
                    )
        
        # Greedy matching: find best matches above threshold
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # Sort by IoU (highest first)
        while True:
            if len(unmatched_detections) == 0 or len(unmatched_tracks) == 0:
                break
            
            # Find maximum IoU
            max_iou = 0
            max_d_idx = -1
            max_t_idx = -1
            
            for d_idx in unmatched_detections:
                for t_idx in unmatched_tracks:
                    if iou_matrix[d_idx, t_idx] > max_iou:
                        max_iou = iou_matrix[d_idx, t_idx]
                        max_d_idx = d_idx
                        max_t_idx = t_idx
            
            # Check if best match is above threshold
            if max_iou < self.iou_threshold:
                break
            
            # Add match
            matches.append((max_d_idx, max_t_idx))
            unmatched_detections.remove(max_d_idx)
            unmatched_tracks.remove(max_t_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections in current frame
            
        Returns:
            List of detections with track IDs assigned
        """
        self.frame_count += 1
        
        # Match detections to tracks
        matches, unmatched_detections, unmatched_tracks = \
            self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])
            detections[det_idx].track_id = self.tracks[track_idx].track_id
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = Track(
                track_id=self.next_id,
                label=detections[det_idx].label,
                bounding_box=detections[det_idx].bounding_box,
                confidence=detections[det_idx].confidence
            )
            self.tracks.append(new_track)
            detections[det_idx].track_id = self.next_id
            self.next_id += 1
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Remove old tracks
        current_time = time.time()
        self.tracks = [
            track for track in self.tracks
            if track.misses < self.max_age and
               (current_time - track.last_seen) < self.max_time_since_update
        ]
        
        # Only return detections from confirmed tracks
        confirmed_detections = [
            det for det in detections
            if det.track_id is not None and
               any(t.track_id == det.track_id and t.hits >= self.min_hits 
                   for t in self.tracks)
        ]
        
        return confirmed_detections
    
    def get_active_tracks(self) -> List[Track]:
        """
        Get list of currently active tracks.
        
        Returns:
            List of active Track objects
        """
        return [track for track in self.tracks if track.hits >= self.min_hits]
    
    def reset(self):
        """Reset the tracker, clearing all tracks."""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0


class MultiStreamTracker:
    """Manages separate trackers for multiple streams."""
    
    def __init__(self, **tracker_kwargs):
        """
        Initialize multi-stream tracker.
        
        Args:
            **tracker_kwargs: Arguments to pass to ObjectTracker constructor
        """
        self.trackers: Dict[str, ObjectTracker] = {}
        self.tracker_kwargs = tracker_kwargs
    
    def get_tracker(self, stream_name: str) -> ObjectTracker:
        """
        Get or create tracker for a stream.
        
        Args:
            stream_name: Name of the stream
            
        Returns:
            ObjectTracker for the stream
        """
        if stream_name not in self.trackers:
            self.trackers[stream_name] = ObjectTracker(**self.tracker_kwargs)
        return self.trackers[stream_name]
    
    def update(self, stream_name: str, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Update tracker for a specific stream.
        
        Args:
            stream_name: Name of the stream
            detections: List of detections
            
        Returns:
            List of detections with track IDs
        """
        tracker = self.get_tracker(stream_name)
        return tracker.update(detections)
    
    def reset_stream(self, stream_name: str):
        """Reset tracker for a specific stream."""
        if stream_name in self.trackers:
            self.trackers[stream_name].reset()
    
    def reset_all(self):
        """Reset all stream trackers."""
        for tracker in self.trackers.values():
            tracker.reset()

