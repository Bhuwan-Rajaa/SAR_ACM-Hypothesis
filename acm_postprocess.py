"""
acm_postprocess.py
==================
Active Contour Model (Morphological Chan-Vese) post-processing filter
for YOLOv8n ship detections on SAR imagery.

Filters false positive detections by:
1. Cropping each detection's bounding box region from the SAR image
2. Applying Morphological Chan-Vese segmentation
3. Extracting shape features from the segmented mask
4. Discarding detections where ACM fails to find a valid ship contour
5. Adjusting confidence scores based on contour quality

Usage:
    from acm_postprocess import ACMFilter
    acm = ACMFilter()
    filtered = acm.filter_detections(image, boxes, confidences)
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from skimage.segmentation import morphological_chan_vese
from skimage.measure import regionprops, label as sk_label
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ShapeFeatures:
    """Shape descriptors extracted from ACM segmentation mask."""
    area_ratio: float = 0.0       # segmented area / bbox area
    compactness: float = 0.0      # 4*pi*area / perimeter^2
    solidity: float = 0.0         # area / convex_hull_area
    elongation: float = 0.0       # major_axis / minor_axis
    segment_area_px: int = 0      # absolute segmented area in pixels
    contrast_ratio: float = 0.0   # mean_foreground / mean_background intensity
    convergence_score: float = 0.0 # how much the contour evolved from init
    is_valid: bool = False         # whether features could be computed


@dataclass
class FilteredDetection:
    """A single detection after ACM filtering."""
    bbox: np.ndarray               # [x1, y1, x2, y2] in image coordinates
    original_conf: float           # YOLOv8 confidence
    adjusted_conf: float           # confidence after ACM adjustment
    kept: bool                     # whether this detection passed the filter
    discard_reason: str = ""       # why it was discarded (if applicable)
    shape_features: ShapeFeatures = field(default_factory=ShapeFeatures)
    acm_mask: Optional[np.ndarray] = None  # the segmentation mask (for viz)
    quality_score: float = 1.0     # ACM-derived quality score [0, 1]


# ---------------------------------------------------------------------------
# ACM Filter
# ---------------------------------------------------------------------------

class ACMFilter:
    """
    Morphological Chan-Vese Active Contour Model filter for YOLOv8 detections.

    For each detection bounding box:
    1. Crops the region (with padding) from the SAR image
    2. Applies light despeckling (median filter)
    3. Runs Morphological Chan-Vese segmentation
    4. Extracts shape features from the resulting mask
    5. Decides to keep or discard based on shape validity
    6. Adjusts confidence score using shape quality
    """

    def __init__(
        self,
        # Chan-Vese parameters
        num_iter: int = 100,
        smoothing: int = 1,
        lambda1: float = 1.0,
        lambda2: float = 1.5,
        init_level_set: str = "checkerboard",
        # Crop parameters
        bbox_padding: float = 0.2,
        # Despeckling
        median_filter_size: int = 3,
        # Filtering thresholds based on feature analysis
        min_area_ratio: float = 0.05,
        max_area_ratio: float = 0.85,    # TPs rarely > 0.75
        max_compactness: float = 0.80,   # TPs rarely > 0.77 (FPs are more circular)
        min_elongation: float = 1.5,     # TPs rarely < 1.64
        min_contrast_ratio: float = 3.0, # TPs rarely < 4.0
        min_segment_area_px: int = 4,
        # Confidence gating
        conf_gate: float = 0.5,  # only apply ACM to detections below this confidence
        # Confidence adjustment
        quality_weight: float = 0.3,  # how much ACM quality influences final conf
    ):
        # Chan-Vese
        self.num_iter = num_iter
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.init_level_set = init_level_set

        # Crop
        self.bbox_padding = bbox_padding

        # Despeckling
        self.median_filter_size = median_filter_size

        # Thresholds
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_compactness = max_compactness
        self.min_elongation = min_elongation
        self.min_segment_area_px = min_segment_area_px
        self.min_contrast_ratio = min_contrast_ratio

        # Quality calculation parameters
        self.quality_threshold = 0.7  # Quality below this gets penalized
        self.penalty_factor = 0.8     # How strongly to penalize (0.8 from simulation)


    def filter_detections(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray,
        return_masks: bool = False,
    ) -> List[FilteredDetection]:
        """
        Apply ACM filtering to all detections in an image.

        Args:
            image: Full SAR image (H, W) grayscale or (H, W, 3) BGR
            boxes: Detection bounding boxes, shape (N, 4) as [x1, y1, x2, y2]
            confidences: Detection confidence scores, shape (N,)
            return_masks: If True, store ACM masks in results (for visualization)

        Returns:
            List of FilteredDetection objects
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Normalize to [0, 1] float for Chan-Vese
        gray = gray.astype(np.float64)
        if gray.max() > 0:
            gray = gray / gray.max()

        img_h, img_w = gray.shape
        results = []

        for i in range(len(boxes)):
            bbox = boxes[i]
            conf = float(confidences[i])

            # Apply ACM to this detection
            filtered = self._process_single_detection(
                gray, bbox, conf, img_h, img_w, return_masks
            )
            results.append(filtered)

        return results

    def _process_single_detection(
        self,
        gray: np.ndarray,
        bbox: np.ndarray,
        conf: float,
        img_h: int,
        img_w: int,
        return_mask: bool,
    ) -> FilteredDetection:
        """Process a single detection through ACM filtering."""

        x1, y1, x2, y2 = bbox.astype(int)
        bw = x2 - x1
        bh = y2 - y1

        # Skip extremely small detections
        if bw < 3 or bh < 3:
            return FilteredDetection(
                bbox=bbox, original_conf=conf, adjusted_conf=0.0,
                kept=False, discard_reason="bbox_too_small",
            )

        # Crop with padding
        pad_x = int(bw * self.bbox_padding)
        pad_y = int(bh * self.bbox_padding)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(img_w, x2 + pad_x)
        cy2 = min(img_h, y2 + pad_y)

        crop = gray[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return FilteredDetection(
                bbox=bbox, original_conf=conf, adjusted_conf=0.0,
                kept=False, discard_reason="empty_crop",
            )

        # Light despeckling
        if self.median_filter_size > 0:
            crop = median_filter(crop, size=self.median_filter_size)

        # Run Morphological Chan-Vese
        try:
            acm_mask = self._segment_crop(crop)
        except Exception:
            return FilteredDetection(
                bbox=bbox, original_conf=conf, adjusted_conf=0.0,
                kept=False, discard_reason="acm_failed",
            )

        # Determine which label (0 or 1) is the foreground (ship)
        # Ships are typically bright in SAR -> foreground is the brighter region
        mean_inside = crop[acm_mask == 1].mean() if acm_mask.sum() > 0 else 0
        mean_outside = crop[acm_mask == 0].mean() if (acm_mask == 0).sum() > 0 else 0
        if mean_inside < mean_outside:
            # Invert: the bright region was labeled 0
            acm_mask = 1 - acm_mask
            mean_inside, mean_outside = mean_outside, mean_inside

        # Compute intensity contrast ratio (key SAR discriminator)
        contrast_ratio = mean_inside / mean_outside if mean_outside > 1e-6 else 0.0

        # Extract shape features
        features = self._extract_shape_features(acm_mask, bw * bh)
        features.contrast_ratio = contrast_ratio

        # Compute quality score [0, 1]
        quality = self._compute_quality_score(features) if features.is_valid else 0.0
        
        # Determine if we keep it (only hard-discard on catastrophic failures)
        kept, reason = self._decide(features, conf)

        # Soft Confidence Scaling
        if kept:
            if quality >= self.quality_threshold:
                # High quality: keep confidence as is (or slight boost if implemented)
                adjusted_conf = conf
            else:
                # Low quality: apply penalty linearly based on how far below threshold
                penalty = 1.0 - self.penalty_factor * (self.quality_threshold - quality)
                # Ensure penalty doesn't go below a minimum floor (e.g. 0.1)
                adjusted_conf = conf * max(0.1, penalty)
        else:
            adjusted_conf = 0.0

        return FilteredDetection(
            bbox=bbox,
            original_conf=conf,
            adjusted_conf=adjusted_conf,
            kept=kept,
            discard_reason=reason,
            shape_features=features,
            acm_mask=acm_mask if return_mask else None,
            quality_score=quality,
        )

    def _segment_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Run Morphological Chan-Vese on a cropped region.

        Returns binary mask (0 = background, 1 = foreground).
        """
        mask = morphological_chan_vese(
            crop,
            num_iter=self.num_iter,
            smoothing=self.smoothing,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            init_level_set=self.init_level_set,
        )
        return mask.astype(np.uint8)

    def _extract_shape_features(
        self, mask: np.ndarray, bbox_area: int
    ) -> ShapeFeatures:
        """Extract shape descriptors from binary segmentation mask."""

        features = ShapeFeatures()
        segment_area = int(mask.sum())
        features.segment_area_px = segment_area

        if segment_area < self.min_segment_area_px or bbox_area <= 0:
            features.is_valid = False
            return features

        features.area_ratio = segment_area / bbox_area

        # Use regionprops on the largest connected component
        labeled = sk_label(mask)
        regions = regionprops(labeled)
        if not regions:
            features.is_valid = False
            return features

        # Take the largest region
        largest = max(regions, key=lambda r: r.area)

        # Compactness (circularity): 4*pi*area / perimeter^2
        if largest.perimeter > 0:
            features.compactness = (4 * np.pi * largest.area) / (largest.perimeter ** 2)
        else:
            features.compactness = 0.0

        # Solidity: area / convex_hull_area
        features.solidity = largest.solidity if hasattr(largest, 'solidity') else 0.0

        # Elongation: major_axis / minor_axis
        minor_len = getattr(largest, 'axis_minor_length', None) or getattr(largest, 'minor_axis_length', 0)
        major_len = getattr(largest, 'axis_major_length', None) or getattr(largest, 'major_axis_length', 0)
        if minor_len > 0:
            features.elongation = major_len / minor_len
        else:
            features.elongation = float('inf')

        features.is_valid = True
        return features

    def _decide(
        self, features: ShapeFeatures, conf: float
    ) -> Tuple[bool, str]:
        """
        Decide whether to keep or discard a detection based on shape features
        and intensity contrast.

        Uses a multi-criteria approach:
        - Contrast ratio is the primary discriminator (ships are bright in SAR)
        - Shape features provide secondary validation
        - A detection must fail BOTH contrast AND shape to be discarded
          (conservative: only discard when we're very confident it's a FP)

        Returns (kept, reason).
        """
        if not features.is_valid:
            return False, "acm_no_valid_segment"

        # Soft Scaling Policy: 
        # We no longer hard-discard based on features because distributions overlap.
        # Hard discards kill Recall. Instead, we heavily penalize confidence later.
        
        # We only hard-discard if the segmentation completely failed.
        if features.area_ratio < 0.01:
            return False, "near_empty_segment"

        return True, ""

    def _compute_quality_score(self, features: ShapeFeatures) -> float:
        """
        Compute a quality score [0, 1] from shape features.
        Higher = more ship-like.

        Scoring rationale:
        - area_ratio in [0.1, 0.6] is ideal for ships in bbox
        - solidity > 0.7 is good (compact, not fragmented)
        - moderate elongation (1.5-5) is ship-like
        - compactness: moderate values typical for elongated ships
        """
        # Use data-driven normalized scores based on TP/FP distributions
        # We linearly map features between the 5th and 95th percentiles of typical noise.
        
        # Contrast: TP mostly > 4, FP mostly < 8.
        s_contrast = np.clip((features.contrast_ratio - 3) / (9 - 3), 0, 1)
        
        # Elongation: TP mostly > 1.6, FP mostly < 5.
        s_elong = np.clip((features.elongation - 1.5) / (5 - 1.5), 0, 1)
        
        # Area ratio: TP mostly < 0.7, FP mostly > 0.2. (Inverted: lower is better)
        s_area = np.clip((0.8 - features.area_ratio) / (0.8 - 0.2), 0, 1)
        
        # Compactness: TP mostly < 0.7, FP mostly > 0.3. (Inverted: lower is better)
        s_comp = np.clip((0.8 - features.compactness) / (0.8 - 0.3), 0, 1)
        
        # Weighted average based on discriminator strength
        quality = (s_contrast * 0.40) + (s_elong * 0.30) + (s_area * 0.15) + (s_comp * 0.15)
        
        return float(quality)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def apply_acm_filter(
    image: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray = None,
    acm_filter: ACMFilter = None,
    return_masks: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[FilteredDetection]]:
    """
    Convenience function: apply ACM filtering and return filtered arrays.

    Args:
        image: SAR image
        boxes: (N, 4) bboxes [x1, y1, x2, y2]
        confidences: (N,) confidence scores
        class_ids: (N,) class IDs (passed through)
        acm_filter: ACMFilter instance (creates default if None)
        return_masks: store masks in results

    Returns:
        filtered_boxes: (M, 4) kept bboxes
        filtered_confs: (M,) adjusted confidences
        filtered_classes: (M,) class IDs (if provided)
        all_results: List of all FilteredDetection objects (kept and discarded)
    """
    if acm_filter is None:
        acm_filter = ACMFilter()

    results = acm_filter.filter_detections(image, boxes, confidences, return_masks)

    kept_indices = [i for i, r in enumerate(results) if r.kept]

    if len(kept_indices) == 0:
        filtered_boxes = np.zeros((0, 4))
        filtered_confs = np.zeros(0)
        filtered_classes = np.zeros(0, dtype=int)
    else:
        filtered_boxes = boxes[kept_indices]
        filtered_confs = np.array([results[i].adjusted_conf for i in kept_indices])
        if class_ids is not None:
            filtered_classes = class_ids[kept_indices]
        else:
            filtered_classes = np.zeros(len(kept_indices), dtype=int)

    return filtered_boxes, filtered_confs, filtered_classes, results
