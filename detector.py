from dataclasses import dataclass
from typing import Optional, List, Tuple
import math

import cv2
import numpy as np

from config import (
    HOLE_MIN_RADIUS,
    HOLE_MAX_RADIUS,
    HOLE_MIN_DARK_CONTRAST,
    HOLE_MIN_PAIR_DX,
    HOLE_MAX_PAIR_DX,
    HOLE_MIN_PAIR_DY,
    HOLE_MAX_PAIR_DY,
    COMPONENT_ROI_WIDTH_FACTOR,
    COMPONENT_ROI_HEIGHT_FACTOR,
    COMPONENT_ROI_Y_OFFSET_FACTOR,
    HOLE_PRESENCE_RATIO,
    HOLE_PATCH_RADIUS_FACTOR,
    ANGLE_TOLERANCE_DEG,
)

from lever_detector import (
    measure_lever,
    chirality_similarity,
    silhouette_handedness,
    rotation_invariant_handedness,
    gold_mask,
)


@dataclass
class ComponentSlot:
    component_id: int
    row: int
    column: int
    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    polygon: np.ndarray
    holes: List[Tuple[float, float, float]]
    hole_spacing: float

@dataclass
class ReferenceComponent:
    component_id: int
    row: int
    column: int
    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    polygon: np.ndarray
    holes: List[Tuple[float, float, float]]
    hole_spacing: float

                                 
    lever_contour: np.ndarray
    lever_bbox: Tuple[int, int, int, int]
    pivot: Tuple[int, int]
    tip: Tuple[int, int]
    angle_deg: float
    reliability: float

                                       
    reference_signature: Optional[np.ndarray]
    reference_area: float
    lever_radius: float

    body_template: Optional[np.ndarray]
    body_mask: Optional[np.ndarray]

                             
    reference_hole_contrasts: Optional[List[float]] = None

@dataclass
class InspectionResult:
    success: bool
    registration: object
    components_detected: int
    levers_detected: int
    results: List[dict]
    error: Optional[str] = None


def angular_difference(a, b):
    """
    Smallest circular difference between two angles.

    Example:
        angular_difference(359, 1) = 2
    """
    if a is None or b is None:
        return None

    d = abs(float(a) - float(b)) % 360.0

    return min(d, 360.0 - d)

def normalize_angle(angle):
    if angle is None:
        return None

    return float(angle) % 360.0


def _ensemble_handedness(
    contour: np.ndarray,
    pivot: Tuple[int, int],
    references: List["ReferenceComponent"],
) -> Tuple[float, float, float]:
    """Compare a current lever against multiple known-good references.

    Every calibrated reference lever is known-correct.  The current silhouette
    is scored against each reference in its normal form and its explicitly
    mirrored form, with arbitrary in-plane rotation allowed.  A robust top-two
    aggregate reduces sensitivity to one noisy reference contour.

    This function is used only as a secondary orientation signal. It never
    changes component registration or lever detection.
    """
    if contour is None or not references:
        return 0.0, 0.0, 0.0

    normal_scores = []
    mirror_scores = []

    for ref in references:
        try:
            reference_local = ref.lever_contour.copy()
            reference_local[:, :, 0] -= int(ref.bbox[0])
            reference_local[:, :, 1] -= int(ref.bbox[1])

            reference_pivot = (
                int(ref.pivot[0] - ref.bbox[0]),
                int(ref.pivot[1] - ref.bbox[1]),
            )

            rn, rm, _ = silhouette_handedness(
                contour,
                pivot,
                reference_local,
                reference_pivot,
            )

            normal_scores.append(float(rn))
            mirror_scores.append(float(rm))
        except Exception:
            continue

    if not normal_scores:
        return 0.0, 0.0, 0.0

    normal_scores.sort(reverse=True)
    mirror_scores.sort(reverse=True)

    if len(normal_scores) >= 2:
        normal = 0.72 * normal_scores[0] + 0.28 * normal_scores[1]
    else:
        normal = normal_scores[0]

    if len(mirror_scores) >= 2:
        mirror = 0.72 * mirror_scores[0] + 0.28 * mirror_scores[1]
    else:
        mirror = mirror_scores[0]

    return (
        float(normal),
        float(mirror),
        float(normal - mirror),
    )

                                                              
                        
                                                              

def _circle_contrast(gray, x, y, r):
    if gray is None or gray.size == 0:
        return 0.0

    h, w = gray.shape

    x = int(round(x))
    y = int(round(y))
    r = max(4, int(round(r)))

    yy, xx = np.ogrid[:h, :w]

    d2 = (xx - x) ** 2 + (yy - y) ** 2

    inner = d2 <= r * r

    outer = (
        (d2 <= int(r * 2.0) ** 2)
        &
        (d2 >= int(r * 1.35) ** 2)
    )

    if not np.any(inner) or not np.any(outer):
        return 0.0

    return float(
        np.mean(gray[outer])
        -
        np.mean(gray[inner])
    )

def _hole_candidates(image):
    """
    Find circular mounting holes.
    """

    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        1.0
    )

    try:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=100,
            param2=25,
            minRadius=HOLE_MIN_RADIUS,
            maxRadius=HOLE_MAX_RADIUS,
        )

    except cv2.error:
        return []

    if circles is None:
        return []

    candidates = []

    h, w = gray.shape

    for x, y, r in np.round(
        circles[0]
    ).astype(np.float32):

        if not (
            0 <= x < w
            and
            0 <= y < h
        ):
            continue

        contrast = _circle_contrast(
            gray,
            x,
            y,
            r
        )

        if contrast < HOLE_MIN_DARK_CONTRAST:
            continue

        candidates.append(
            {
                "x": float(x),
                "y": float(y),
                "r": float(r),
                "contrast": float(contrast),
            }
        )

    return candidates

def _pair_holes(candidates):
    """
    Find the 10 physical component hole-pairs.
    """

    pair_candidates = []

    for i, a in enumerate(candidates):

        for j, b in enumerate(candidates):

            if i == j:
                continue

            dx = b["x"] - a["x"]
            dy = b["y"] - a["y"]

            if dx <= 0:
                continue

            if not (
                HOLE_MIN_PAIR_DX
                <= dx
                <= HOLE_MAX_PAIR_DX
            ):
                continue

            if not (
                HOLE_MIN_PAIR_DY
                <= dy
                <= HOLE_MAX_PAIR_DY
            ):
                continue

            radius_delta = abs(
                a["r"] - b["r"]
            )

            contrast_score = min(
                a["contrast"],
                b["contrast"]
            )

            pair_candidates.append(
                {
                    "i": i,
                    "j": j,
                    "dx": dx,
                    "dy": dy,
                    "distance": math.hypot(
                        dx,
                        dy
                    ),
                    "radius_delta": radius_delta,
                    "contrast": contrast_score,
                }
            )

    if len(pair_candidates) < 10:
        return []

    dx_values = np.array(
        [
            p["dx"]
            for p in pair_candidates
        ],
        dtype=np.float32
    )

    dy_values = np.array(
        [
            p["dy"]
            for p in pair_candidates
        ],
        dtype=np.float32
    )

    target_dx = float(
        np.median(dx_values)
    )

    target_dy = float(
        np.median(dy_values)
    )

    for p in pair_candidates:

        geometry_error = math.hypot(
            (
                p["dx"]
                -
                target_dx
            )
            /
            max(target_dx, 1.0),

            (
                p["dy"]
                -
                target_dy
            )
            /
            max(target_dy, 1.0)
        )

        p["score"] = (
            3.0
            *
            p["contrast"]
            /
            60.0

            +

            1.5
            *
            math.exp(
                -4.0
                *
                geometry_error
            )

            -

            0.03
            *
            p["radius_delta"]
        )

    pair_candidates.sort(
        key=lambda p: p["score"],
        reverse=True
    )

    selected = []

    used = set()

    for p in pair_candidates:

        if (
            p["i"] in used
            or
            p["j"] in used
        ):
            continue

        selected.append(p)

        used.add(p["i"])
        used.add(p["j"])

        if len(selected) == 10:
            break

    if len(selected) != 10:
        return []

    return [
        (
            candidates[p["i"]],
            candidates[p["j"]]
        )
        for p in selected
    ]

                                                              
                          
                                                              

def detect_component_slots(reference):
    """
    Detect the 10 physical component slots from the reference image.

    Layout:
        5 top
        5 bottom
    """

    candidates = _hole_candidates(
        reference
    )

    pairs = _pair_holes(
        candidates
    )

    if len(pairs) != 10:

        raise RuntimeError(
            "REFERENCE REGISTRATION FAILED: "
            "could not resolve exactly 10 "
            f"component pairs ({len(pairs)}/10)."
        )

    raw = []

    for a, b in pairs:

        cx = (
            a["x"]
            +
            b["x"]
        ) / 2.0

        cy = (
            a["y"]
            +
            b["y"]
        ) / 2.0

        spacing = math.hypot(
            b["x"] - a["x"],
            b["y"] - a["y"]
        )

        raw.append(
            (
                cx,
                cy,
                spacing,
                a,
                b
            )
        )

    ordered = sorted(
        raw,
        key=lambda v: (
            v[1],
            v[0]
        )
    )

    top = sorted(
        ordered[:5],
        key=lambda v: v[0]
    )

    bottom = sorted(
        ordered[5:],
        key=lambda v: v[0]
    )

    if (
        len(top) != 5
        or
        len(bottom) != 5
    ):
        raise RuntimeError(
            "REFERENCE REGISTRATION FAILED: "
            "expected a 5 + 5 component layout."
        )

    slots = []

    for row_index, row in enumerate(
        [top, bottom]
    ):

        for col_index, (
            cx,
            cy,
            spacing,
            a,
            b
        ) in enumerate(row):

            width = int(
                round(
                    COMPONENT_ROI_WIDTH_FACTOR
                    *
                    spacing
                )
            )

            height = int(
                round(
                    COMPONENT_ROI_HEIGHT_FACTOR
                    *
                    spacing
                )
            )

            roi_cx = cx

            roi_cy = (
                cy
                +
                COMPONENT_ROI_Y_OFFSET_FACTOR
                *
                spacing
            )

            x = int(
                round(
                    roi_cx
                    -
                    width / 2.0
                )
            )

            y = int(
                round(
                    roi_cy
                    -
                    height / 2.0
                )
            )

            x = max(
                0,
                min(
                    x,
                    reference.shape[1] - 1
                )
            )

            y = max(
                0,
                min(
                    y,
                    reference.shape[0] - 1
                )
            )

            width = min(
                width,
                reference.shape[1] - x
            )

            height = min(
                height,
                reference.shape[0] - y
            )

            polygon = np.array(
                [
                    [x, y],
                    [
                        x + width - 1,
                        y
                    ],
                    [
                        x + width - 1,
                        y + height - 1
                    ],
                    [
                        x,
                        y + height - 1
                    ],
                ],
                dtype=np.int32
            )

            slots.append(
                ComponentSlot(
                    component_id=(
                        row_index * 5
                        +
                        col_index
                        +
                        1
                    ),

                    row=row_index,

                    column=col_index,

                    center=(
                        cx,
                        cy
                    ),

                    bbox=(
                        x,
                        y,
                        width,
                        height
                    ),

                    polygon=polygon,

                    holes=[
                        (
                            a["x"],
                            a["y"],
                            a["r"]
                        ),
                        (
                            b["x"],
                            b["y"],
                            b["r"]
                        )
                    ],

                    hole_spacing=spacing,
                )
            )

    spacing_values = np.array(
        [
            s.hole_spacing
            for s in slots
        ],
        dtype=np.float32
    )

    if (
        np.min(spacing_values) <= 0
        or
        np.max(spacing_values)
        /
        np.min(spacing_values)
        >
        1.35
    ):
        raise RuntimeError(
            "REFERENCE REGISTRATION FAILED: "
            "component geometry is inconsistent."
        )

    return slots

                                                              
               
                                                              

def _extract_roi(image, bbox):
    if image is None or image.size == 0:
        return None

    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return None

    x0 = max(0, x)
    y0 = max(0, y)

    x1 = min(
        image.shape[1],
        x + w
    )

    y1 = min(
        image.shape[0],
        y + h
    )

    if (
        x1 <= x0
        or
        y1 <= y0
    ):
        return None

    return image[
        y0:y1,
        x0:x1
    ]

def _relative_point(point, bbox):
    if point is None:
        return None

    return (
        point[0] - bbox[0],
        point[1] - bbox[1]
    )

                                                              
                       
                                                              

def _reference_signature(
    contour,
    pivot
):
    """
    Optional reference signature.

    It is retained for compatibility, but IMPORTANT:
    this signature is NOT supplied to the current inspection
    measurement.
    """

    from lever_detector import (
        _polar_signature
    )

    return _polar_signature(
        contour,
        pivot
    )

def _build_body_model(
    reference,
    slot,
    lever_contour,
    pivot
):
    """
    Build a rough body model for component-presence validation.

    This model is NOT used to calculate lever orientation.
    """

    roi = _extract_roi(
        reference,
        slot.bbox
    )

    if roi is None:
        return None, None

    h, w = roi.shape[:2]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB
    )

    L = lab[:, :, 0]

    keep = np.ones(
        (h, w),
        dtype=np.uint8
    ) * 255

    shifted = lever_contour.copy()

    shifted[:, :, 0] -= slot.bbox[0]
    shifted[:, :, 1] -= slot.bbox[1]

    cv2.drawContours(
        keep,
        [shifted],
        -1,
        0,
        -1
    )

    yy, xx = np.ogrid[
        :h,
        :w
    ]

    cx = w / 2.0
    cy = h / 2.0

    ellipse = (
        (
            (xx - cx)
            /
            max(w * 0.48, 1.0)
        )
        ** 2

        +

        (
            (yy - cy)
            /
            max(h * 0.48, 1.0)
        )
        ** 2
        <= 1.0
    )

    keep[~ellipse] = 0

    local_pivot = (
        int(
            round(
                pivot[0]
                -
                slot.bbox[0]
            )
        ),
        int(
            round(
                pivot[1]
                -
                slot.bbox[1]
            )
        )
    )

    sweep_radius = int(
        round(
            1.20
            *
            slot.hole_spacing
        )
    )

    yy2, xx2 = np.ogrid[
        :h,
        :w
    ]

    sweep = (
        (
            xx2
            -
            local_pivot[0]
        )
        ** 2

        +

        (
            yy2
            -
            local_pivot[1]
        )
        ** 2

        <= sweep_radius ** 2
    )

    keep[sweep] = 0

    values = L[
        keep > 0
    ]

    if values.size < 100:
        return None, None

    threshold, _ = cv2.threshold(
        values.reshape(-1, 1).astype(
            np.uint8
        ),
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU
    )

    body_mask = (
        (L <= threshold)
        &
        (keep > 0)
    ).astype(
        np.uint8
    ) * 255

    body_mask = cv2.morphologyEx(
        body_mask,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    if cv2.countNonZero(
        body_mask
    ) < 100:
        return None, None

    return gray, body_mask

def _build_reference_component(
    reference,
    slot
):
    """
    Calibrate one component from the known-good reference.

    This establishes:
        - reference pivot
        - reference tip
        - reference angle

    These values are ONLY the expected values.
    """

    roi = _extract_roi(
        reference,
        slot.bbox
    )

    if roi is None:
        return None

    d = slot.hole_spacing

    expected_pivot_global = (
        slot.center[0]
        +
        0.12 * d,

        slot.center[1]
        +
        COMPONENT_ROI_Y_OFFSET_FACTOR * d
        -
        0.02 * d
    )

    expected_pivot = _relative_point(
        expected_pivot_global,
        slot.bbox
    )

    measurement = measure_lever(
        roi,
        expected_pivot=expected_pivot,
        expected_radius=max(55.0, 1.05 * float(d))
    )

    if measurement is None:
        return None