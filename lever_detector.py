from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import cv2
import numpy as np

from config import (
    GOLD_H_MIN,
    GOLD_H_MAX,
    GOLD_S_MIN,
    GOLD_L_MIN,
    GOLD_B_MIN,
    MIN_LEVER_AREA,
    MAX_LEVER_AREA,
    MIN_LEVER_DIMENSION,
    PIVOT_MIN_RADIUS,
    PIVOT_MAX_RADIUS,
    PIVOT_SEARCH_RADIUS,
    TIP_MIN_DISTANCE,
    POLAR_BINS,
    POLAR_SMOOTHING,
)


@dataclass
class LeverMeasurement:
    contour: np.ndarray
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    pivot: Tuple[int, int]
    tip: Tuple[int, int]
    angle_deg: float
    reliability: float
    segmentation_score: float
    shape_score: float


def gold_mask(image: np.ndarray, adaptive: bool = False) -> np.ndarray:
    """
    Segment the brass lever.

    Normal mode is conservative and is used where a stable mask is preferable.
    Adaptive mode is used for inspection fallback and is deliberately tolerant
    of shadows, mixed exposure and colour shifts.

    The classifier never uses this mask to decide clockwise/anticlockwise; the
    mask only supplies the physical lever silhouette.
    """
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    h, sat, value = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    if not adaptive:
        material = (
            (h >= GOLD_H_MIN)
            & (h <= GOLD_H_MAX)
            & (sat >= GOLD_S_MIN)
            & (B >= GOLD_B_MIN - 10)
        )

        values = L[material]
        if values.size < 40:
            return np.zeros(L.shape, dtype=np.uint8)

        otsu, _ = cv2.threshold(
            values.reshape(-1, 1).astype(np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        threshold = max(float(otsu), float(GOLD_L_MIN))
        mask = (
            material
            & (L >= threshold)
        ).astype(np.uint8) * 255

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        return mask

                                                                     
                                        
                                                                     
    Lf = L.astype(np.float32)
    Sf = sat.astype(np.float32)
    Bf = B.astype(np.float32)
    Af = A.astype(np.float32)
    Vf = value.astype(np.float32)

    blue = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    red = image[:, :, 2].astype(np.float32)

    rg_excess = 0.5 * (red + green) - blue

                                                                          
                                                                      
    b10, b90 = np.percentile(Bf, [10, 90])
    s10, s90 = np.percentile(Sf, [10, 90])
    rg10, rg90 = np.percentile(rg_excess, [10, 90])

    b_norm = np.clip(
        (Bf - b10) / max(float(b90 - b10), 1.0),
        0.0,
        1.0,
    )
    s_norm = np.clip(
        (Sf - s10) / max(float(s90 - s10), 1.0),
        0.0,
        1.0,
    )
    rg_norm = np.clip(
        (rg_excess - rg10) / max(float(rg90 - rg10), 1.0),
        0.0,
        1.0,
    )

                                                                    
    brass_score = (
        0.50 * b_norm
        + 0.28 * s_norm
        + 0.22 * rg_norm
    )

                                                                          
                                                       
    local_B = Bf - cv2.GaussianBlur(Bf, (0, 0), 11.0)
    local_S = Sf - cv2.GaussianBlur(Sf, (0, 0), 9.0)
    local_L = Lf - cv2.GaussianBlur(Lf, (0, 0), 11.0)

    hue_ok = (
        (h <= 85)
        & (sat >= 5)
    )

    score_gate = (
        brass_score >= max(
            0.34,
            float(np.percentile(brass_score, 66)),
        )
    )

    local_gate = (
        (local_B >= np.percentile(local_B, 74))
        |
        (local_S >= np.percentile(local_S, 78))
        |
        (local_L >= np.percentile(local_L, 78))
    )

    broad_colour = (
        hue_ok
        & (
            score_gate
            | local_gate
        )
    )

                                                                              
                                                                  
    warm_branch = (
        (h <= 95)
        & (sat >= 3)
        & (rg_norm >= 0.50)
        & (b_norm >= 0.42)
    )

                                                                      
    bright_branch = (
        (h <= 60)
        & (sat >= 12)
        & (Bf >= max(82.0, float(np.percentile(Bf, 35))))
        & (Lf >= max(35.0, float(np.percentile(Lf, 30))))
    )

    mask = (
        broad_colour
        |
        warm_branch
        |
        bright_branch
    ).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=1,
    )

    return mask

def _candidate_contours(roi: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    mask = gold_mask(roi)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    valid = []
    rh, rw = mask.shape
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_LEVER_AREA or area > MAX_LEVER_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_LEVER_DIMENSION or h < MIN_LEVER_DIMENSION:
            continue
                                                                             
                                                                           
        if w > rw * 0.85 or h > rh * 0.90:
            continue
        valid.append(contour)
    return mask, valid


def _dark_contrast(gray: np.ndarray, x: int, y: int, radius: int) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    inner = d2 <= radius ** 2
    outer = (d2 <= int(radius * 2.0) ** 2) & ~(
        d2 <= int(radius * 1.25) ** 2
    )
    if not np.any(inner) or not np.any(outer):
        return 0.0
    return float(np.mean(gray[outer]) - np.mean(gray[inner]))


def _find_pivot(
    roi_gray: np.ndarray,
    lever_mask: np.ndarray,
    expected_pivot: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[int, int]]:
    """Find the physical lever pivot without confusing it with mounting holes.

    The previous implementation trusted any Hough circle that overlapped the
    gold mask. In photographs with glare/shadow, one of the nearby black
    mounting holes could therefore win and become the pivot. That corrupts the
    handedness calculation because the entire lever silhouette is then measured
    around the wrong origin.

    New priority:
        1. Distance-transform maximum of the lever silhouette.
           The physical pivot boss is the thickest part of the lever.
        2. If an expected pivot is available, prefer the thickest point in a
           generous neighbourhood around that expected location.
        3. Only then use circular Hough evidence as a fallback, and only when
           it is close to a strong silhouette thickness maximum.

    No mounting-hole position is used to determine CW/ACW.
    """
    if (
        roi_gray is None
        or roi_gray.size == 0
        or lever_mask is None
        or cv2.countNonZero(lever_mask) == 0
    ):
        return None

                                                                     
                                
                                                                     
    dist = cv2.distanceTransform(
        lever_mask,
        cv2.DIST_L2,
        5,
    )

    global_max = float(np.max(dist))
    if global_max <= 0.0:
        return None

    global_y, global_x = np.unravel_index(
        int(np.argmax(dist)),
        dist.shape,
    )
    global_loc = (
        int(global_x),
        int(global_y),
    )

                                                                     
                                                     
                                                                     
    if expected_pivot is not None:

        ex = float(expected_pivot[0])
        ey = float(expected_pivot[1])

        h, w = dist.shape
        yy, xx = np.ogrid[:h, :w]

                                                                             
                                                                            
        search_radius = max(
            float(PIVOT_SEARCH_RADIUS) * 2.0,
            global_max * 3.2,
            0.80 * max(h, w),
        )

        local = (
            (xx - ex) ** 2
            +
            (yy - ey) ** 2
            <=
            search_radius ** 2
        )

        if np.any(local):

            restricted = np.where(
                local,
                dist,
                0.0,
            )

            local_max = float(
                np.max(restricted)
            )

            if local_max >= 0.32 * global_max:

                                                                             
                plateau = (
                    restricted
                    >=
                    0.82 * local_max
                ).astype(
                    np.uint8
                )

                if cv2.countNonZero(
                    plateau
                ) >= 3:

                    moments = cv2.moments(
                        plateau,
                        binaryImage=True,
                    )

                    if moments["m00"] > 0:

                        px = (
                            moments["m10"]
                            /
                            moments["m00"]
                        )

                        py = (
                            moments["m01"]
                            /
                            moments["m00"]
                        )

                        return (
                            int(round(px)),
                            int(round(py)),
                        )

                _, _, _, loc = cv2.minMaxLoc(
                    restricted
                )

                return (
                    int(loc[0]),
                    int(loc[1]),
                )

                                                                     
                                   
                                                                     
                                                                             
                                                                    
    if expected_pivot is None:
        return global_loc

                                                                     
                                                  
                                                                     
    blurred = cv2.GaussianBlur(
        roi_gray,
        (5, 5),
        1.1,
    )

    try:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=16,
            param1=80,
            param2=14,
            minRadius=PIVOT_MIN_RADIUS,
            maxRadius=PIVOT_MAX_RADIUS,
        )
    except cv2.error:
        circles = None

    if circles is not None:

        candidates = []
        ex, ey = map(
            float,
            expected_pivot,
        )

        for x, y, r in np.round(
            circles[0]
        ).astype(int):

            if not (
                0 <= x < lever_mask.shape[1]
                and
                0 <= y < lever_mask.shape[0]
            ):
                continue

            rr = max(
                3,
                int(
                    round(
                        r * 0.7
                    )
                ),
            )

            yy, xx = np.ogrid[
                :lever_mask.shape[0],
                :lever_mask.shape[1],
            ]

            disk = (
                (xx - x) ** 2
                +
                (yy - y) ** 2
                <=
                rr ** 2
            )

            if not np.any(disk):
                continue

            overlap = float(
                np.mean(
                    lever_mask[disk] > 0
                )
            )

            if overlap < 0.40:
                continue

            thickness = float(
                dist[y, x]
            )

            distance_to_expected = math.hypot(
                x - ex,
                y - ey,
            )

            if thickness < 0.25 * global_max:
                continue

            score = (
                3.0 * thickness
                +
                0.7 * r
                +
                12.0 * overlap
                -
                1.4 * distance_to_expected
            )

            candidates.append(
                (
                    score,
                    x,
                    y,
                )
            )

        if candidates:

            candidates.sort(
                reverse=True
            )

            return (
                int(candidates[0][1]),
                int(candidates[0][2]),
            )

                              
    return global_loc

def _polar_signature(
    contour: np.ndarray,
    pivot: Tuple[int, int],
    bins: int = POLAR_BINS,
) -> np.ndarray:
    """
    Directed radial silhouette signature around the physical pivot.

    Unlike PCA, the signature preserves the asymmetric end of the lever, so
    it has a real 0..360 direction rather than a 180-degree axis ambiguity.
    """
    points = contour.reshape(-1, 2).astype(np.float32)
    px, py = float(pivot[0]), float(pivot[1])
    dx = points[:, 0] - px
    dy = points[:, 1] - py
    angles = np.mod(np.degrees(np.arctan2(dy, dx)), 360.0)
    radii = np.hypot(dx, dy)

    signature = np.zeros(bins, dtype=np.float32)
    for angle, radius in zip(angles, radii):
        idx = min(bins - 1, int(angle * bins / 360.0))
        if radius > signature[idx]:
            signature[idx] = radius

                                                                       
                                                                    
    nonzero = np.flatnonzero(signature > 0)
    if nonzero.size >= 2:
        for i in range(bins):
            if signature[i] == 0:
                prev = nonzero[nonzero < i]
                nxt = nonzero[nonzero > i]
                p = int(prev[-1]) if prev.size else int(nonzero[-1]) - bins
                n = int(nxt[0]) if nxt.size else int(nonzero[0]) + bins
                t = (i - p) / max(n - p, 1)
                signature[i] = (1.0 - t) * signature[p % bins] + t * signature[n % bins]

    k = max(3, int(POLAR_SMOOTHING) | 1)
    pad = k // 2
    extended = np.r_[signature[-pad:], signature, signature[:pad]]
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smoothed = np.convolve(extended, kernel, mode="same")[pad:-pad]
    return smoothed.astype(np.float32)


def _tip_from_signature(
    contour: np.ndarray,
    pivot: Tuple[int, int],
) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
    """Find the *physical pointed end* of the lever and measure its angle.

    The old implementation used the maximum radial value of the polar
    signature. That can select a broad/rounded contour extremity and can make
    a reversed lever report an angle close to the reference. The lever is
    asymmetric: its pointed/tapered end is the direction marker.

    This routine therefore uses contour curvature + distance from the physical
    pivot. The pointed end has a much smaller local turning angle than the
    rounded pivot/base. Several nearby contour samples are combined so a single
    noisy pixel cannot move the measured angle.
    """
    if contour is None or len(contour) < 12:
        return None, None

    pts = contour.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    px, py = map(float, pivot)

    radii = np.hypot(pts[:, 0] - px, pts[:, 1] - py)
    max_radius = float(np.max(radii))
    if max_radius < TIP_MIN_DISTANCE:
        return None, None

                                                                           
                                                                             
                                                
    hull = cv2.convexHull(contour, returnPoints=True).reshape(-1, 2).astype(np.float32)
    m = len(hull)
    if m < 5:
        return None, None

    hx = hull[:, 0] - px
    hy = hull[:, 1] - py
    hr = np.hypot(hx, hy)
    hmax = float(np.max(hr))
    if hmax < TIP_MIN_DISTANCE:
        return None, None

                                                                        
                                                                            
                                                     
    step = max(2, int(round(m * 0.035)))
    candidates = []
    for i in range(m):
        r = float(hr[i])
                                                                            
                                                                            
        if r < max(TIP_MIN_DISTANCE, 0.52 * hmax):
            continue

        prev = hull[(i - step) % m] - hull[i]
        nxt = hull[(i + step) % m] - hull[i]
        a = float(np.linalg.norm(prev))
        b = float(np.linalg.norm(nxt))
        if a < 1e-6 or b < 1e-6:
            continue

        cosang = float(np.dot(prev, nxt) / (a * b))
        interior = math.degrees(math.acos(float(np.clip(cosang, -1.0, 1.0))))

                                                                              
                                                                          
        pointedness = max(0.0, 180.0 - interior) / 180.0
        radial_score = min(r / max(hmax, 1.0), 1.0)
        score = 1.8 * pointedness + 0.7 * radial_score
        candidates.append((score, interior, r, i))

    if not candidates:
        return None, None

    candidates.sort(reverse=True)
    _, _, _, best_i = candidates[0]

                                                                            
                                                                            
    neighbourhood = max(1, int(round(m * 0.025)))
    ids = [((best_i + j) % m) for j in range(-neighbourhood, neighbourhood + 1)]
    tip_xy = np.mean(hull[ids], axis=0)

    dx = float(tip_xy[0] - px)
    dy = float(tip_xy[1] - py)
    distance = math.hypot(dx, dy)
    if distance < TIP_MIN_DISTANCE:
        return None, None

    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    tip = (int(round(tip_xy[0])), int(round(tip_xy[1])))
    return tip, float(angle)


def _directed_shape_similarity(
    contour: np.ndarray,
    pivot: Tuple[int, int],
    reference_signature: Optional[np.ndarray],
) -> float:
    """Compare lever silhouettes while preserving absolute direction.

    Unlike the legacy circular-correlation score, this deliberately does not
    search over rotations. A 180-degree-reversed lever must therefore score
    differently from the known-good reference shape.
    """
    if reference_signature is None:
        return 0.5
    current = _polar_signature(contour, pivot, len(reference_signature))
    if np.max(current) <= 0 or np.max(reference_signature) <= 0:
        return 0.0
    a = current / max(float(np.max(current)), 1.0)
    b = reference_signature / max(float(np.max(reference_signature)), 1.0)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    corr = float(np.dot(a, b) / denom)
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))



def directed_rotation_offset(
    contour: np.ndarray,
    pivot: Tuple[int, int],
    reference_signature: Optional[np.ndarray],
) -> Tuple[Optional[float], float]:
    """Estimate the lever's absolute rotation relative to the reference.

    Both signatures are measured around the physical pivot. We search the
    circular shift that best aligns the CURRENT silhouette to the calibrated
    reference silhouette. Unlike PCA, this retains the asymmetric lever tip,
    so a 180-degree reversal produces an approximately 180-degree offset.

    Returns (offset_degrees, similarity), where positive offset means the
    current lever is rotated counter-clockwise in image coordinates relative
    to the reference.
    """
    if reference_signature is None or len(reference_signature) < 16:
        return None, 0.0
    current = _polar_signature(contour, pivot, len(reference_signature))
    if float(np.max(current)) <= 0 or float(np.max(reference_signature)) <= 0:
        return None, 0.0

    a = current.astype(np.float32)
    b = reference_signature.astype(np.float32)
    a /= max(float(np.max(a)), 1.0)
    b /= max(float(np.max(b)), 1.0)

    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return None, 0.0

                                                                       
                                                
    best_shift = 0
    best_corr = -1.0
    for shift in range(len(a)):
        corr = float(np.dot(np.roll(a, shift), b) / (na * nb))
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    offset = (-best_shift * 360.0 / len(a)) % 360.0
    if offset > 180.0:
        offset -= 360.0
    similarity = float(np.clip((best_corr + 1.0) * 0.5, 0.0, 1.0))
    return float(offset), similarity


def _best_rotational_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the best cosine similarity after ANY circular rotation.

    The circular shift removes the absolute camera/image rotation from the
    decision.  The sequence itself is not reflected, so clockwise and its
    mirror image remain different.
    """
    if a is None or b is None or len(a) != len(b) or len(a) < 16:
        return 0.0

    aa = np.asarray(a, dtype=np.float32).copy()
    bb = np.asarray(b, dtype=np.float32).copy()

    aa /= max(float(np.max(aa)), 1e-6)
    bb /= max(float(np.max(bb)), 1e-6)

    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))

    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0

                                                                            
                                                           
    best = -1.0
    for shift in range(len(aa)):
        rolled = np.roll(aa, shift)
        corr = float(np.dot(rolled, bb) / (na * nb))
        if corr > best:
            best = corr

    return float(np.clip((best + 1.0) * 0.5, 0.0, 1.0))


def _mirrored_polar_signature(signature: np.ndarray) -> np.ndarray:
    """Return the angularly reflected version of a polar signature.

    Reflection reverses clockwise angular order.  Any axis of reflection is
    equivalent after the rotational search, so a simple reversal is enough.
    """
    return np.asarray(signature, dtype=np.float32)[::-1].copy()


def chirality_similarity(
    contour: np.ndarray,
    pivot: Tuple[int, int],
    reference_signature: Optional[np.ndarray],
) -> Tuple[float, float, float]:
    """Measure normal-vs-mirrored lever shape, independent of rotation.

    Returns:
        normal_score: best rotational match to the known-good clockwise shape
        mirror_score: best rotational match to its reflected shape
        margin:       normal_score - mirror_score

    This is the actual orientation classifier.  It does not use mounting-hole
    geometry and it does not compare absolute tip angles.
    """
    if reference_signature is None or len(reference_signature) < 16:
        return 0.0, 0.0, 0.0

    current = _polar_signature(
        contour,
        pivot,
        len(reference_signature),
    )

    if float(np.max(current)) <= 0 or float(np.max(reference_signature)) <= 0:
        return 0.0, 0.0, 0.0

    normal = _best_rotational_similarity(
        current,
        reference_signature,
    )

    mirrored_reference = _mirrored_polar_signature(reference_signature)
    mirrored = _best_rotational_similarity(
        current,
        mirrored_reference,
    )

    return float(normal), float(mirrored), float(normal - mirrored)

def _shape_similarity(
    contour: np.ndarray,
    pivot: Tuple[int, int],
    reference_signature: Optional[np.ndarray],
) -> float:
    if reference_signature is None:
        return 0.5

    current = _polar_signature(contour, pivot, len(reference_signature))
    if np.max(current) <= 0 or np.max(reference_signature) <= 0:
        return 0.0

    a = current / max(float(np.max(current)), 1.0)
    b = reference_signature / max(float(np.max(reference_signature)), 1.0)

                                                                               
    best = -1.0
    for shift in range(0, len(a), 2):
        rolled = np.roll(a, shift)
        denom = float(np.linalg.norm(rolled) * np.linalg.norm(b))
        corr = float(np.dot(rolled, b) / denom) if denom > 1e-8 else 0.0
        best = max(best, corr)
    return float(np.clip((best + 1.0) * 0.5, 0.0, 1.0))



def _resample_contour(contour: np.ndarray, count: int = 256) -> Optional[np.ndarray]:
    """Resample a closed contour at uniform arc-length intervals."""
    if contour is None or len(contour) < 8:
        return None
    pts = contour.reshape(-1, 2).astype(np.float32)
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(np.sum(seg))
    if total <= 1e-6:
        return None
    cumulative = np.r_[0.0, np.cumsum(seg)]
    targets = np.linspace(0.0, total, count, endpoint=False)
    out = np.zeros((count, 2), dtype=np.float32)
    for i, t in enumerate(targets):
        idx = int(np.searchsorted(cumulative, t, side="right") - 1)
        idx = min(max(idx, 0), len(pts) - 1)
        length = float(seg[idx])
        alpha = 0.0 if length <= 1e-8 else (t - cumulative[idx]) / length
        out[i] = pts[idx] * (1.0 - alpha) + pts[(idx + 1) % len(pts)] * alpha
    return out


def _normalized_polar_from_contour(
    contour: np.ndarray,
    pivot: Tuple[int, int],
    bins: int = 720,
) -> Optional[np.ndarray]:
    """Dense rotation-indexed radial silhouette around the physical pivot."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    px, py = float(pivot[0]), float(pivot[1])
    dx = pts[:, 0] - px
    dy = pts[:, 1] - py
    ang = np.mod(np.degrees(np.arctan2(dy, dx)), 360.0)
    rad = np.hypot(dx, dy)
    result = np.zeros(bins, dtype=np.float32)
    for a, r in zip(ang, rad):
        k = int((a / 360.0) * bins) % bins
        if r > result[k]:
            result[k] = r
    nz = np.flatnonzero(result > 0)
    if nz.size < 8:
        return None
    for i in range(bins):
        if result[i] <= 0:
            prev = nz[nz < i]
            nxt = nz[nz > i]
            p = int(prev[-1]) if prev.size else int(nz[-1]) - bins
            n = int(nxt[0]) if nxt.size else int(nz[0]) + bins
            t = (i - p) / max(float(n - p), 1.0)
            result[i] = (1.0 - t) * result[p % bins] + t * result[n % bins]
    k = 11
    pad = k // 2
    ext = np.r_[result[-pad:], result, result[:pad]]
    result = np.convolve(
        ext,
        np.ones(k, dtype=np.float32) / k,
        mode="same",
    )[pad:-pad]
    scale = float(np.max(result))
    if scale <= 1e-6:
        return None
    result /= scale
    return result.astype(np.float32)


def _best_circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) != len(b):
        return -1.0
    aa = a.astype(np.float32) - float(np.mean(a))
    bb = b.astype(np.float32) - float(np.mean(b))
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= 1e-8 or nb <= 1e-8:
        return -1.0
    best = -1.0
                                                                     
    step = max(1, len(aa) // 180)
    for shift in range(0, len(aa), step):
        rolled = np.roll(aa, shift)
        best = max(best, float(np.dot(rolled, bb) / (na * nb)))
    return float(np.clip(best, -1.0, 1.0))



def _normalize_shape_for_procrustes(contour: np.ndarray, pivot: Tuple[int, int], count: int = 160) -> Optional[np.ndarray]:
    pts = _resample_contour(contour, count)
    if pts is None:
        return None
    q = pts.astype(np.float32).copy()
    q[:, 0] -= float(pivot[0])
    q[:, 1] -= float(pivot[1])
    scale = float(np.max(np.linalg.norm(q, axis=1)))
    if scale <= 1e-6:
        return None
    q /= scale
    return q


def _best_contour_procrustes(
    contour: np.ndarray,
    reference_contour: np.ndarray,
    current_pivot: Tuple[int, int] = None,
    reference_pivot: Tuple[int, int] = None,
    mirrored: bool = False,
) -> Optional[float]:
    """Best rigid/scale shape distance, allowing cyclic start-point shifts.

    The normal score allows rotation but never reflection. The mirror score
    explicitly reflects the reference first. This makes the comparison
    rotation-invariant yet reflection-sensitive.
    """
    if contour is None or reference_contour is None:
        return None

    if current_pivot is None:
        m = cv2.moments(contour)
        if m["m00"] <= 0:
            return None
        current_pivot = (
            int(round(m["m10"] / m["m00"])),
            int(round(m["m01"] / m["m00"])),
        )

    if reference_pivot is None:
        m = cv2.moments(reference_contour)
        if m["m00"] <= 0:
            return None
        reference_pivot = (
            int(round(m["m10"] / m["m00"])),
            int(round(m["m01"] / m["m00"])),
        )

    a = _normalize_shape_for_procrustes(
        contour,
        current_pivot,
        160,
    )
    b = _normalize_shape_for_procrustes(
        reference_contour,
        reference_pivot,
        160,
    )

    if a is None or b is None:
        return None

    if mirrored: