from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import itertools
import math

import cv2
import numpy as np

from lever_detector import gold_mask

from config import (
    REFERENCE_FILE,
    MIN_SIFT_FEATURES,
    MIN_GOOD_MATCHES,
    MIN_HOMOGRAPHY_INLIERS,
    MIN_INLIER_RATIO,
    RANSAC_REPROJECTION_THRESHOLD,
    MIN_HOMOGRAPHY_CORNER_SCALE,
    MAX_HOMOGRAPHY_CORNER_SCALE,
)


@dataclass
class RegistrationResult:
    aligned: np.ndarray
    homography_ref_to_current: np.ndarray
    homography_current_to_ref: np.ndarray
    good_matches: int
    inliers: int
    inlier_ratio: float
    fixture_inverted: bool = False


class FixtureRegistration:
    """
    Rotation/crop/scale-tolerant registration for the physical 10-component
    fixture.

    Registration answers only one question:
        Where is the real fixture in the current image?

    Lever orientation is NOT used to choose a registration.

    Primary method:
        four physical mounting holes on the large fixture plate
        -> four-point projective transform
        -> verify the warped image using the 10 component layout

    Fallbacks:
        AKAZE
        SIFT

    This prevents a feature matcher from accepting a visually plausible but
    physically incorrect homography.
    """

    def __init__(self, reference_path=REFERENCE_FILE):
        self.reference_path = (
            Path(reference_path)
            if not isinstance(reference_path, Path)
            else reference_path
        )

        self.reference = cv2.imread(
            str(self.reference_path),
            cv2.IMREAD_COLOR,
        )

        if self.reference is None:
            raise RuntimeError(
                f"Could not load reference image: {self.reference_path}"
            )

        self.reference_gray = cv2.cvtColor(
            self.reference,
            cv2.COLOR_BGR2GRAY,
        )

        self.reference_size = (
            self.reference.shape[1],
            self.reference.shape[0],
        )

        self.reference_centres = self._get_reference_centres()

        # Learn the large fixture-plate mounting-hole geometry from the
        # supplied reference image. Nothing about the plate coordinates is
        # hard-coded.
        self.reference_plate = self._find_plate_quad(
            self.reference,
            require_reference_layout=True,
        )

        self.reference_plate_signature = (
            self._quad_signature(
                self.reference_plate
            )
            if self.reference_plate is not None
            else None
        )

        # -------------------------------------------------------------
        # SIFT
        # -------------------------------------------------------------
        self.sift = cv2.SIFT_create(
            nfeatures=7000,
            contrastThreshold=0.012,
            edgeThreshold=10,
        )

        gold = gold_mask(self.reference)
        gold = cv2.dilate(
            gold,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )

        feature_mask = cv2.bitwise_not(gold)

        try:
            (
                self.reference_keypoints,
                self.reference_descriptors,
            ) = self.sift.detectAndCompute(
                self.reference_gray,
                feature_mask,
            )
        except cv2.error:
            self.reference_keypoints = []
            self.reference_descriptors = None

        if (
            self.reference_descriptors is None
            or len(self.reference_keypoints) < MIN_SIFT_FEATURES
        ):
            self.reference_keypoints = []
            self.reference_descriptors = None

        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        # -------------------------------------------------------------
        # AKAZE
        # -------------------------------------------------------------
        self.akaze = cv2.AKAZE_create()
        self.akaze_matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

    # =================================================================
    # REFERENCE CENTRES
    # =================================================================

    def _get_reference_centres(self) -> np.ndarray:
        try:
            from detector import detect_component_slots

            slots = detect_component_slots(
                self.reference
            )

            if len(slots) != 10:
                raise RuntimeError(
                    f"Reference component discovery returned {len(slots)}/10."
                )

            return np.float32(
                [slot.center for slot in slots]
            )

        except Exception as exc:
            raise RuntimeError(
                "REFERENCE REGISTRATION FAILED: "
                "could not calibrate the 10 component centres."
            ) from exc

    # =================================================================
    # GENERIC POINT HELPERS
    # =================================================================

    @staticmethod
    def _warp_points(
        H: np.ndarray,
        points: np.ndarray,
    ) -> Optional[np.ndarray]:
        if H is None or points is None:
            return None

        try:
            p = np.float32(points).reshape(-1, 1, 2)
            out = cv2.perspectiveTransform(
                p,
                H,
            ).reshape(-1, 2)
        except (
            cv2.error,
            ValueError,
            TypeError,
        ):
            return None

        if not np.isfinite(out).all():
            return None

        return out

    @staticmethod
    def _order_quad(
        points: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(
            points,
            dtype=np.float32,
        ).reshape(4, 2)

        center = np.mean(
            points,
            axis=0,
        )

        angles = np.arctan2(
            points[:, 1] - center[1],
            points[:, 0] - center[0],
        )

        return points[
            np.argsort(angles)
        ].astype(np.float32)

    @staticmethod
    def _point_in_quad(
        point: np.ndarray,
        quad: np.ndarray,
    ) -> bool:
        try:
            return (
                cv2.pointPolygonTest(
                    quad.reshape(-1, 1, 2),
                    (
                        float(point[0]),
                        float(point[1]),
                    ),
                    False,
                )
                >= 0
            )
        except cv2.error:
            return False

    # =================================================================
    # HOLE CANDIDATES
    # =================================================================

    @staticmethod
    def _circle_contrast(
        gray: np.ndarray,
        x: float,
        y: float,
        r: float,
    ) -> float:
        h, w = gray.shape

        cx = int(round(x))
        cy = int(round(y))
        rr = max(4, int(round(r)))

        yy, xx = np.ogrid[:h, :w]
        d2 = (
            (xx - cx) ** 2
            +
            (yy - cy) ** 2
        )

        inner = d2 <= int(
            (0.60 * rr) ** 2
        )
        outer = (
            (
                d2 >= int(
                    (1.05 * rr) ** 2
                )
            )
            &
            (
                d2 <= int(
                    (1.55 * rr) ** 2
                )
            )
        )

        if not inner.any() or not outer.any():
            return 0.0

        return float(
            np.mean(gray[outer])
            -
            np.mean(gray[inner])
        )

    def _hole_candidates(
        self,
        image: np.ndarray,
    ):
        """
        Find dark circular holes at several Hough sensitivities.

        Multiple passes make this robust to exposure, focus and JPEG changes.
        """
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (7, 7),
            1.2,
        )

        h, w = gray.shape
        min_dim = float(
            min(h, w)
        )

        min_radius = max(
            10,
            int(round(min_dim * 0.008)),
        )
        # The fixture mounting holes are substantially smaller than the
        # circular component bodies. Limiting the Hough radius prevents body
        # edges from being mistaken for holes.
        max_radius = min(
            42,
            max(
                30,
                int(round(min_dim * 0.030)),
            ),
        )

        min_dist = max(
            24,
            int(round(min_dim * 0.018)),
        )

        raw = []

        for param2 in (
            20,
            23,
            26,
            29,
        ):
            try:
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=1.15,
                    minDist=min_dist,
                    param1=100,
                    param2=param2,
                    minRadius=min_radius,
                    maxRadius=max_radius,
                )
            except cv2.error:
                continue

            if circles is None:
                continue

            for x, y, r in np.round(
                circles[0],
                decimals=1,
            ):
                contrast = self._circle_contrast(
                    gray,
                    x,
                    y,
                    r,
                )

                if contrast < 7.0:
                    continue

                raw.append(
                    (
                        float(x),
                        float(y),
                        float(r),
                        float(contrast),
                    )
                )

        if not raw:
            return []

        # Strongest circles first. Keep enough candidates for the plate corners
        # while avoiding an expensive 4-combination explosion.
        raw.sort(
            key=lambda c: (
                c[3] * max(c[2], 1.0)
            ),
            reverse=True,
        )

        unique = []

        for candidate in raw:
            x, y, r, contrast = candidate

            duplicate = False

            for ux, uy, ur, _ in unique:
                if math.hypot(
                    x - ux,
                    y - uy,
                ) < max(
                    10.0,
                    0.45 * min(r, ur),
                ):
                    duplicate = True
                    break

            if not duplicate:
                unique.append(candidate)

            if len(unique) >= 45:
                break

        return unique

    # =================================================================
    # QUAD GEOMETRY
    # =================================================================

    @staticmethod
    def _quad_metrics(
        points: np.ndarray,
    ):
        """
        Return:
            area,
            rectangle_quality,
            aspect_ratio
        """
        q = FixtureRegistration._order_quad(
            points
        )

        area = abs(
            float(
                cv2.contourArea(
                    q.reshape(-1, 1, 2)
                )
            )
        )

        if area < 100.0:
            return 0.0, 0.0, float("inf")

        sides = np.linalg.norm(
            np.roll(q, -1, axis=0)
            -
            q,
            axis=1,
        )

        if np.any(sides <= 1.0):
            return 0.0, 0.0, float("inf")

        side_a = float(
            (sides[0] + sides[2]) * 0.5
        )
        side_b = float(
            (sides[1] + sides[3]) * 0.5
        )

        opposite_error = (
            abs(sides[0] - sides[2])
            /
            max(side_a, 1.0)
            +
            abs(sides[1] - sides[3])
            /
            max(side_b, 1.0)
        )

        corner_errors = []

        for i in range(4):
            v1 = (
                q[i]
                -
                q[(i - 1) % 4]
            )
            v2 = (
                q[(i + 1) % 4]
                -
                q[i]
            )

            denom = (
                np.linalg.norm(v1)
                *
                np.linalg.norm(v2)
            )

            if denom <= 1e-6:
                return 0.0, 0.0, float("inf")

            cosine = abs(
                float(
                    np.dot(v1, v2)
                    /
                    denom
                )
            )

            corner_errors.append(
                cosine
            )

        rectangle_quality = (
            max(
                0.0,
                1.0 - opposite_error,
            )
            *
            max(
                0.0,
                1.0
                -
                float(
                    np.mean(
                        corner_errors
                    )
                ),
            )
        )

        aspect_ratio = (
            max(side_a, side_b)
            /
            max(
                min(side_a, side_b),
                1.0,
            )
        )

        return (
            area,
            rectangle_quality,
            aspect_ratio,
        )

    @staticmethod
    def _quad_signature(
        points: np.ndarray,
    ) -> np.ndarray:
        q = np.asarray(
            points,
            dtype=np.float32,
        ).reshape(4, 2)

        distances = []

        for i in range(4):
            for j in range(i + 1, 4):
                distances.append(
                    float(
                        np.linalg.norm(
                            q[i] - q[j]
                        )
                    )
                )

        distances = np.sort(
            np.asarray(
                distances,
                dtype=np.float32,
            )
        )

        scale = max(
            float(distances[-1]),
            1e-6,
        )

        return distances / scale

    def _quad_shape_error(
        self,
        candidate: np.ndarray,
    ) -> float:
        if self.reference_plate_signature is None:
            return 0.0

        candidate_signature = (
            self._quad_signature(
                candidate
            )
        )

        return float(
            np.mean(
                np.abs(
                    candidate_signature
                    -
                    self.reference_plate_signature
                )
            )
        )

    # =================================================================
    # FIND THE LARGE FIXTURE PLATE QUAD
    # =================================================================

    def _find_plate_quad(
        self,
        image: np.ndarray,
        require_reference_layout: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Find the four mounting holes on the large fixture plate.

        Reference calibration has an especially strong constraint: all ten
        calibrated component centres must be inside the plate quadrilateral.

        Current images use the reference quad shape plus the largest convincing
        rectangular geometry. This is tolerant of crop/scale/rotation.
        """
        candidates = self._hole_candidates(
            image
        )

        if len(candidates) < 4:
            return None

        # The true plate occupies a large region. Keeping the strongest 35-45
        # candidates is enough for the fixture while suppressing noise.
        pool_size = min(
            len(candidates),
            35,
        )

        pool = candidates[:pool_size]

        best = None

        for indices in itertools.combinations(
            range(len(pool)),
            4,
        ):
            points = np.float32(
                [
                    [
                        pool[i][0],
                        pool[i][1],
                    ]
                    for i in indices
                ]
            )

            q = self._order_quad(
                points
            )

            area, rect_quality, aspect = (
                self._quad_metrics(
                    q
                )
            )

            if rect_quality < 0.45:
                continue

            if aspect > 2.5:
                continue

            if require_reference_layout:
                # The reference plate is the largest strong rectangular
                # mounting-hole pattern. Component-hole rectangles are much
                # smaller, so area and rectangularity separate the plate from
                # the internal component grid without hard-coded coordinates.
                score = (
                    area
                    *
                    rect_quality
                    *
                    rect_quality
                )

            else:
                # Current image: use the learned reference plate geometry plus
                # area and rectangularity. A crop/scale/rotation changes only
                # the absolute size and cyclic order, not this normalized shape.
                shape_error = self._quad_shape_error(q)
                shape_score = math.exp(
                    -shape_error / 0.10
                )
                score = (
                    area
                    *
                    shape_score
                    *
                    rect_quality
                    *
                    rect_quality
                )

            candidate = (
                score,
                q,
                area,
                rect_quality,
                aspect,
            )

            if (
                best is None
                or
                candidate[0]
                >
                best[0]
            ):
                best = candidate

        if best is None:
            return None

        return best[1]

    # =================================================================
    # PLATE SHIFTS
    # =================================================================

    def _plate_hypotheses(
        self,
        current_plate: np.ndarray,
    ):
        """
        Four cyclic correspondences.

        We NEVER test reversed winding because that would allow a mirror image
        to masquerade as a normal photograph.
        """
        current = self._order_quad(
            current_plate
        )

        reference = self.reference_plate

        if reference is None:
            return []

        for shift in range(4):
            current_order = np.roll(
                current,
                -shift,
                axis=0,
            )

            try:
                H = cv2.getPerspectiveTransform(
                    current_order.astype(
                        np.float32
                    ),
                    reference.astype(
                        np.float32
                    ),
                )
            except cv2.error:
                continue

            if not self._homography_sane(
                H
            ):
                continue

            yield shift, H

    # =================================================================
    # VALIDATE PLATE ALIGNMENT WITH THE 10 REAL COMPONENTS
    # =================================================================

    def _layout_error_after_alignment(
        self,
        aligned: np.ndarray,
    ):
        """
        Run the existing component detector AFTER the candidate transform.

        This is crucial for close-up images: the raw image can be too large or
        cropped for the detector's old absolute hole-pair thresholds, while
        the plate-normalized image is back in the calibrated coordinate system.
        """
        try:
            from detector import detect_component_slots

            slots = detect_component_slots(
                aligned
            )
        except Exception:
            return None

        if len(slots) != 10:
            return None

        centres = np.float32(
            [slot.center for slot in slots]
        )

        errors = np.linalg.norm(
            centres
            -
            self.reference_centres,
            axis=1,
        )

        return {
            "slots": slots,
            "centres": centres,
            "errors": errors,
            "median": float(
                np.median(errors)
            ),
            "mean": float(
                np.mean(errors)
            ),
            "max": float(
                np.max(errors)
            ),
            "count": 10,
        }

    def _register_by_plate(
        self,
        image: np.ndarray,
    ) -> Optional[RegistrationResult]:
        """
        Primary registration method.
        """
        if self.reference_plate is None:
            return None

        current_plate = self._find_plate_quad(
            image,
            require_reference_layout=False,
        )

        if current_plate is None:
            return None

        best = None

        for shift, H_current_to_ref in (
            self._plate_hypotheses(
                current_plate
            )
        ):
            try:
                aligned = cv2.warpPerspective(
                    image,
                    H_current_to_ref,
                    self.reference_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            except cv2.error:
                continue

            layout = (
                self._layout_error_after_alignment(
                    aligned
                )
            )

            if layout is None:
                # Keep as a weak fallback candidate. Another shift may provide
                # a real 10-component layout.
                score = 1e6
                candidate = (
                    score,
                    shift,
                    H_current_to_ref,
                    aligned,
                    0,
                    float("inf"),
                    float("inf"),
                )

            else:
                median = layout["median"]
                mean = layout["mean"]
                maximum = layout["max"]

                # Correct correspondence should put all ten component slots
                # in the correct fixture neighbourhood. Because the plate and
                # component faces can sit at slightly different depths, a
                # plate-only homography can leave a few dozen pixels of
                # residual error. That is why this gate is deliberately wider
                # than the final post-refinement validation.
                if (
                    median > 70.0
                    or
                    maximum > 110.0
                ):
                    continue

                score = (
                    median
                    +
                    0.15 * mean
                    +
                    0.03 * maximum
                )

                candidate = (
                    score,
                    shift,
                    H_current_to_ref,
                    aligned,
                    10,
                    median,
                    maximum,
                )

            if (
                best is None
                or
                candidate[0]
                <
                best[0]
            ):
                best = candidate

        if best is None:
            return None

        (
            _,
            _,
            H_current_to_ref,
            aligned,
            inliers,
            median,
            maximum,
        ) = best

        if inliers != 10:
            return None

        # -------------------------------------------------------------
        # Refine the plate homography using the measured component centres.
        # -------------------------------------------------------------
        layout = self._layout_error_after_alignment(
            aligned
        )

        if layout is not None:
            aligned_centres = layout["centres"]

            try:
                H_ref_to_current_initial = (
                    np.linalg.inv(
                        H_current_to_ref
                    )
                )
            except np.linalg.LinAlgError:
                H_ref_to_current_initial = None

            if H_ref_to_current_initial is not None:
                current_centres = self._warp_points(
                    H_ref_to_current_initial,
                    aligned_centres,
                )

                if current_centres is not None:
                    try:
                        H_refined, mask = (
                            cv2.findHomography(
                                current_centres.reshape(
                                    -1,
                                    1,
                                    2,
                                ),
                                self.reference_centres.reshape(
                                    -1,
                                    1,
                                    2,
                                ),
                                0,
                            )
                        )
                    except cv2.error:
                        H_refined = None
                        mask = None
