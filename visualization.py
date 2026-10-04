from pathlib import Path
import cv2
import numpy as np


GREEN = (0, 220, 0)
RED = (0, 0, 230)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)


def _transform_point(H, point):
    if H is None or point is None:
        return None
    try:
        p = np.float32([[[float(point[0]), float(point[1])]]])
        out = cv2.perspectiveTransform(p, H)[0, 0]
        if not np.isfinite(out).all():
            return None
        return int(round(float(out[0]))), int(round(float(out[1])))
    except (cv2.error, TypeError, ValueError, IndexError):
        return None


def _transform_contour(H, contour):
    if H is None or contour is None:
        return None
    try:
        p = contour.astype(np.float32)
        out = cv2.perspectiveTransform(p, H)
        if not np.isfinite(out).all():
            return None
        return np.round(out).astype(np.int32)
    except (cv2.error, TypeError, ValueError):
        return None


def _transform_polygon(H, polygon):
    if H is None or polygon is None:
        return None
    try:
        p = polygon.astype(np.float32).reshape(1, -1, 2)
        out = cv2.perspectiveTransform(p, H)[0]
        if not np.isfinite(out).all():
            return None
        return np.round(out).astype(np.int32)
    except (cv2.error, TypeError, ValueError):
        return None


def _clean_contour(contour):
    """
    Build a clean contour for DISPLAY ONLY.

    The original detector contour is never modified.
    A temporary binary mask is cleaned and the largest external contour is
    recovered so small branches/noise do not appear as scribbles.
    """
    if contour is None or len(contour) < 5:
        return contour

    try:
        points = contour.reshape(-1, 2).astype(np.int32)

        bx, by, bw, bh = cv2.boundingRect(points)

        if bw <= 0 or bh <= 0:
            return contour

        pad = max(
            4,
            int(round(0.05 * max(bw, bh))),
        )

        local = points.copy()
        local[:, 0] -= (bx - pad)
        local[:, 1] -= (by - pad)

        canvas_w = max(
            24,
            bw + 2 * pad + 2,
        )
        canvas_h = max(
            24,
            bh + 2 * pad + 2,
        )

        mask = np.zeros(
            (canvas_h, canvas_w),
            dtype=np.uint8,
        )

        cv2.fillPoly(
            mask,
            [local.reshape(-1, 1, 2)],
            255,
        )

        
        close_size = max(
            3,
            int(round(0.008 * max(bw, bh))) | 1,
        )

        kernel = np.ones(
            (close_size, close_size),
            np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            return contour

        largest = max(
            contours,
            key=cv2.contourArea,
        )

        original_area = float(
            cv2.contourArea(
                local.reshape(-1, 1, 2)
            )
        )

        cleaned_area = float(
            cv2.contourArea(
                largest
            )
        )

        
        if (
            original_area <= 0
            or
            cleaned_area < 0.50 * original_area
        ):
            return contour

        largest = largest.copy()

        largest[:, 0, 0] += (
            bx - pad
        )
        largest[:, 0, 1] += (
            by - pad
        )

        
        perimeter = float(
            cv2.arcLength(
                largest,
                True,
            )
        )

        simplified = cv2.approxPolyDP(
            largest,
            max(
                0.45,
                0.0010 * perimeter,
            ),
            True,
        )

        if (
            simplified is not None
            and
            len(simplified) >= 8
        ):
            return simplified

        return largest

    except (
        cv2.error,
        TypeError,
        ValueError,
    ):
        return contour

def _draw_clean_lever(
    output,
    contour,
    pivot,
    tip,
    color,
):
    """
    Draw the actual lever silhouette with a clean direction arrow.

    Detection/classification is NOT changed here.
    """

    clean = _clean_contour(
        contour
    )

    if clean is None:
        return

    # --------------------------------------------------------
    # ACTUAL SHAPE OUTLINE
    # --------------------------------------------------------

    cv2.polylines(
        output,
        [clean],
        True,
        color,
        2,
        cv2.LINE_AA,
    )

    if pivot is None:
        return

    # --------------------------------------------------------
    # PIVOT
    # --------------------------------------------------------

    cv2.circle(
        output,
        pivot,
        4,
        color,
        -1,
        cv2.LINE_AA,
    )



    pts = clean.reshape(
        -1,
        2,
    ).astype(
        np.float32
    )

    px = float(
        pivot[0]
    )
    py = float(
        pivot[1]
    )

    distances = np.hypot(
        pts[:, 0] - px,
        pts[:, 1] - py,
    )

    if distances.size == 0:
        return

    visual_tip = pts[
        int(
            np.argmax(
                distances
            )
        )
    ]

   
    if tip is not None:

        detector_tip = np.float32(
            [
                float(
                    tip[0]
                ),
                float(
                    tip[1]
                ),
            ]
        )

        detector_distance = float(
            np.linalg.norm(
                detector_tip
                -
                np.float32(
                    [
                        px,
                        py,
                    ]
                )
            )
        )

        visual_distance = float(
            np.max(
                distances
            )
        )

        if detector_distance > visual_distance:
            visual_tip = detector_tip

    dx = float(
        visual_tip[0]
        -
        px
    )

    dy = float(
        visual_tip[1]
        -
        py
    )

    distance = float(
        np.hypot(
            dx,
            dy
        )
    )

    if distance < 12.0:
        return



    start_ratio = 0.18
    end_ratio = 0.86

    start = (
        int(
            round(
                px
                +
                dx
                *
                start_ratio
            )
        ),
        int(
            round(
                py
                +
                dy
                *
                start_ratio
            )
        ),
    )

    end = (
        int(
            round(
                px
                +
                dx
                *
                end_ratio
            )
        ),
        int(
            round(
                py
                +
                dy
                *
                end_ratio
            )
        ),
    )

    
    cv2.arrowedLine(
        output,
        start,
        end,
        color,
        2,
        cv2.LINE_AA,
        tipLength=0.12,
    )

def _draw_component_label(
    output,
    polygon,
    display_id,
    color,
):
    """
    Large, readable C01-C10 label.

    The label is placed at the upper-left corner of the component
    rectangle and never inside the lever.
    """

    if (
        polygon is None
        or
        len(polygon) == 0
    ):
        return

    x = int(
        np.min(
            polygon[:, 0]
        )
    )

    y = int(
        np.min(
            polygon[:, 1]
        )
    )

    text = (
        f"C{display_id:02d}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thickness = 2

    (
        tw,
        th,
    ), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    
    text_x = max(
        6,
        x + 9,
    )

    text_y = max(
        th + baseline + 5,
        y + th + 7,
    )

    
    cv2.rectangle(
        output,
        (
            text_x - 5,
            text_y - th - 6,
        ),
        (
            text_x + tw + 6,
            text_y + baseline + 4,
        ),
        DARK,
        -1,
    )

    cv2.putText(
        output,
        text,
        (
            text_x,
            text_y,
        ),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def _polygon_center(polygon):
    if polygon is None or len(polygon) == 0:
        return None

    try:
        moments = cv2.moments(polygon)

        if moments["m00"] != 0:
            return (
                float(
                    moments["m10"]
                    /
                    moments["m00"]
                ),
                float(
                    moments["m01"]
                    /
                    moments["m00"]
                ),
            )

        return (
            float(
                np.mean(
                    polygon[:, 0]
                )
            ),
            float(
                np.mean(
                    polygon[:, 1]
                )
            ),
        )

    except cv2.error:
        return None


def _current_display_order(results, H):
    """
    Determine operator-facing numbering from the CURRENT image position.

    Internal/reference component IDs are not changed.

    Horizontal:
        top row    -> C01..C05
        bottom row -> C06..C10

    Vertical:
        left column  -> C01..C05
        right column -> C06..C10

    PCA is used so the same rule works when the fixture is rotated.
    """

    positions = []

    for index, result in enumerate(results):
        polygon = _transform_polygon(
            H,
            result.get(
                "slot_polygon"
            ),
        )

        center = _polygon_center(
            polygon
        )

        if center is None:
            contour = _transform_contour(
                H,
                result.get(
                    "contour"
                ),
            )

            if (
                contour is not None
                and
                len(contour) > 0
            ):
                center = (
                    float(
                        np.mean(
                            contour[:, 0, 0]
                        )
                    ),
                    float(
                        np.mean(
                            contour[:, 0, 1]
                        )
                    ),
                )

        if center is not None:
            positions.append(
                (
                    index,
                    center[0],
                    center[1],
                )
            )

    if (
        len(positions) != len(results)
        or
        len(results) != 10
    ):
        return {
            index: int(
                result.get(
                    "id",
                    index + 1,
                )
            )
            for index, result
            in enumerate(results)
        }

    points = np.float32(
        [
            [x, y]
            for _, x, y in positions
        ]
    )

    center = np.mean(
        points,
        axis=0,
    )

    centered = points - center

    try:
        covariance = np.cov(
            centered.T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        long_axis = eigenvectors[
            :,
            int(
                np.argmax(
                    eigenvalues
                )
            ),
        ]

        short_axis = eigenvectors[
            :,
            int(
                np.argmin(
                    eigenvalues
                )
            ),
        ]

    except (
        ValueError,
        np.linalg.LinAlgError,
    ):
        long_axis = np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )
        short_axis = np.array(
            [0.0, 1.0],
            dtype=np.float32,
        )

    long_values = centered @ long_axis
    short_values = centered @ short_axis

    # Split by the short axis into the two visible rows/columns.
    order_short = np.argsort(
        short_values,
        kind="stable",
    )

    group_a = order_short[:5]
    group_b = order_short[5:]


    horizontal = (
        abs(float(long_axis[0]))
        >=
        abs(float(long_axis[1]))
    )

    if horizontal:
        
        first_group, second_group = sorted(
            [group_a, group_b],
            key=lambda group: float(
                np.mean(points[group, 1])
            ),
        )

        first_group = first_group[
            np.argsort(
                points[first_group, 0],
                kind="stable",
            )
        ]

        second_group = second_group[
            np.argsort(
                points[second_group, 0],
                kind="stable",
            )
        ]
    else:

        first_group, second_group = sorted(
            [group_a, group_b],
            key=lambda group: float(
                np.mean(points[group, 0])
            ),
        )

        first_group = first_group[
            np.argsort(
                points[first_group, 1],
                kind="stable",
            )
        ]

        second_group = second_group[
            np.argsort(
                points[second_group, 1],
                kind="stable",
            )
        ]

    display_order = (
        list(first_group)
        +
        list(second_group)
    )

    return {
        int(
            positions[index][0]
        ): int(
            display_id
        )
        for display_id, index
        in enumerate(
            display_order,
            start=1,
        )
    }


def get_current_display_map(inspection):
    """Public helper so the UI result panel uses exactly the same numbering."""
    if inspection is None:
        return {}

    H = None

    if inspection.registration is not None:
        H = inspection.registration.homography_ref_to_current

    return _current_display_order(
        inspection.results,
        H,
    )



def draw_inspection(
    image,
    inspection,
):
    output = image.copy()

    if (
        inspection is None
        or
        not inspection.success
    ):
        cv2.putText(
            output,
            "FIXTURE REGISTRATION FAILED",
            (
                30,
                60,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            RED,
            3,
            cv2.LINE_AA,
        )
        return output

    H = None

    if inspection.registration is not None:
        H = inspection.registration.homography_ref_to_current

    display_map = _current_display_order(
        inspection.results,
        H,
    )

    for result_index, result in enumerate(
        inspection.results
    ):
        status = result.get(
            "status",
            "NOT DETECTED",
        )

        color = (
            GREEN
            if status == "CORRECT"
            else RED
            if status == "WRONG ORIENTATION"
            else ORANGE
        )

        polygon = _transform_polygon(
            H,
            result.get(
                "slot_polygon"
            ),
        )

        contour = _transform_contour(
            H,
            result.get(
                "contour"
            ),
        )

        pivot = _transform_point(
            H,
            result.get(
                "pivot"
            ),
        )

        tip = _transform_point(
            H,
            result.get(
                "tip"
            ),
        )


        if polygon is not None:
            cv2.polylines(
                output,
                [polygon],
                True,
                color,
                2,
                cv2.LINE_AA,
            )

        # Clean lever visualization.
        _draw_clean_lever(
            output,
            contour,
            pivot,
            tip,
            color,
        )

        display_id = int(
            display_map.get(
                result_index,
                result.get(
                    "display_id",
                    result.get(
                        "id",
                        result_index + 1,
                    ),
                ),
            )
        )

        _draw_component_label(
            output,
            polygon,
            display_id,
            color,
        )

    # Summary only; no CW/ACW words are drawn on components.
    summary = (
        f"Components {inspection.components_detected}/10  "
        f"Levers {inspection.levers_detected}/10"
    )

    cv2.rectangle(
        output,
        (
            15,
            15,
        ),
        (
            470,
            65,
        ),
        DARK,
        -1,
    )

    cv2.putText(
        output,
        summary,
        (
            25,
            48,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    return output


def draw_reference_discovery(
    reference,
    components,
):
    output = reference.copy()

    for comp in components:
        if comp.polygon is not None:
            cv2.polylines(
                output,
                [comp.polygon],
                True,
                GREEN,
                2,
                cv2.LINE_AA,
            )

        _draw_clean_lever(
            output,
            comp.lever_contour,
            comp.pivot,
            comp.tip,
            GREEN,
        )

        _draw_component_label(
            output,
            comp.polygon,
            comp.component_id,
            GREEN,
        )

    return output


def draw_stage_component_detection(
    image,
    components,
):
    output = image.copy()

    for comp in components:
        if comp.polygon is not None:
            cv2.polylines(
                output,
                [comp.polygon],
                True,
                GREEN,
                2,
                cv2.LINE_AA,
            )

        _draw_component_label(
            output,
            comp.polygon,
            comp.component_id,
            GREEN,
        )

        for hole in comp.holes:
            cv2.circle(
                output,
                (
                    int(
                        round(
                            hole[0]
                        )
                    ),
                    int(
                        round(
                            hole[1]
                        )
                    ),
                ),
                max(
                    4,
                    int(
                        round(
                            hole[2]
                        )
                    ),
                ),
                GREEN,
                2,
                cv2.LINE_AA,
            )

    return output