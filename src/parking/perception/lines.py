"""Conservative line refinement of predicted parking polygons in image coordinates."""
from __future__ import annotations
import cv2
import numpy as np


def _quad(points, fit="conservative"):
    hull = cv2.convexHull(np.asarray(points, np.float32).reshape(-1, 2))
    if fit == "enclosing" and len(hull) >= 4 and hasattr(cv2, "approxPolyN"):
        polygon = cv2.approxPolyN(hull, 4, epsilon_percentage=.25, ensure_convex=True).reshape(-1, 2)
        if len(polygon) == 4:
            return polygon.astype(np.float32)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in (0.01, 0.02, 0.035, 0.05):
        polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
        if len(polygon) == 4:
            return polygon.astype(np.float32)
    return None


def refine_spaces(frame, detections, *, strict_quadrilaterals=True, snap_lines=True, quad_fit="conservative"):
    """Return copies; quadrilateral candidates survive missing line evidence.

    Bright white/yellow paint and local Hough segments support each boundary.
    No right-angle constraint is imposed on perspective-distorted CCTV polygons.
    Confidence remains the model confidence, not a calibrated geometry score.
    """
    output = [dict(d) for d in detections]
    labels = {"parking_space", "empty", "occupied", "vacant", "not occupied"}
    if not any(d.get("label") in labels for d in output):
        return output
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 145), (179, 85, 255))
    yellow = cv2.inRange(hsv, (15, 70, 110), (40, 255, 255))
    paint = cv2.bitwise_or(white, yellow)
    edges = cv2.Canny(paint, 50, 150)
    for detection in output:
        if detection.get("label") not in labels:
            continue
        detection["line_refined"] = False
        detection["line_supported_edges"] = 0
        points = detection.get("points")
        if points is None or len(points) < 4:
            # Axis-aligned object boxes do not provide perspective boundary directions.
            continue
        quad = _quad(points, fit=quad_fit)
        if quad is None or not np.isfinite(quad).all():
            continue
        source_area = abs(cv2.contourArea(np.asarray(points, np.float32)))
        quad_area = abs(cv2.contourArea(quad))
        # Simplify only already quadrilateral-like regions, never arbitrary blobs.
        area_low, area_high = (.65, 1.8) if quad_fit == "enclosing" else (.85, 1.15)
        if source_area < 1 or not area_low <= quad_area / source_area <= area_high:
            continue
        if (quad < 0).any() or (quad[:, 0] >= width).any() or (quad[:, 1] >= height).any():
            continue
        detection["raw_points"] = points
        detection["points"] = quad.tolist()
        detection["geometry_regularized"] = True
        if not snap_lines:
            continue
        sides = np.roll(quad, -1, axis=0) - quad
        lengths = np.linalg.norm(sides, axis=1)
        if lengths.min() < 12:
            continue
        tolerance = max(3., min(12., float(lengths.min()) * 0.12))
        lo = np.maximum(np.floor(quad.min(0) - tolerance * 2), 0).astype(int)
        hi = np.minimum(np.ceil(quad.max(0) + tolerance * 2), (width, height)).astype(int)
        if np.any(hi <= lo):
            continue
        segments = cv2.HoughLinesP(edges[lo[1]:hi[1], lo[0]:hi[0]], 1, np.pi / 180,
                                   threshold=12, minLineLength=max(10, int(lengths.min() * .25)),
                                   maxLineGap=6)
        if segments is None:
            continue
        segments = segments.reshape(-1, 2, 2).astype(float) + lo
        lines, supported = [], 0
        for origin, direction, length in zip(quad, sides / lengths[:, None], lengths):
            matches = []
            for segment in segments:
                vector = segment[1] - segment[0]
                segment_length = np.linalg.norm(vector)
                if segment_length < 1 or abs(np.dot(vector / segment_length, direction)) < np.cos(np.deg2rad(12)):
                    continue
                delta = segment - origin
                distance = np.abs(delta[:, 0] * direction[1] - delta[:, 1] * direction[0])
                along = delta @ direction
                overlap = min(length, along.max()) - max(0, along.min())
                if distance.max() <= tolerance and overlap >= min(length * .2, segment_length * .6):
                    matches.extend(segment)
            if matches:
                samples = np.asarray(matches, np.float32)
                projections = (samples - origin) @ direction
                if np.ptp(projections) >= length * .35:
                    vx, vy, x, y = cv2.fitLine(samples, cv2.DIST_HUBER, 0, .01, .01).ravel()
                    fitted = np.array([vx, vy], float)
                    if abs(fitted @ direction) >= np.cos(np.deg2rad(12)):
                        lines.append((np.array([x, y]), fitted))
                        supported += 1
                        continue
            lines.append((origin, direction))
        detection["line_supported_edges"] = supported
        if supported < 2:
            continue
        corners = []
        for i in range(4):
            a, u = lines[i - 1]
            b, v = lines[i]
            matrix = np.column_stack((u, -v))
            if abs(np.linalg.det(matrix)) < .15:
                break
            corners.append(a + np.linalg.solve(matrix, b - a)[0] * u)
        if len(corners) != 4:
            continue
        candidate = np.asarray(corners, np.float32)
        original_area = abs(cv2.contourArea(quad))
        area = abs(cv2.contourArea(candidate))
        if (not np.isfinite(candidate).all() or not cv2.isContourConvex(candidate)
                or original_area < 1 or not .75 <= area / original_area <= 1.25
                or np.linalg.norm(candidate - quad, axis=1).max() > tolerance * 2
                or (candidate < 0).any() or (candidate[:, 0] >= width).any()
                or (candidate[:, 1] >= height).any()):
            continue
        detection["raw_points"] = points
        detection["points"] = candidate.tolist()
        minimum, maximum = candidate.min(0), candidate.max(0)
        detection["xyxy"] = (*np.floor(minimum).astype(int).tolist(), *np.ceil(maximum).astype(int).tolist())
        detection["line_refined"] = True
    if strict_quadrilaterals:
        # Reject non-quadrilateral parking blobs, but never filter vehicles.
        return [d for d in output if d.get("label") not in labels or d.get("geometry_regularized", False)]
    return output
