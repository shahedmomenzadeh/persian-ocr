"""Column-aware RTL reading-order sorter for Persian document blocks and text lines."""

from __future__ import annotations

from statistics import median

from src.schemas.document import BBox, LayoutBlock, TextLine


def _center_x(bbox: BBox) -> float:
    return (bbox[0] + bbox[2]) / 2.0


def _width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _vertical_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    min_h = min(_height(a), _height(b))
    if min_h <= 1e-6:
        return 0.0
    return overlap / min_h


def _x_intervals_overlap(a_range: tuple[float, float], b_range: tuple[float, float], tol: float = 0.0) -> bool:
    return min(a_range[1], b_range[1]) + tol >= max(a_range[0], b_range[0])


def sort_lines_rtl(lines: list[TextLine]) -> list[TextLine]:
    """Sort lines top-to-bottom by visual row, and right-to-left within each row."""
    if len(lines) <= 1:
        return list(lines)

    heights = [max(1.0, _height(line.bbox)) for line in lines]
    med_h = median(heights)
    row_y_tol = med_h * 0.45

    # Sort top-to-bottom first to form visual rows deterministically
    by_y = sorted(lines, key=lambda ln: (ln.bbox[1], -_center_x(ln.bbox)))
    rows: list[list[TextLine]] = []
    row_y_spans: list[tuple[float, float]] = []

    for line in by_y:
        placed = False
        line_cy = (line.bbox[1] + line.bbox[3]) / 2.0
        for idx, row in enumerate(rows):
            ry1, ry2 = row_y_spans[idx]
            row_cy = (ry1 + ry2) / 2.0
            # Same row if vertical overlap is substantial or centers are close
            if (
                _vertical_overlap_ratio(line.bbox, row[0].bbox) >= 0.45
                or abs(line_cy - row_cy) <= row_y_tol
            ):
                row.append(line)
                row_y_spans[idx] = (min(ry1, line.bbox[1]), max(ry2, line.bbox[3]))
                placed = True
                break
        if not placed:
            rows.append([line])
            row_y_spans.append((line.bbox[1], line.bbox[3]))

    # Sort rows top-to-bottom, and segments within each row right-to-left
    row_pairs = sorted(zip(row_y_spans, rows), key=lambda item: item[0][0])
    sorted_lines: list[TextLine] = []
    for _, row in row_pairs:
        row.sort(key=lambda ln: -_center_x(ln.bbox))
        sorted_lines.extend(row)
    return sorted_lines


def _sort_band_columns_rtl(band_blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    """Cluster blocks in a horizontal band into columns, then order RTL then top-to-bottom."""
    if len(band_blocks) <= 1:
        return sorted(band_blocks, key=lambda b: (b.bbox[1], -_center_x(b.bbox)))

    # Cluster by x-interval overlap (using small tolerance for slightly misaligned columns)
    clusters: list[tuple[float, float, list[LayoutBlock]]] = []
    # Sort largest/topmost first or rightmost first to seed columns stably
    for block in sorted(band_blocks, key=lambda b: (-_center_x(b.bbox), b.bbox[1])):
        bx1, _, bx2, _ = block.bbox
        matched_idx: int | None = None
        best_overlap = -1.0

        for idx, (cx1, cx2, _) in enumerate(clusters):
            if _x_intervals_overlap((bx1, bx2), (cx1, cx2), tol=0.0):
                overlap = min(bx2, cx2) - max(bx1, cx1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched_idx = idx

        if matched_idx is not None:
            cx1, cx2, items = clusters[matched_idx]
            items.append(block)
            # Only widen conservatively or keep weighted bounds
            clusters[matched_idx] = (min(cx1, bx1), max(cx2, bx2), items)
        else:
            clusters.append((bx1, bx2, [block]))

    # Merge any clusters whose x-ranges now overlap significantly (>30% of narrower cluster)
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        clusters.sort(key=lambda c: -((c[0] + c[1]) / 2.0))
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ax1, ax2, a_items = clusters[i]
                bx1, bx2, b_items = clusters[j]
                overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                min_w = min(max(1.0, ax2 - ax1), max(1.0, bx2 - bx1))
                if overlap / min_w > 0.30:
                    clusters[i] = (min(ax1, bx1), max(ax2, bx2), a_items + b_items)
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    # Sort columns right-to-left (descending column center-x), blocks within column top-to-bottom
    clusters.sort(key=lambda c: -((c[0] + c[1]) / 2.0))
    ordered: list[LayoutBlock] = []
    for _, _, items in clusters:
        items.sort(key=lambda b: (b.bbox[1], -_center_x(b.bbox)))
        ordered.extend(items)
    return ordered


def sort_blocks_rtl(blocks: list[LayoutBlock], page_width: int) -> list[LayoutBlock]:
    """Sort layout blocks in Persian RTL reading order (band-aware, rightmost column first)."""
    if not blocks:
        return []

    headers = sorted(
        [b for b in blocks if b.type == "header"],
        key=lambda b: (b.bbox[1], -_center_x(b.bbox)),
    )
    footers = sorted(
        [b for b in blocks if b.type == "footer"],
        key=lambda b: (b.bbox[1], -_center_x(b.bbox)),
    )
    body = [b for b in blocks if b.type not in {"header", "footer"}]

    if body:
        min_x = min(b.bbox[0] for b in body)
        max_x = max(b.bbox[2] for b in body)
        content_span = max(float(page_width) * 0.6, max_x - min_x, 1.0)
        span_threshold = content_span * 0.65

        # Sort body top-to-bottom to detect full-width spanning separators (e.g., titles, wide tables)
        body_by_y = sorted(body, key=lambda b: (b.bbox[1], -_center_x(b.bbox)))
        ordered_body: list[LayoutBlock] = []
        current_band: list[LayoutBlock] = []

        for blk in body_by_y:
            is_spanning = _width(blk.bbox) >= span_threshold
            if is_spanning:
                # Check if this spanning block overlaps vertically with the current band;
                # if it lies below (or clearly separates), flush current_band first.
                if current_band:
                    ordered_body.extend(_sort_band_columns_rtl(current_band))
                    current_band = []
                ordered_body.append(blk)
            else:
                current_band.append(blk)

        if current_band:
            ordered_body.extend(_sort_band_columns_rtl(current_band))
    else:
        ordered_body = []

    ordered = headers + ordered_body + footers
    result: list[LayoutBlock] = []
    for idx, blk in enumerate(ordered):
        sorted_lines = sort_lines_rtl(blk.lines) if blk.lines else []
        updated_text = "\n".join(ln.text for ln in sorted_lines if ln.text) if sorted_lines else blk.text
        result.append(
            blk.model_copy(
                update={
                    "reading_order": idx,
                    "lines": sorted_lines,
                    "text": updated_text,
                }
            )
        )
    return result
