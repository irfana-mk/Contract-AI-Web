import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text and tables from a PDF using pdfplumber.
    Tables are rendered as pipe-separated rows and interleaved with body text
    in top-to-bottom reading order.
    """
    page_parts: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts: list[tuple[float, str]] = []  # (y_position, text)
            table_bboxes: list[tuple] = []

            # ── 1. Tables ────────────────────────────────────────────────────
            try:
                tables = page.find_tables()
                for table_obj in tables:
                    bbox = table_obj.bbox  # (x0, top, x1, bottom)
                    table_bboxes.append(bbox)
                    rows: list[str] = []
                    for row in table_obj.extract():
                        if row is None:
                            continue
                        cells = [
                            str(cell).strip()
                            for cell in row
                            if cell is not None and str(cell).strip()
                        ]
                        if cells:
                            rows.append(" | ".join(cells))
                    if rows:
                        parts.append((bbox[1], "\n".join(rows)))
            except Exception as e:
                print(f"Table extraction skipped for a page: {e}")

            # ── 2. Body text (words outside table bounding boxes) ────────────
            try:
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=True,
                )
                # Group words into lines by their top-y coordinate
                lines: dict[float, list[str]] = {}
                for word in words:
                    top = round(word["top"], 1)
                    # Skip words that fall inside a table bbox
                    in_table = any(
                        bbox[0] <= word["x0"] and word["x1"] <= bbox[2]
                        and bbox[1] <= word["top"] and word["bottom"] <= bbox[3]
                        for bbox in table_bboxes
                    )
                    if in_table:
                        continue
                    lines.setdefault(top, []).append(word["text"])

                for top_y in sorted(lines):
                    line_text = " ".join(lines[top_y]).strip()
                    if line_text:
                        parts.append((top_y, line_text))
            except Exception as e:
                print(f"Body text extraction skipped for a page: {e}")

            # ── 3. Sort top-to-bottom and join ───────────────────────────────
            parts.sort(key=lambda t: t[0])
            page_text = "\n".join(fragment for _, fragment in parts).strip()
            if page_text:
                page_parts.append(page_text)

    return "\n\n".join(page_parts)
