"""Conservative repair of fused side-by-side heading blocks using native PDF text.

No OCR guesses or document-specific phrases: require a clear vertical gutter and
exact character conservation. Plain body columns without a fused heading are not reinterpreted.
"""
from collections import Counter
from statistics import median
import re


def _characters(text):
    return Counter(re.sub(r"\s", "", text))


def separate_paragraphs(document, page_no, textpage, markdown):
    """Restore paragraphs and consecutive numbered entries at native PDF line starts."""
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

    height = document.pages[page_no].size.height
    serializer = MarkdownDocSerializer(doc=document)
    textpage.count_rects()
    rects = [textpage.get_rect(i) for i in range(textpage.count_rects())]
    tables = [p.bbox.to_bottom_left_origin(height) for table in document.tables
              for p in table.prov if p.page_no == page_no]
    for item in document.texts:
        if item.label.value != "text" or not item.prov:
            continue
        boxes = [p.bbox.to_bottom_left_origin(height) for p in item.prov if p.page_no == page_no]
        if not boxes:
            continue
        l, b, r, t = min(v.l for v in boxes), min(v.b for v in boxes), max(v.r for v in boxes), max(v.t for v in boxes)
        if any(l < box.r and r > box.l and b < box.t and t > box.b for box in tables):
            continue
        local = [v for v in rects if l <= (v[0] + v[2]) / 2 <= r and b <= (v[1] + v[3]) / 2 <= t]
        rows = []
        for box in sorted(local, key=lambda v: -(v[1] + v[3]) / 2):
            if rows and abs((box[1] + box[3] - rows[-1][1] - rows[-1][3]) / 2) < 4:
                old = rows[-1]
                rows[-1] = (min(old[0], box[0]), min(old[1], box[1]), max(old[2], box[2]), max(old[3], box[3]))
            else:
                rows.append(box)
        if len(rows) < 2:
            continue
        steps = [a[3] - z[3] for a, z in zip(rows, rows[1:])]
        spacing = median(steps)
        breaks = []
        quotes = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
        searchable = item.text.translate(quotes)
        previous_number = None
        for index, row in enumerate(rows):
            line = textpage.get_text_bounded(row[0] - 1, row[1], row[2] + 1, row[3]).strip()
            marker = re.match(r"^(\d+)[.)]\s+", line)
            number = int(marker[1]) if marker else None
            numbered_entry = number is not None and previous_number is not None and number == previous_number + 1
            if number is not None:
                previous_number = number
            if index == 0 or (not numbered_entry and steps[index - 1] <= 1.25 * spacing):
                continue
            prefix = line[:40].translate(quotes)
            if len(prefix) >= 12 and searchable.count(prefix) == 1:
                position = searchable.index(prefix)
                if position > 0 and searchable[position - 1].isspace():
                    breaks.append(position)
        text = item.text
        for index in sorted(set(breaks), reverse=True):
            text = text[:index].rstrip() + "\n\n" + text[index:]
        if breaks:
            original = serializer.serialize(item=item).text
            replacement = serializer.serialize(item=item.model_copy(update={"text": text})).text
            if original and markdown.count(original) == 1:
                markdown = markdown.replace(original, replacement, 1)
    return markdown


def separate_columns(document, page_no, textpage, markdown):
    """Repair a fused heading plus its following short column blocks, not a whole page."""
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

    height = document.pages[page_no].size.height
    serializer = MarkdownDocSerializer(doc=document)
    items = [item for item in document.texts if item.prov and item.prov[0].page_no == page_no]
    textpage.count_rects()
    rects = [textpage.get_rect(i) for i in range(textpage.count_rects())]
    table_boxes = [p.bbox.to_bottom_left_origin(height) for table in document.tables
                   for p in table.prov if p.page_no == page_no]

    def bounds(item):
        boxes = [p.bbox.to_bottom_left_origin(height) for p in item.prov if p.page_no == page_no]
        return min(b.l for b in boxes), min(b.b for b in boxes), max(b.r for b in boxes), max(b.t for b in boxes)

    def divide(item, cut=None):
        if item.label.value not in {"section_header", "title", "text", "list_item"}:
            return None
        l, b, r, t = bounds(item)
        if any(l < box.r and r > box.l and b < box.t and t > box.b for box in table_boxes):
            return None  # Never reassign table cells to independent text columns.
        if cut is not None:
            if r <= cut:
                return [(0, item.text)]
            if l >= cut:
                return [(1, item.text)]
        local = [v for v in rects if l <= (v[0] + v[2]) / 2 <= r and b <= (v[1] + v[3]) / 2 <= t]
        intervals = []
        for left, right in sorted((v[0], v[2]) for v in local):
            if intervals and left <= intervals[-1][1] + 1:
                intervals[-1][1] = max(right, intervals[-1][1])
            else:
                intervals.append([left, right])
        if len(intervals) < 2:
            return None
        gaps = [(right[0] - left[1], left[1], right[0]) for left, right in zip(intervals, intervals[1:])]
        if cut is None:
            gap, gap_l, gap_r = max(gaps)
            if gap < max(20, 2 * median(v[3] - v[1] for v in local)):
                return None
            cut = (gap_l + gap_r) / 2
        elif not any(gap_l <= cut <= gap_r and gap >= 20 for gap, gap_l, gap_r in gaps):
            return None
        # Do not pad vertically: adjacent Vietnamese accent glyphs can leak into a crop.
        left = textpage.get_text_bounded(l - 1, b, cut, t).replace("\r\n", "\n").strip()
        right = textpage.get_text_bounded(cut, b, r + 1, t).replace("\r\n", "\n").strip()
        if not left or not right or _characters(left + right) != _characters(item.text):
            return None
        return cut, [(0, left), (1, right)]

    repaired, index = 0, 0
    while index < len(items):
        item = items[index]
        if item.label.value not in {"section_header", "title"}:
            index += 1
            continue
        split = divide(item)
        if split is None:
            index += 1
            continue
        cut, parts = split
        group = [(item, parts)]
        end = index + 1
        bottom = bounds(item)[1]
        while end < len(items):
            following = items[end]
            if following.label.value in {"section_header", "title"}:
                break
            box = bounds(following)
            if box[3] > bottom + 2 or bottom - box[3] > 0.08 * height:
                break
            following_parts = divide(following, cut)
            if following_parts is None:
                break
            if isinstance(following_parts, tuple):
                following_parts = following_parts[1]
            group.append((following, following_parts))
            bottom = box[1]
            end += 1
        original = "\n\n".join(serializer.serialize(item=node).text for node, _ in group)
        columns = [[], []]
        for node, pieces in group:
            for side, text in pieces:
                columns[side].append(serializer.serialize(item=node.model_copy(update={"text": text, "orig": text})).text)
        replacement = "\n\n".join(columns[0] + columns[1])
        if original and markdown.count(original) == 1:
            markdown = markdown.replace(original, replacement, 1)
            repaired += 1
        index = end
    return markdown, repaired
