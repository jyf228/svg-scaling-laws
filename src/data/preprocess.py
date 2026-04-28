"""
SVG normalization and cleaning utilities.
"""

import logging
import re
from pathlib import Path
from typing import Optional
from lxml import etree
from cairosvg import svg2png

from src.utils.config import get_config
from src.utils.io_utils import write_data_to_disk


logger = logging.getLogger(__name__)

config = get_config("data/data")

if config["debug_mode"]:
    logger.setLevel(logging.DEBUG)

# Configure XMLParser for SVG cleaning
PARSER = etree.XMLParser(
    remove_blank_text=True,
    remove_comments=True,
    remove_pis=True,
)

# Metadata and non-rendering tags to drop
DROP_TAGS = set(config["cleaning"]["drop_tags"])

# Number of decimal places to round coordinates to
COORDINATE_PRECISION = config["cleaning"]["coordinate_precision"]

# Matches floats with a decimal point (including scientific notation and negatives)
RE_FLOAT = re.compile(r"-?\d+\.\d+(?:e[+-]?\d+)?", re.IGNORECASE)

# Length filtering thresholds
MIN_CHARS = config["cleaning"]["min_char_length"]
MAX_CHARS = config["cleaning"]["max_char_length"]

VALIDATE_RENDER = config["cleaning"]["validate_render"]


def _parse(svg: str) -> Optional[etree._Element]:
    """
    Parse an SVG string, stripping comments, processing instructions, and unnecessary whitespace.
    """
    try:
        return etree.fromstring(svg.encode(), parser=PARSER)
    except etree.XMLSyntaxError:
        logger.debug("Failed to parse SVG XML - skipping. Error: %s", etree.XMLSyntaxError)
        return None


def _drop_tags(root: etree._Element, tags_to_drop: set[str]) -> None:
    """
    Drop all elements that have non-rendering tags from the XML tree.
    """
    for el in list(root.iter()):
        tag = etree.QName(el).localname
        if tag in tags_to_drop:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def _strip_leading_and_trailing_whitespace(root: etree._Element) -> None:
    """
    Strip leading and trailing whitespace from element text and tail strings.
    """
    for el in root.iter():
        if el.text:
            el.text = el.text.strip()
        if el.tail:
            el.tail = el.tail.strip()


def _strip_nonascii_attr_values(root: etree._Element) -> None:
    """
    Remove attributes whose values contain non-ASCII characters.
    """
    for el in root.iter():
        non_ascii = [k for k, v in el.attrib.items() if not v.isascii()]
        for k in non_ascii:
            del el.attrib[k]


def _canonicalize_attributes(root: etree._Element) -> None:
    """
    Canonicalize attribute order by sorting them alphabetically for each element.
    Ensures consistent attribute ordering.
    """
    for el in root.iter():
        if el.attrib:
            items = sorted(el.attrib.items())
            el.attrib.clear()
            el.attrib.update(items)


def _round_match(precision: int):
    """Returns a function that rounds regex matches to the specified decimal precision."""
    fmt = f"{{:.{precision}f}}"

    def _replace(m: re.Match) -> str:
        rounded = round(float(m.group()), precision)
        text = fmt.format(rounded)
        return text

    return _replace


def _normalize_coordinate_precision(root: etree._Element, precision: int) -> None:
    """Round coordinate values to the specified precision to reduce vocab size."""
    replacer = _round_match(precision)

    for el in root.iter():
        for attr, val in el.attrib.items():
            # Skip coordinate precision normalization if there are no floats in the value 
            # (determined by the presence of a decimal point)
            if "." in val:
                # Modify the attribute value with the rounded version
                el.set(attr, RE_FLOAT.sub(replacer, val))


def _filter_length(svg: str, min_chars: int, max_chars: int) -> bool:
    """Returns True if the SVG string length is within the specified char lengths."""
    length = len(svg)
    if length < min_chars:
        logger.debug(f"SVG too short - skipping.")
        return False
    if length > max_chars:
        logger.debug(f"SVG too long - skipping.")
        return False
    return True


def _validate_svg_render(svg: str) -> bool:
    """Validates that the SVG can be rendered by CairoSVG, returns False if not."""
    try:
        svg2png(bytestring=svg.encode())
        return True
    except Exception:
        logger.debug("SVG failed to render - skipping.")
        return False
    

def clean_svg(raw_svg: str) -> Optional[str]:
    """ 
    Clean and normalize a single SVG string.
    Returns the cleaned SVG, or None if the SVG is invalid, empty after cleaning, or fails to render.
    """
    # Step 1: Parse the SVG XML, performing basic cleaning
    svg = _parse(raw_svg)
    # Exit if parsing fails (XML is not well-formed)
    if svg is None:
        return None
    
    # Step 2: Drop non-rendering tags from the parsed XML tree
    _drop_tags(svg, DROP_TAGS)

    # Step 3: Remove attributes with non-ASCII values
    _strip_nonascii_attr_values(svg)

    # Step 4: Strip leading/trailing whitespace from text and tail
    _strip_leading_and_trailing_whitespace(svg)

    # Step 5: Canonicalize attribute order
    _canonicalize_attributes(svg)

    # Step 6: Round coordinate precision to reduce vocabulary size
    _normalize_coordinate_precision(svg, COORDINATE_PRECISION)

    # Serialize back to a string
    svg_str = etree.tostring(svg, encoding="unicode")
    if not svg_str:
        return None

    # Step 7: Re-parse the serialized output to sanity check the result is still valid XML
    if _parse(svg_str) is None:
        return None

    # Step 7: Length filtering
    if not _filter_length(svg_str, MIN_CHARS, MAX_CHARS):
        return None

    # Step 8: Render validation
    if VALIDATE_RENDER:
        if not _validate_svg_render(svg_str):
            return None

    return svg_str


def process_svgs(svgs: list[str]) -> list[str]:
    """
    Preprocesses a list of raw SVG strings, dropping invalid ones. 
    Returns a list of cleaned and normalized SVG strings.
    """
    logger.info(f"Processing {len(svgs)} SVGs...")

    cleaned: list[str] = []
    for raw in svgs:
        processed_svg = clean_svg(raw)
        if processed_svg is not None:
            cleaned.append(processed_svg)

    total = len(svgs)
    kept = len(cleaned)
    logger.info(f"SVG cleaning: {kept} / {total} kept (invalid={total - kept})")

    # Persist to disk as a plain text file
    cleaned_dir = config["data_paths"]["cleaned_dir"]
    out_path = Path(cleaned_dir) / "cleaned.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_data_to_disk(cleaned, out_path)
    logger.info(f"Saved cleaned SVGs to {out_path}.")

    return cleaned
