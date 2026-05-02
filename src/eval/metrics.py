"""
Quantitative evaluation metrics for generated SVG samples.
Metrics computed:
  - XML validity rate
  - SVG render rate
  - Structural validity
  - Perplexity on a token dataset
"""

import logging
from pathlib import Path

import torch
import numpy as np
from lxml import etree
import cairosvg

logger = logging.getLogger(__name__)


def is_valid_xml(svg: str) -> bool:
    """Return True if the SVG string parses as valid XML."""
    try:
        etree.fromstring(svg.encode())
        return True
    except etree.XMLSyntaxError:
        return False


def renders_to_png(svg: str) -> bool:
    """Return True the SVG renders without error."""
    try:
        cairosvg.svg2png(bytestring=svg.encode())
        return True
    except Exception:
        return False


def is_structurally_valid(svg: str) -> bool:
    """
    Return True if the SVG passes the following structural checks:
      - Root element is <svg>
      - Has a viewBox attribute
      - All tags are closed
    """
    try:
        root = etree.fromstring(svg.encode())
    except etree.XMLSyntaxError:
        return False

    # Root must be <svg>
    tag = root.tag
    if not (tag == "svg" or tag.endswith("}svg")):
        return False

    # Must have a viewBox
    attrs = {k.split("}")[-1]: v for k, v in root.attrib.items()}
    if "viewBox" not in attrs:
        return False

    return True


def evaluate_samples(svgs: list[str]) -> dict:
    """Compute quantitative metrics on a list of SVG strings."""
    total = len(svgs)
    xml_valid   = [is_valid_xml(s) for s in svgs]
    renders     = [renders_to_png(s) for s in svgs]
    structural  = [is_structurally_valid(s) for s in svgs]

    metrics = {
        "n_total":                  total,
        "n_xml_valid":              sum(xml_valid),
        "n_renders":                sum(renders),
        "n_structurally_valid":     sum(structural),
        "xml_validity_rate":        sum(xml_valid) / total if total else 0.0,
        "render_rate":              sum(renders)   / total if total else 0.0,
        "structural_validity_rate": sum(structural)/ total if total else 0.0,
    }

    logger.info(
        f"Evaluation over {total} samples: "
        f"XML valid={metrics['xml_validity_rate']:.1%}  "
        f"Renders={metrics['render_rate']:.1%}  "
        f"Structural={metrics['structural_validity_rate']:.1%}"
    )
    return metrics
