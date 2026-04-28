"""
I/O utilities
"""

def write_data_to_disk(svg_strings: list[str], out_path: str) -> None:
    """Utility function to write a list of SVG strings to a text file."""
    with open(out_path, "w", encoding="utf-8") as fh:
        for svg in svg_strings:
            # Make sure each SVG is on one line
            fh.write(svg.replace("\n", " ") + "\n")
