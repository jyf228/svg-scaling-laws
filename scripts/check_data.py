import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.tokenizer import SVGTokenizer

# Load tokenizer
tok = SVGTokenizer.from_pretrained("data/tokenizer")

# Load a slice of train.bin
data = np.fromfile("data/processed/train.bin", dtype=np.uint16)

# Decode first few sequences
print("=== First 500 tokens decoded ===")
print(tok.decode(data[:500].tolist()))

print("\n=== Random sample from middle ===")
mid = len(data) // 2
print(tok.decode(data[mid:mid+500].tolist()))

# Test encoding/decoding consistency
test_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M 10.75 5.8 L 9.5 7.0"/></svg>'
ids = tok.encode(test_svg)
decoded = tok.decode(ids)
print("Original: ", test_svg)
print("Decoded:  ", decoded)
print("Match:", test_svg.strip() == decoded.strip())
