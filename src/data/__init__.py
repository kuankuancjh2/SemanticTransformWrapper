# Re-export data utilities. corpus.py / dataset.py live at src/ top level;
# this package exposes them under src.data for cleaner imports.
from ..corpus import build_bundled_corpus
from ..dataset import (TextPairDataset, collate_pairs, load_split, save_split)

__all__ = ["build_bundled_corpus", "TextPairDataset", "collate_pairs",
           "load_split", "save_split"]
