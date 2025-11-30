"""
Fusion module for hybrid search result combination.

Provides strategies for fusing results from multiple search methods
(vector and BM25) into unified rankings.

Strategies:
    - FusionStrategy: Protocol defining the fusion interface
    - RSFusion: Relative Score Fusion with min-max normalization
    - DBSFusion: Distribution-Based Score Fusion (3-sigma normalization)
    - RRFusion: Reciprocal Rank Fusion

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.search.fusion.base import FusionStrategy
from zapomni_core.search.fusion.dbsf import DBSFusion, fuse_dbsf
from zapomni_core.search.fusion.rrf import RRFusion
from zapomni_core.search.fusion.rsf import RSFusion, fuse_rsf

__all__ = [
    "FusionStrategy",
    "RRFusion",
    "RSFusion",
    "fuse_rsf",
    "DBSFusion",
    "fuse_dbsf",
]
