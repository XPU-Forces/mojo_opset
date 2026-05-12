"""UC normalization operators are intentionally not registered yet.

Dynamic norm kernels currently hit TileLang-UC lifter language-boundary gaps
(dynamic scalar RHS and tail masking / select). Keep this module class-free so
Mojo falls back to the core implementation until those gaps are resolved.
"""
