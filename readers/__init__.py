"""
Data readers for shapegraphs.

Each sub-module implements a reader for a specific tracking-data format and
exposes a ``generate_shapegraphs_from_files`` entry-point that returns the
same ``Dict[int, {"original": G, "nominal": G_nominal}]`` structure produced
by ``shapegraphs.frame2sg.generate_shapegraphs``.

Available readers
-----------------
bassek
    DFL / Bassek et al. XML format (requires the ``idsse`` package).
    Install with: ``pip install shapegraphs[bassek]``
"""
