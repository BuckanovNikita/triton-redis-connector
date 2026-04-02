"""ModelBundle groups processors for a single model into a pipeline segment.

A convenience wrapper so users can declare per-model processor sets
and expand them into the flat list expected by ``Application.build()``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelBundle:
    """Groups the four core processors for one model.

    Each field holds a processor instance that will be registered in
    the pipeline. The ``name`` is used as a prefix for node names
    when registering the bundle's processors.
    """

    name: str
    preprocessor: object
    inference: object
    postprocessor: object
    sink: object


def expand_bundle(bundle: ModelBundle) -> list[object]:
    """Return the bundle's processors in pipeline order.

    The returned list is suitable for passing directly into
    ``Application.build(processors=[...])`` or extending a larger
    processor list.
    """
    return [
        bundle.preprocessor,
        bundle.inference,
        bundle.postprocessor,
        bundle.sink,
    ]
