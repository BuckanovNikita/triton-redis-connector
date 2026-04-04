"""Mermaid diagram generator for pipeline visualization at startup.

Produces a Mermaid flowchart from the PipelineRegistry, showing
sources, processors, sinks, and event type edges between them.
Error and metric edges are excluded to keep the diagram readable.
"""

from typed_flow.engine.registry import SINK_VARIANTS, SOURCE_VARIANTS, PipelineRegistry
from typed_flow.errors.models import PipelineError
from typed_flow.metrics.collector import MetricEvent


def _is_excluded_type(t: type) -> bool:
    """Check if a type should be excluded from the diagram (errors, metrics)."""
    try:
        return issubclass(t, (PipelineError, MetricEvent))
    except TypeError:
        return False


def _node_shape(node_name: str, variant_group: str) -> str:
    if variant_group == "source":
        return f"    {node_name}([{node_name}])"
    if variant_group == "sink":
        return f"    {node_name}[[{node_name}]]"
    return f"    {node_name}[{node_name}]"


def generate_mermaid(registry: PipelineRegistry) -> str:
    """Generate a Mermaid graph LR flowchart from the pipeline registry.

    Node shapes:
        Sources  -> stadium shape  ([...])
        Processors -> rectangle    [...]
        Sinks    -> subroutine     [[...]]

    Edges are labeled with event type names. MetricEvent edges are excluded.
    """
    lines: list[str] = ["graph LR"]

    nodes = registry.nodes
    for name, node in sorted(nodes.items()):
        if node.variant in SOURCE_VARIANTS:
            group = "source"
        elif node.variant in SINK_VARIANTS:
            group = "sink"
        else:
            group = "processor"
        lines.append(_node_shape(name, group))

    type_consumers = registry.type_consumers
    for name, node in sorted(nodes.items()):
        for out_type in sorted(node.output_types, key=lambda t: t.__name__):
            if _is_excluded_type(out_type):
                continue
            type_name = out_type.__name__
            lines.extend(
                f"    {name} -->|{type_name}| {consumer_name}"
                for consumer_name in sorted(type_consumers.get(out_type, []))
            )

    return "\n".join(lines)
