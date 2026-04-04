"""Build-time validation for the pipeline processor graph.

Checks structural integrity of the registered processor graph:
sources exist, sinks exist, all outputs have consumers, and the
graph is connected. Called by ``Application.build()`` before
returning the wired application.
"""

from typed_flow.engine.registry import SINK_VARIANTS, SOURCE_VARIANTS, PipelineRegistry


def validate_graph(registry: PipelineRegistry) -> list[str]:
    """Validate pipeline graph integrity and return error messages.

    An empty list means the graph is valid. Checks performed:
    1. At least one source node exists.
    2. At least one sink node exists.
    3. Every source output type has at least one consumer.
    4. The graph is connected (no orphan nodes).

    Delegates to ``PipelineRegistry.validate()`` for structural checks
    (orphans, unconsumed types, connectivity), and adds source/sink
    presence checks on top.
    """
    errors: list[str] = []
    nodes = registry.nodes

    if not nodes:
        errors.append("No nodes registered in the pipeline")
        return errors

    source_names = [name for name, node in nodes.items() if node.variant in SOURCE_VARIANTS]
    if not source_names:
        errors.append("Pipeline must have at least one source node")

    sink_names = [name for name, node in nodes.items() if node.variant in SINK_VARIANTS]
    if not sink_names:
        errors.append("Pipeline must have at least one sink node")

    errors.extend(registry.validate())

    return errors
