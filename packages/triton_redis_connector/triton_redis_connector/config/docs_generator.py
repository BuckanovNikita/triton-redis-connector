"""Generate markdown documentation from pydantic config models.

Introspects pydantic BaseModel classes via ``model_json_schema()`` and
produces a human-readable markdown table for each model and its nested
sub-models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


def generate_config_docs(config_class: type[BaseModel]) -> str:
    """Generate markdown documentation from a pydantic config model.

    Walks the JSON schema produced by ``model_json_schema()`` and formats
    each model as a markdown section with a field table. Nested models
    referenced via ``$defs`` are expanded into their own sections.

    Args:
        config_class: The root pydantic BaseModel class to document.

    Returns:
        A complete markdown string with all fields, types, and defaults.

    """
    schema = config_class.model_json_schema()
    sections: list[str] = []

    _render_model_section(
        sections,
        title=config_class.__name__,
        model_schema=schema,
        defs=schema.get("$defs", {}),
    )

    rendered_refs: set[str] = set()
    _render_nested_defs(sections, schema.get("$defs", {}), rendered_refs)

    return "\n".join(sections)


def _render_model_section(
    sections: list[str],
    title: str,
    model_schema: dict[str, Any],
    defs: dict[str, Any],
) -> None:
    description = model_schema.get("description", "")
    sections.append(f"## {title}")
    if description:
        sections.append(f"\n{description}\n")
    sections.append("")

    properties: dict[str, Any] = model_schema.get("properties", {})
    required_fields: set[str] = set(model_schema.get("required", []))

    if not properties:
        sections.append("*No fields.*\n")
        return

    sections.append("| Field | Type | Required | Default | Description |")
    sections.append("|-------|------|----------|---------|-------------|")

    for field_name, field_schema in properties.items():
        type_str = _resolve_type(field_schema, defs)
        is_required = field_name in required_fields
        required_str = "yes" if is_required else "no"
        default_val = _format_default(field_schema)
        field_description = field_schema.get("description", "")
        sections.append(f"| `{field_name}` | `{type_str}` | {required_str} | {default_val} | {field_description} |")

    sections.append("")


def _render_nested_defs(
    sections: list[str],
    defs: dict[str, Any],
    rendered: set[str],
) -> None:
    for def_name, def_schema in defs.items():
        if def_name in rendered:
            continue
        rendered.add(def_name)
        _render_model_section(sections, title=def_name, model_schema=def_schema, defs=defs)


def _resolve_type(field_schema: dict[str, Any], defs: dict[str, Any]) -> str:  # noqa: PLR0911
    if "$ref" in field_schema:
        return _ref_name(field_schema["$ref"])

    if "allOf" in field_schema:
        parts = [_resolve_type(sub, defs) for sub in field_schema["allOf"]]
        return parts[0] if len(parts) == 1 else " & ".join(parts)

    if "anyOf" in field_schema:
        parts = [_resolve_type(sub, defs) for sub in field_schema["anyOf"]]
        return " | ".join(parts)

    if "enum" in field_schema:
        enum_values = ", ".join(repr(v) for v in field_schema["enum"])
        return f"enum({enum_values})"

    json_type: str = field_schema.get("type", "any")

    if json_type == "array":
        items = field_schema.get("items", {})
        item_type = _resolve_type(items, defs)
        if "prefixItems" in field_schema:
            prefix_types = [_resolve_type(p, defs) for p in field_schema["prefixItems"]]
            return f"tuple[{', '.join(prefix_types)}]"
        return f"list[{item_type}]"

    if json_type == "object":
        additional = field_schema.get("additionalProperties", {})
        if additional:
            val_type = _resolve_type(additional, defs)
            return f"dict[str, {val_type}]"
        return "object"

    type_map: dict[str, str] = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "null": "None",
    }
    return type_map.get(json_type, json_type)


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", maxsplit=1)[-1]


def _format_default(field_schema: dict[str, Any]) -> str:
    if "default" not in field_schema:
        return "-"
    default = field_schema["default"]
    if default is None:
        return "`None`"
    if isinstance(default, str):
        return f'`"{default}"`'
    if isinstance(default, bool):
        return f"`{default}`"
    return f"`{default}`"
