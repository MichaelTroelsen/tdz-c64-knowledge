"""EntitiesMixin assembled from its sub-mixins (R12 step 4).

kb/entities.py was 2,554 lines / 27 methods, just over the ~2,500-line
module target. It is split here into two sub-mixins by responsibility -
entity extraction plus the background extraction-job queue, and entity
relationships/the relationship graph/analytics - each its own module,
comfortably under the target.

The public name `EntitiesMixin` (and `from kb import EntitiesMixin`) is
unchanged: this package's __init__ composes it from the sub-mixins below,
so server.py needs no edit.

Splitting a mixin into pieces and recombining by inheritance means a
method defined in two pieces would silently shadow one of them, and the
method count would still look right. Guard it explicitly, the same way
mcp_tools/handlers.py guards its handler-dict merge: assert no method name
appears twice across the sub-mixins, and that the union equals the
original 27-name set.
"""
from ._extraction import _ExtractionMixin
from ._relationships import _RelationshipsMixin

_SUB_MIXINS = [_ExtractionMixin, _RelationshipsMixin]

_EXPECTED_METHODS = frozenset({
    "_normalize_entity_text", "_extract_entities_regex", "extract_entities",
    "_extraction_worker_loop", "_process_extraction_job",
    "_recover_extraction_jobs", "queue_entity_extraction",
    "get_extraction_status", "get_all_extraction_jobs", "get_entities",
    "search_entities", "find_docs_by_entity", "get_entity_stats",
    "export_entities", "extract_entities_bulk", "export_relationships",
    "extract_entity_relationships", "get_entity_relationships",
    "find_related_entities", "search_by_entity_pair",
    "extract_relationships_bulk", "add_relationship", "remove_relationship",
    "get_entity_analytics", "get_relationships", "get_related_documents",
    "get_relationship_graph",
})

_seen: dict = {}
for _cls in _SUB_MIXINS:
    for _name in vars(_cls):
        if _name.startswith("__"):
            continue
        if _name in _seen:
            raise RuntimeError(
                f"duplicate method {_name!r} defined in both "
                f"{_seen[_name].__name__} and {_cls.__name__} - this would "
                "silently shadow one of them"
            )
        _seen[_name] = _cls

_actual = frozenset(_seen)
if _actual != _EXPECTED_METHODS:
    _missing = _EXPECTED_METHODS - _actual
    _extra = _actual - _EXPECTED_METHODS
    raise RuntimeError(
        f"EntitiesMixin sub-mixin method set changed unexpectedly: "
        f"missing={sorted(_missing)} extra={sorted(_extra)}"
    )


class EntitiesMixin(_ExtractionMixin, _RelationshipsMixin):
    pass
