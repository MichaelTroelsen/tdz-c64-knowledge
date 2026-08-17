"""KnowledgeBase split into domain mixins (R12 step 2).

server.py assembles these into the KnowledgeBase class. CoreMixin is listed
last in the bases because it alone defines __init__; the others deliberately
define none, so there is no MRO ambiguity to reason about.
"""
from .ingest import IngestMixin
from .figures import FiguresMixin
from .search import SearchMixin
from .entities import EntitiesMixin
from .graph import GraphMixin
from .topics import TopicsMixin
from .temporal import TemporalMixin
from .admin import AdminMixin
from .core import CoreMixin

__all__ = ["IngestMixin", "FiguresMixin", "SearchMixin", "EntitiesMixin", "GraphMixin", "TopicsMixin", "TemporalMixin", "AdminMixin", "CoreMixin"]
