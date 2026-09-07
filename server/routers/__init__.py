"""routers package — exports all submodules and their router objects.

Re-exporting router objects here lets pyright resolve submodule attributes
(e.g. `from routers import lifecycle` then `lifecycle.router`) without relying
on pyright re-indexing each new .py file.
"""

from . import (
    backup,
    cognition,
    documents,
    graph,
    knowledge,
    lifecycle,
    logs,
    memories,
    mental_models,
    offload,
    reflect,
    relation,
    search,
    settings,
    stats,
)

# Expose each submodule's router as a package-level attribute.
# pyright needs this for `from routers import lifecycle` → `lifecycle.router`
# to resolve without re-scanning each newly-added router module.
router = relation.router
backup_router = backup.router
documents_router = documents.router
graph_router = graph.router
knowledge_router = knowledge.router
lifecycle_router = lifecycle.router
logs_router = logs.router
memories_router = memories.router
mental_models_router = mental_models.router
offload_router = offload.router
reflect_router = reflect.router
search_router = search.router
settings_router = settings.router
stats_router = stats.router
cognition_router = cognition.router

__all__ = [
    "backup",
    "backup_router",
    "documents",
    "documents_router",
    "graph",
    "graph_router",
    "knowledge",
    "knowledge_router",
    "lifecycle",
    "lifecycle_router",
    "logs",
    "logs_router",
    "memories",
    "memories_router",
    "mental_models",
    "mental_models_router",
    "offload",
    "offload_router",
    "reflect",
    "reflect_router",
    "relation",
    "search",
    "search_router",
    "cognition",
    "cognition_router",
    "stats",
    "stats_router",
]
