# MSTR Options Protection — Architecture Diagrams

Standalone Mermaid files live in [`diagrams/`](diagrams/).

| File | Type | What it shows |
|---|---|---|
| [`diagrams/context.mmd`](diagrams/context.mmd) | `flowchart TB` | System boundary — external sources (Yahoo Finance, file cache, results dir) and internal module data flow |
| [`diagrams/class.mmd`](diagrams/class.mmd) | `classDiagram` | Classes, module-level API, and dependency relationships between all modules |
| [`diagrams/sequence.mmd`](diagrams/sequence.mmd) | `sequenceDiagram` | Complete `python main.py` execution flow across all 10 participants, from data load to saved results |
