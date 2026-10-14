# data/

| Path | Contents |
| --- | --- |
| `test_set/` | the eight benchmark test sets used in the paper (see `test_set/README.md`) |
| `train/` | not shipped; built from the official training splits with `python -m kghopper.prepare_data` |
| `synthetic/` | not shipped; a small smoke-test set generated with `python -m kghopper.build_data` |
| `kg_cache/` | not shipped; created at run time (simulated-KG neighbourhoods, relation selections) |

How the training data is built, the sources, and the JSONL schema: [`docs/DATA.md`](../docs/DATA.md).
