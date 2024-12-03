# PrivJail

## Getting Started

To install PrivJail:
```sh
pip install .
```

To run a decision tree example:
```sh
cd examples/
./download_dataset.bash
python decision_tree.py
```

## Development

This project is managed using uv.

Launch a REPL with PrivJail loaded:
```sh
uv run python
```

Test:
```sh
uv run pytest
```

Type check:
```sh
uv run mypy --strict src/ test/
```
