# PrivJail

PrivJail (Privacy Jail) enforces differential privacy in Python.

**Warning: This project is in development and not recommended for production use.**

Security and stability are not guaranteed, and breaking changes may occur. Use at your own risk.

## Getting Started

To install PrivJail:
```sh
pip install privjail
```

To run a decision tree example:
```sh
cd examples/
./download_dataset_adult.bash
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

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2025 TOYOTA MOTOR CORPORATION.
