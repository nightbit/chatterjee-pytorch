#!/usr/bin/env bash
# Make sure our top-level folder is on PYTHONPATH so `import losses…` works
export PYTHONPATH="$(pwd)"
pytest -q