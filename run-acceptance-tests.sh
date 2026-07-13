#!/usr/bin/env bash
set -euo pipefail
python acceptance-pipeline/parse_specs.py
python acceptance-pipeline/generate_tests.py
python -m pytest generated-acceptance-tests -q
