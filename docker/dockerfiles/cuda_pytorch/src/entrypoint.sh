#!/bin/sh
set -e

pip install -r /tmp/editable_requirements.txt --no-deps
exec "$@"