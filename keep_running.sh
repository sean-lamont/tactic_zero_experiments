#!/bin/bash
until python3 experiment_transformer_end_to_end.py; do
    echo "Process crashed with exit code $?.  Respawning.." >&2
    echo $(date) >> crash_times.txt
    sleep 1
done



