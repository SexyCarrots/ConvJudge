
#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

$PYTHON_BIN evaluate_simulated_conversations.py --model gpt-4o &
$PYTHON_BIN evaluate_simulated_conversations.py --model "gpt-4o-mini" &
$PYTHON_BIN evaluate_simulated_conversations.py --model "gpt-5" &
$PYTHON_BIN evaluate_simulated_conversations.py --model "gpt-5-mini" &
$PYTHON_BIN evaluate_simulated_conversations.py --model "o3" &
$PYTHON_BIN evaluate_simulated_conversations.py --model "o4-mini" &

wait

$PYTHON_BIN evaluate_simulated_conversations_pairwise.py --model gpt-4o &
$PYTHON_BIN evaluate_simulated_conversations_pairwise.py --model "gpt-4o-mini" &
$PYTHON_BIN evaluate_simulated_conversations_pairwise.py --model "gpt-5" &
$PYTHON_BIN evaluate_simulated_conversations_pairwise.py --model "gpt-5-mini" &
$PYTHON_BIN evaluate_simulated_conversations_pairwise.py --model "o3" &
$PYTHON_BIN evaluate_simulated_conversations_pairwise.py --model "o4-mini" &

wait
