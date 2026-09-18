"""
Single-command entry point to reproduce this project's headline results.

    python evaluate.py                    # fast path (~1-2 min, no API calls)
    python evaluate.py --regenerate-agent  # also re-runs the LLM agent (slow, needs API key)

See src/evaluation/eval_harness.py for what this actually does.
"""
from src.evaluation.eval_harness import main

if __name__ == "__main__":
    main()
