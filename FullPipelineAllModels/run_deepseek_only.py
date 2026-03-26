#!/usr/bin/env python3
"""
Simple wrapper to run the full pipeline for DeepSeek only.

This just imports and runs the main pipeline, which has been configured
to only evaluate DeepSeek (other models are commented out).
"""

from run_pipeline_all_models import main

if __name__ == "__main__":
    main()

