#!/usr/bin/env python3
"""
Diagnostic script for inspecting Khmer Unicode strings, code points, and normalization.
Usage:
    python scripts/inspect_unicode.py --word "សាលា"
    python scripts/inspect_unicode.py --word "សន្តិភាព"
"""

import argparse
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_preparation.unicode_diagnostics import format_diagnostic_report

def main():
    parser = argparse.ArgumentParser(description="Inspect Khmer Unicode characters and normalization.")
    parser.add_argument("--word", required=True, help="Khmer word or text snippet to inspect.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("KHMER UNICODE DIAGNOSTIC REPORT")
    print("=" * 60)
    print(format_diagnostic_report(args.word))
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
