"""CLI entrypoint — topic + domain_hint + max_retries + debug flag."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from config import settings
from graph.build_graph import run
from src.common.logging_setup import setup_logging

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Self-Evaluating Lesson Content Generator — Phase-1.5")
    parser.add_argument("topic", nargs="?", default="Introduction to RAG", help="Lesson topic")
    parser.add_argument("--domain-hint", default="", help="Optional user domain hint e.g. 'AI/ML'")
    parser.add_argument("--max-retries", type=int, default=settings.MAX_RETRIES)
    parser.add_argument("--debug-force-fail", action="store_true", help="Corrupt attempt 1 for Loom demo")
    parser.add_argument("--json-out", default="", help="Optional path to write full output JSON")
    args = parser.parse_args()

    result = run(topic=args.topic, domain_hint=args.domain_hint, max_retries=args.max_retries, debug_force_fail=args.debug_force_fail)

    # Always print summary
    print("\n=== RESULT ===")
    print(f"Topic: {result.get('topic')} | wiki: {result.get('wiki_page_title')} | grounding={result.get('grounding_status')}/{result.get('grounding_reason')} | final_status={result.get('final_status')} | overall_pass={result.get('overall_pass')}")
    if result.get("rejection_log"):
        print("\nRejection log:")
        for entry in result["rejection_log"]:
            print(f"  attempt {entry['attempt_num']} failed {[c['name'] for c in entry['failed_checks']]} | correction: {entry['correction_applied'][:120]}...")
    # print draft excerpt
    draft = result.get("draft_lesson", "")
    print("\n--- Draft (first 800 chars) ---")
    print(draft[:800])

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nWritten to {args.json_out}")

    # exit code 0 even on max_retries_exhausted — we always ship last draft
    sys.exit(0)

if __name__ == "__main__":
    main()
