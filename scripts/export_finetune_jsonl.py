#!/usr/bin/env python3
"""
Convert traffic-bank records into OpenAI fine-tuning JSONL.

Reads:  /data/traffic/<route>/*.jsonl  (records where layer == "llm")
Writes: finetune_<route>_<date>.jsonl  in {"messages":[system,user,assistant]} format
        ready for upload at https://platform.openai.com/finetune (base: gpt-4o-mini)

Usage:
  python scripts/export_finetune_jsonl.py --route orchestrator
  python scripts/export_finetune_jsonl.py --route orchestrator --strip-static --max 20000

--strip-static removes the ~3K-token instruction block from each system prompt,
keeping only the dynamic Context section. Use this for the "instructions baked
into weights" variant: train and serve WITHOUT the instruction block, then have
the gateway strip it at inference time too (must match!).
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

DYN_START_RE = re.compile(r"(Context:\s*1\.\s*Conversation History:)", re.IGNORECASE)
DYN_END_RE = re.compile(r"(Decision order\s*\(top to bottom\)\s*:)", re.IGNORECASE)


def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="orchestrator")
    ap.add_argument("--bank-dir", default=os.getenv("TRAFFIC_BANK_DIR", "/data/traffic"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max", type=int, default=50000)
    ap.add_argument("--strip-static", action="store_true",
                    help="drop the static instruction block, keep dynamic context only")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.bank_dir, args.route, "*.jsonl*")))
    if not files:
        sys.exit(f"no traffic files under {args.bank_dir}/{args.route}")

    out_path = args.out or f"finetune_{args.route}_{time.strftime('%Y%m%d')}.jsonl"
    seen, written, skipped = set(), 0, 0

    with open(out_path, "w", encoding="utf-8") as out:
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("layer") != "llm":
                        continue  # only LLM-labeled examples are ground truth
                    req = rec.get("request_body") or {}
                    assistant = rec.get("assistant_content") or ""
                    msgs = req.get("messages") or []
                    if not msgs or not assistant:
                        skipped += 1
                        continue
                    # validate assistant output is parseable JSON for JSON-output routes
                    if args.route in ("orchestrator", "agent-node"):
                        try:
                            json.loads(assistant)
                        except json.JSONDecodeError:
                            skipped += 1
                            continue

                    sys_txt, user_txt = "", ""
                    for m in msgs:
                        if m.get("role") == "system":
                            sys_txt = content_to_text(m.get("content"))
                        elif m.get("role") == "user":
                            user_txt = content_to_text(m.get("content"))

                    if args.strip_static:
                        sm = DYN_START_RE.search(sys_txt)
                        if sm:
                            em = DYN_END_RE.search(sys_txt, sm.end())
                            # keep only the dynamic middle; both static blocks
                            # (instructions + decision steps) get learned by the model
                            sys_txt = (sys_txt[sm.start(): em.start()].strip()
                                       if em else sys_txt[sm.start():])

                    # dedupe on (dynamic context + user input) so one hot query
                    # doesn't dominate the training mix
                    key = hashlib.sha256((sys_txt[-3000:] + "||" + user_txt).encode()).hexdigest()
                    if key in seen:
                        skipped += 1
                        continue
                    seen.add(key)

                    out.write(json.dumps({"messages": [
                        {"role": "system", "content": sys_txt},
                        {"role": "user", "content": user_txt},
                        {"role": "assistant", "content": assistant},
                    ]}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= args.max:
                        break
            if written >= args.max:
                break

    print(f"wrote {written} examples to {out_path} (skipped {skipped})")
    print("next: upload at https://platform.openai.com/finetune, base model gpt-4o-mini")
    if written < 500:
        print("NOTE: <500 examples -- let traffic accumulate; 1k-10k is the sweet spot")


if __name__ == "__main__":
    main()
