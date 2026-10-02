#!/usr/bin/env python3
"""
Interactive human review tool for auto-transcribed word crops.

Usage:
    python scripts/review_labels.py --writer-id W001 [--port 8765]

Opens a local web page where you see each PENDING crop next to its
AI-guessed transcription, and Confirm, Correct, or Reject it. Every
decision is written straight back to metadata.csv (review_status becomes
"confirmed" / "corrected" / "rejected"), so you can stop and resume at
any time without losing progress.

Why this exists: run_quality_control() (quality_control.py) only checks
STRUCTURAL validity -- file exists, box in bounds, non-blank, Unicode
decodes cleanly, etc. It has no way to know whether an auto-transcribed
label is actually the right word. That requires a human who can read the
original handwriting, which is what this tool is for. As of the last
count, 339 of 363 crops for W001 were still "pending" -- i.e. NONE of
their auto-generated labels have been confirmed by a person yet.

No new dependencies: uses only the Python standard library, so nothing
needs to be added to requirements.txt for a one-time review pass.
"""

import argparse
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_preparation.metadata import read_metadata_csv, write_metadata_csv  # noqa: E402

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Label Review</title>
<style>
  :root {
    --bg: #1c1b19; --panel: #262421; --ink: #f1ede4; --muted: #a39c8e;
    --rule: #3a3733; --accent: #d98a3d; --good: #4caf7d; --bad: #d9634a;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--ink); font-family: -apple-system, "Segoe UI", sans-serif;
    margin: 0; padding: 2rem 1.5rem; display: flex; flex-direction: column; align-items: center;
  }
  .wrap { width: 100%; max-width: 720px; }
  h1 { font-size: 1.15rem; font-weight: 600; margin: 0 0 0.25rem; }
  .progress { color: var(--muted); font-size: 0.88rem; margin-bottom: 1.25rem; }
  .card {
    background: var(--panel); border: 1px solid var(--rule); border-radius: 8px;
    padding: 1.5rem; display: flex; flex-direction: column; gap: 1.1rem;
  }
  .crop-frame {
    background: #fff; border-radius: 6px; padding: 1rem;
    display: flex; align-items: center; justify-content: center; min-height: 110px;
  }
  .crop-frame img { max-width: 100%; max-height: 320px; image-rendering: auto; }
  .meta { display: flex; gap: 1.25rem; font-size: 0.78rem; color: var(--muted); font-variant-numeric: tabular-nums; }
  .notes { font-size: 0.8rem; color: var(--accent); font-style: italic; }
  label { font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
  input[type=text] {
    width: 100%; font-size: 1.7rem; font-family: "Khmer Sangam MN", "Noto Sans Khmer", sans-serif;
    background: #14130f; color: var(--ink); border: 1.5px solid var(--rule); border-radius: 6px;
    padding: 0.6rem 0.8rem; margin-top: 0.35rem;
  }
  input[type=text]:focus { outline: none; border-color: var(--accent); }
  .actions { display: flex; gap: 0.7rem; }
  button {
    flex: 1; font-size: 0.95rem; font-weight: 600; padding: 0.7rem; border-radius: 6px;
    border: none; cursor: pointer; color: #fff;
  }
  button:active { transform: translateY(1px); }
  .btn-confirm { background: var(--good); }
  .btn-reject { background: var(--bad); }
  .hint { font-size: 0.72rem; color: var(--muted); text-align: center; }
  .done { text-align: center; padding: 3rem 1rem; }
  .done h2 { color: var(--good); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Label Review &mdash; W001</h1>
  <div class="progress" id="progress">Loading...</div>
  <div id="content"></div>
</div>
<script>
let current = null;

async function loadNext() {
  const res = await fetch('/api/pending');
  const data = await res.json();
  document.getElementById('progress').textContent = data.total_remaining + ' remaining';
  const content = document.getElementById('content');
  if (!data.item) {
    content.innerHTML = '<div class="card done"><h2>All caught up</h2><p>No pending labels left to review.</p></div>';
    current = null;
    return;
  }
  current = data.item;
  content.innerHTML = `
    <div class="card">
      <div class="crop-frame"><img src="/crop/${encodeURIComponent(current.image_path)}"></div>
      <div class="meta">
        <span>label confidence: ${current.label_confidence}</span>
        <span>segmentation confidence: ${current.segmentation_confidence}</span>
      </div>
      ${current.notes ? `<div class="notes">${current.notes}</div>` : ''}
      <div>
        <label for="labelInput">Transcription</label>
        <input type="text" id="labelInput" value="${current.label_normalized}" autocomplete="off">
      </div>
      <div class="actions">
        <button class="btn-confirm" id="confirmBtn">Confirm / Save &nbsp;(Enter)</button>
        <button class="btn-reject" id="rejectBtn">Reject &mdash; not usable</button>
      </div>
      <div class="hint">Edit the text above if the guess is wrong, then Confirm / Save.</div>
    </div>
  `;
  const input = document.getElementById('labelInput');
  input.focus();
  input.select();
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') submit('save'); });
  document.getElementById('confirmBtn').addEventListener('click', () => submit('save'));
  document.getElementById('rejectBtn').addEventListener('click', () => submit('reject'));
}

async function submit(kind) {
  if (!current) return;
  const input = document.getElementById('labelInput');
  const newLabel = input.value.trim();
  const action = kind === 'reject' ? 'reject' : (newLabel === current.label_normalized ? 'confirm' : 'correct');
  await fetch('/api/review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({crop_id: current.crop_id, action, corrected_label: newLabel})
  });
  loadNext();
}

loadNext();
</script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    writer_dir = None  # set in main() before serving

    def log_message(self, format, *args):
        pass  # keep the terminal quiet during review

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_page()
        elif parsed.path == "/api/pending":
            self._serve_pending()
        elif parsed.path.startswith("/crop/"):
            self._serve_crop(unquote(parsed.path[len("/crop/"):]))
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/review":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self._handle_review(body)
        else:
            self.send_error(404)

    def _serve_page(self):
        body = PAGE_TEMPLATE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    MIN_DISPLAY_HEIGHT = 280  # crops are often <40px tall -- unreadable at native size

    def _serve_crop(self, rel_path):
        full_path = os.path.abspath(os.path.join(self.writer_dir, rel_path))
        if not full_path.startswith(os.path.abspath(self.writer_dir) + os.sep) or not os.path.exists(full_path):
            self.send_error(404)
            return

        from io import BytesIO
        from PIL import Image

        image = Image.open(full_path)
        if image.height < self.MIN_DISPLAY_HEIGHT:
            scale = self.MIN_DISPLAY_HEIGHT / image.height
            image = image.resize((round(image.width * scale), self.MIN_DISPLAY_HEIGHT), Image.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()

        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_pending(self):
        csv_path = os.path.join(self.writer_dir, "metadata.csv")
        records = read_metadata_csv(csv_path)
        pending = sorted((r for r in records if r.get("review_status") == "pending"), key=lambda r: r["crop_id"])
        item = None
        if pending:
            r = pending[0]
            item = {
                "crop_id": r["crop_id"],
                "image_path": r["image_path"],
                "label_normalized": r["label_normalized"],
                "label_confidence": r["label_confidence"],
                "segmentation_confidence": r["segmentation_confidence"],
                "notes": r.get("notes", ""),
            }
        self._send_json({"total_remaining": len(pending), "item": item})

    def _handle_review(self, body):
        csv_path = os.path.join(self.writer_dir, "metadata.csv")
        records = read_metadata_csv(csv_path)
        crop_id = body.get("crop_id")
        action = body.get("action")
        corrected_label = (body.get("corrected_label") or "").strip()

        target = next((r for r in records if r["crop_id"] == crop_id), None)
        if target is None:
            self._send_json({"error": f"crop_id {crop_id} not found"}, status=404)
            return

        if action == "confirm":
            target["review_status"] = "confirmed"
        elif action == "correct":
            target["label_raw"] = corrected_label
            target["label_normalized"] = corrected_label
            target["review_status"] = "corrected"
        elif action == "reject":
            target["review_status"] = "rejected"
        else:
            self._send_json({"error": f"unknown action {action!r}"}, status=400)
            return

        write_metadata_csv(records, csv_path)
        self._send_json({"ok": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writer-id", required=True, help="e.g. W001")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args()

    writer_dir = os.path.join(BASE_DIR, "data", "handwritten_external", args.writer_id)
    if not os.path.isdir(writer_dir):
        parser.error(f"no such writer directory: {writer_dir}")
    if not os.path.exists(os.path.join(writer_dir, "metadata.csv")):
        parser.error(f"no metadata.csv found in {writer_dir}")

    ReviewHandler.writer_dir = writer_dir
    server = ThreadingHTTPServer(("localhost", args.port), ReviewHandler)
    url = f"http://localhost:{args.port}/"
    print(f"Reviewing labels for {args.writer_id} -- open {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Your progress has already been saved to metadata.csv after each decision.")


if __name__ == "__main__":
    main()
