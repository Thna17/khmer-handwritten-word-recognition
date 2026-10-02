#!/usr/bin/env python3
"""Local, crash-safe browser workflow for Khmer line transcription.

The application edits only transcription/review fields in ``line_metadata.csv``.
Source images, crop images, crop IDs, image paths, and bounding boxes are never
modified. Labels are stored exactly as submitted; Unicode normalization belongs
to the later verification/normalization stage.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image


STATUSES = ("untranscribed", "transcribed", "unclear", "verified", "rejected")
LEGACY_STATUS_MAP = {
    "": "untranscribed",
    "pending": "untranscribed",
    "confirmed": "verified",
    "corrected": "verified",
    "accepted": "verified",
    "auto_aligned": "transcribed",
}
FORBIDDEN_LABEL_MARKERS = (
    "[unclear]",
    "[illegible]",
    "[unknown]",
    "[rejected]",
    "<unclear>",
    "<illegible>",
)
EDITABLE_COLUMNS = ("label_raw", "review_status", "reviewer_notes", "label_confidence")


def natural_key(value: object) -> tuple[Any, ...]:
    import re

    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


def canonical_status(value: object) -> str:
    status = str(value or "").strip().lower()
    return LEGACY_STATUS_MAP.get(status, status if status in STATUSES else "untranscribed")


def validate_entry(label_raw: str, status: str, confidence: str) -> tuple[str, str, str]:
    """Validate without trimming or normalizing the label or reviewer notes."""
    if not isinstance(label_raw, str):
        raise ValueError("label_raw must be a Unicode string")
    if status not in STATUSES:
        raise ValueError(f"Unsupported review_status {status!r}")
    folded = label_raw.casefold()
    if any(marker in folded for marker in FORBIDDEN_LABEL_MARKERS):
        raise ValueError("Do not put readability markers in label_raw; use review_status='unclear'.")
    if status in {"untranscribed", "unclear"} and label_raw.strip():
        raise ValueError(f"review_status={status!r} requires a blank label_raw")
    if status in {"transcribed", "verified"} and not label_raw.strip():
        raise ValueError(f"review_status={status!r} requires a non-empty label_raw")

    confidence = str(confidence)
    if confidence.strip():
        try:
            number = float(confidence)
        except ValueError as error:
            raise ValueError("Confidence must be blank or a number between 0 and 1") from error
        if not 0.0 <= number <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        confidence = format(number, ".2f")
    else:
        confidence = ""
    return label_raw, status, confidence


class DatasetStore:
    """Read and atomically update line metadata across all writer folders."""

    def __init__(self, metadata_root: Path):
        self.metadata_root = metadata_root.resolve()
        self._lock = threading.RLock()
        self._assert_unique_ids()

    def metadata_paths(self) -> list[Path]:
        return sorted(self.metadata_root.glob("W*/line_metadata.csv"), key=lambda path: natural_key(path.parent.name))

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Metadata has no header: {path}")
            return list(reader.fieldnames), [dict(row) for row in reader]

    def _all_records(self) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for path in self.metadata_paths():
            _, file_records = self._read_csv(path)
            for row in file_records:
                item = dict(row)
                item["_metadata_path"] = str(path)
                item["_canonical_status"] = canonical_status(item.get("review_status"))
                records.append(item)
        return sorted(records, key=lambda row: natural_key(row.get("crop_id", "")))

    def _assert_unique_ids(self) -> None:
        counts = Counter(row.get("crop_id", "") for row in self._all_records())
        duplicates = sorted((crop_id for crop_id, count in counts.items() if not crop_id or count > 1), key=natural_key)
        if duplicates:
            raise ValueError(f"Empty or duplicate crop IDs prevent safe editing: {duplicates[:10]}")

    def writers(self) -> list[str]:
        return sorted({row.get("writer_id", "") for row in self._all_records()}, key=natural_key)

    def progress(self) -> list[dict[str, Any]]:
        grouped: dict[str, Counter[str]] = {}
        for row in self._all_records():
            grouped.setdefault(row.get("writer_id", ""), Counter())[row["_canonical_status"]] += 1
        result = []
        for writer_id in sorted(grouped, key=natural_key):
            counts = grouped[writer_id]
            result.append(
                {
                    "writer_id": writer_id,
                    **{status: counts[status] for status in STATUSES},
                    "usable": sum(counts[status] for status in STATUSES if status != "rejected"),
                    "completed": counts["transcribed"] + counts["unclear"] + counts["verified"],
                }
            )
        return result

    def list_items(self, writer_id: str = "all", status: str = "untranscribed") -> list[dict[str, str]]:
        if status not in {*STATUSES, "usable", "all"}:
            raise ValueError(f"Unsupported status filter {status!r}")
        items = []
        for row in self._all_records():
            canonical = row["_canonical_status"]
            if writer_id != "all" and row.get("writer_id") != writer_id:
                continue
            if status == "usable" and canonical == "rejected":
                continue
            if status not in {"usable", "all"} and canonical != status:
                continue
            items.append(
                {
                    "crop_id": row.get("crop_id", ""),
                    "writer_id": row.get("writer_id", ""),
                    "source_page": row.get("source_page", ""),
                    "review_status": canonical,
                }
            )
        return items

    def get_item(self, crop_id: str) -> dict[str, str]:
        row = next((record for record in self._all_records() if record.get("crop_id") == crop_id), None)
        if row is None:
            raise KeyError(crop_id)
        return {
            "crop_id": row.get("crop_id", ""),
            "writer_id": row.get("writer_id", ""),
            "source_page": row.get("source_page", ""),
            "label_raw": row.get("label_raw", ""),
            "review_status": row["_canonical_status"],
            "reviewer_notes": row.get("reviewer_notes", ""),
            "confidence": row.get("label_confidence", ""),
            "extraction_notes": row.get("notes", ""),
            "image_url": f"/api/image/{quote(crop_id, safe='')}",
        }

    def resolve_image(self, crop_id: str) -> Path:
        row = next((record for record in self._all_records() if record.get("crop_id") == crop_id), None)
        if row is None:
            raise KeyError(crop_id)
        metadata_path = Path(row["_metadata_path"])
        raw_path = metadata_path.parent / row.get("image_path", "")
        processed_dir = "lines_processed" if raw_path.parent.name == "lines_raw" else "crops_processed"
        processed_path = metadata_path.parent / processed_dir / raw_path.name
        image_path = processed_path if processed_path.is_file() else raw_path
        resolved = image_path.resolve()
        if not resolved.is_relative_to(self.metadata_root) or not resolved.is_file():
            raise FileNotFoundError(image_path)
        return resolved

    def save_entry(
        self,
        crop_id: str,
        *,
        label_raw: str,
        review_status: str,
        reviewer_notes: str,
        confidence: str,
    ) -> dict[str, str]:
        label_raw, review_status, confidence = validate_entry(label_raw, review_status, confidence)
        with self._lock:
            matches: list[tuple[Path, list[str], list[dict[str, str]], int]] = []
            for path in self.metadata_paths():
                fieldnames, rows = self._read_csv(path)
                for index, row in enumerate(rows):
                    if row.get("crop_id") == crop_id:
                        matches.append((path, fieldnames, rows, index))
            if len(matches) != 1:
                raise KeyError(f"Expected exactly one record for {crop_id!r}; found {len(matches)}")

            path, fieldnames, rows, index = matches[0]
            original_identity = {key: rows[index].get(key, "") for key in ("crop_id", "image_path", "source_page", "writer_id")}
            for column in EDITABLE_COLUMNS:
                if column not in fieldnames:
                    fieldnames.append(column)
            rows[index]["label_raw"] = label_raw
            rows[index]["review_status"] = review_status
            rows[index]["reviewer_notes"] = reviewer_notes
            rows[index]["label_confidence"] = confidence

            self._atomic_write(path, fieldnames, rows)
            written_identity = {key: rows[index].get(key, "") for key in original_identity}
            if written_identity != original_identity:  # defensive invariant
                raise RuntimeError("Immutable crop identity changed during save")
        return self.get_item(crop_id)

    @staticmethod
    def _atomic_write(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Khmer Line Transcription</title>
<style>
:root{--bg:#101416;--panel:#192024;--line:#344047;--text:#edf1ef;--muted:#9eaaa5;--accent:#f0a44b;--ok:#51b887;--bad:#e16b62}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,-apple-system,sans-serif}
main{max-width:1300px;margin:auto;padding:18px}.top{display:flex;gap:14px;align-items:end;flex-wrap:wrap}.top h1{margin:0 auto 0 0;font-size:20px}
label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px} select,input,textarea,button{font:inherit}
select,input[type=number],textarea{background:#0d1113;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:9px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:16px;margin-top:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:15px}
.image-frame{height:360px;background:white;border-radius:7px;overflow:auto;display:flex;align-items:center;padding:12px}.image-frame img{height:240px;max-width:none;display:block;margin:auto}
.identity{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0;color:var(--muted)}.identity b{color:var(--text)}
.field{margin-top:12px}.khmer{width:100%;min-height:100px;resize:vertical;font:28px/1.7 "Noto Sans Khmer","Khmer Sangam MN",sans-serif}
.notes{width:100%;min-height:70px;resize:vertical}.row{display:flex;gap:10px;align-items:end;flex-wrap:wrap}.row .grow{flex:1;min-width:180px}
.actions{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}button{border:0;border-radius:6px;padding:10px 14px;background:#39464d;color:white;cursor:pointer}button.primary{background:var(--ok);font-weight:700}button:disabled{opacity:.45;cursor:not-allowed}
.message{min-height:22px;margin-top:9px}.error{color:#ff8b82}.success{color:#6ed5a3}.help{color:var(--muted);font-size:12px}.extract{color:var(--accent);font-size:12px;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:6px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}.summary h2{font-size:15px;margin-top:0}.empty{padding:70px 20px;text-align:center;color:var(--muted)}
@media(max-width:900px){.layout{grid-template-columns:1fr}.image-frame{height:290px}.summary{order:2}}
</style></head><body><main>
<div class="top"><h1>Khmer line transcription</h1><div><label>Writer</label><select id="writer"></select></div><div><label>Status</label><select id="status"></select></div><div id="position" class="help"></div></div>
<div class="layout"><section class="card" id="editor"><div class="empty">Loading…</div></section><aside class="card summary"><h2>Progress by writer</h2><div id="summary"></div></aside></div>
</main><script>
const statuses=['untranscribed','transcribed','unclear','verified','rejected'];
let items=[],index=0,current=null,dirty=false;
const $=id=>document.getElementById(id);
function escapeHtml(v){return String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
async function request(url,options){const r=await fetch(url,options);const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);return data;}
async function bootstrap(){const data=await request('/api/bootstrap');$('writer').innerHTML='<option value="all">All writers</option>'+data.writers.map(w=>`<option>${escapeHtml(w)}</option>`).join('');$('status').innerHTML='<option value="usable">All usable</option>'+statuses.map(s=>`<option>${s}</option>`).join('')+'<option value="all">All including rejected</option>';const initial=new URLSearchParams(location.search);const requestedWriter=initial.get('writer_id')||'all';const requestedStatus=initial.get('status')||'untranscribed';$('writer').value=data.writers.includes(requestedWriter)?requestedWriter:'all';$('status').value=[...statuses,'usable','all'].includes(requestedStatus)?requestedStatus:'untranscribed';$('writer').onchange=()=>loadList(0);$('status').onchange=()=>loadList(0);await loadList(0);}
async function loadList(preferred=0,advanceFromId=null){const q=new URLSearchParams({writer_id:$('writer').value,status:$('status').value});const data=await request('/api/list?'+q);items=data.items;renderSummary(data.progress);if(advanceFromId){const stillHere=items.findIndex(x=>x.crop_id===advanceFromId);index=stillHere>=0?stillHere+1:preferred;}else{index=preferred;}index=Math.max(0,Math.min(index,items.length-1));if(items.length)await loadItem(index);else renderEmpty();}
async function loadItem(next){if(dirty&&!confirm('Discard unsaved changes?'))return;index=Math.max(0,Math.min(next,items.length-1));current=await request('/api/item/'+encodeURIComponent(items[index].crop_id));dirty=false;renderEditor();}
function renderEditor(){const s=current.review_status;$('position').textContent=`${index+1} / ${items.length}`;$('editor').innerHTML=`
<div class="image-frame"><img alt="Line crop ${escapeHtml(current.crop_id)}" src="${current.image_url}?t=${Date.now()}"></div>
<div class="identity"><span>crop_id: <b>${escapeHtml(current.crop_id)}</b></span><span>writer: <b>${escapeHtml(current.writer_id)}</b></span><span>page: <b>${escapeHtml(current.source_page)}</b></span></div>
${current.extraction_notes?`<div class="extract">Extraction note: ${escapeHtml(current.extraction_notes)}</div>`:''}
<div class="field"><label for="labelRaw">label_raw — enter Khmer only; never add [UNCLEAR] or similar markers</label><textarea id="labelRaw" class="khmer" spellcheck="false"></textarea></div>
<div class="row"><div class="grow"><label for="entryStatus">Review status</label><select id="entryStatus">${statuses.map(x=>`<option ${x===s?'selected':''}>${x}</option>`).join('')}</select></div><div><label for="confidence">Confidence (0–1)</label><input id="confidence" type="number" min="0" max="1" step="0.01"></div></div>
<div class="field"><label for="reviewerNotes">Reviewer notes</label><textarea id="reviewerNotes" class="notes"></textarea></div>
<div class="actions"><button id="prev">← Previous</button><button id="save">Save</button><button id="saveNext" class="primary">Save & next</button><button id="next">Next →</button></div>
<div class="help">Ctrl/⌘+Enter: save and next. “unclear” and “untranscribed” require an empty label. Every save is atomic.</div><div id="message" class="message"></div>`;
$('labelRaw').value=current.label_raw;$('reviewerNotes').value=current.reviewer_notes;$('confidence').value=current.confidence;
['labelRaw','entryStatus','reviewerNotes','confidence'].forEach(id=>$(id).addEventListener('input',()=>dirty=true));
$('entryStatus').onchange=()=>{dirty=true;if(['unclear','untranscribed'].includes($('entryStatus').value))$('labelRaw').value='';};
$('prev').onclick=()=>loadItem(index-1);$('next').onclick=()=>loadItem(index+1);$('save').onclick=()=>save(false);$('saveNext').onclick=()=>save(true);$('prev').disabled=index===0;$('next').disabled=index===items.length-1;$('labelRaw').focus();}
async function save(moveNext){const payload={crop_id:current.crop_id,label_raw:$('labelRaw').value,review_status:$('entryStatus').value,reviewer_notes:$('reviewerNotes').value,confidence:$('confidence').value};try{await request('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});dirty=false;if(moveNext){await loadList(index,current.crop_id);}else{current=await request('/api/item/'+encodeURIComponent(current.crop_id));renderEditor();$('message').textContent='Saved';$('message').className='message success';await refreshSummary();}}catch(e){$('message').textContent=e.message;$('message').className='message error';}}
async function refreshSummary(){const q=new URLSearchParams({writer_id:$('writer').value,status:$('status').value});const data=await request('/api/list?'+q);renderSummary(data.progress);}
function renderSummary(rows){$('summary').innerHTML='<table><thead><tr><th>Writer</th><th>Done</th><th>Usable</th><th>Unclear</th><th>Verified</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${escapeHtml(r.writer_id)}</td><td>${r.completed}</td><td>${r.usable}</td><td>${r.unclear}</td><td>${r.verified}</td></tr>`).join('')+'</tbody></table>';}
function renderEmpty(){current=null;dirty=false;$('position').textContent='0 / 0';$('editor').innerHTML='<div class="empty">No samples match these filters.</div>';}
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});window.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'&&current){e.preventDefault();save(true);}});bootstrap().catch(e=>{$('editor').innerHTML=`<div class="error">${escapeHtml(e.message)}</div>`;});
</script></body></html>"""


def make_handler(store: DatasetStore) -> type[BaseHTTPRequestHandler]:
    class TranscriptionHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, error: Exception, status: int = 400) -> None:
            self._json({"error": str(error)}, status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = PAGE_TEMPLATE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/api/bootstrap":
                    self._json({"writers": store.writers(), "statuses": STATUSES})
                elif parsed.path == "/api/list":
                    query = parse_qs(parsed.query)
                    writer_id = query.get("writer_id", ["all"])[0]
                    status = query.get("status", ["untranscribed"])[0]
                    self._json({"items": store.list_items(writer_id, status), "progress": store.progress()})
                elif parsed.path.startswith("/api/item/"):
                    self._json(store.get_item(unquote(parsed.path.removeprefix("/api/item/"))))
                elif parsed.path.startswith("/api/image/"):
                    self._serve_image(unquote(parsed.path.removeprefix("/api/image/")))
                else:
                    self.send_error(404)
            except KeyError as error:
                self._error(error, 404)
            except (ValueError, FileNotFoundError) as error:
                self._error(error, 400)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/save":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                item = store.save_entry(
                    str(payload.get("crop_id", "")),
                    label_raw=payload.get("label_raw", ""),
                    review_status=str(payload.get("review_status", "")),
                    reviewer_notes=payload.get("reviewer_notes", ""),
                    confidence=str(payload.get("confidence", "")),
                )
                self._json({"ok": True, "item": item})
            except KeyError as error:
                self._error(error, 404)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self._error(error, 400)

        def _serve_image(self, crop_id: str) -> None:
            path = store.resolve_image(crop_id)
            with Image.open(path) as opened:
                image = opened.convert("L")
                if image.height < 180:
                    scale = min(6.0, 180 / image.height)
                    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
            body = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return TranscriptionHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=Path("data/handwritten_external"))
    parser.add_argument("--host", default="127.0.0.1", help="Keep 127.0.0.1 unless you intentionally want network access")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = DatasetStore(args.metadata_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}/"
    print(f"Khmer line transcription is available at {url}")
    print("Every submitted entry is saved atomically. Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. All submitted entries were already saved.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
