#!/usr/bin/env python3
"""MethodLM desktop GUI -- a local quantum console. No cloud, no framework.

  python methodlm_gui.py            # opens http://127.0.0.1:8777 in your browser

Serves methodlm_gui.html (the quantum front-end) wrapped in an HTML skeleton, plus a
tiny JSON API the page talks to when it detects the backend:
  GET  /api/ping                 -> {ok:true}         (page flips to 'live' mode)
  POST /api/describe {path,...}  -> validate() result (columns, candidates, suggestions)
  POST /api/run {path,target}    -> runs the real investigation, returns the ledger text
The same HTML runs standalone as an Artifact (no backend -> simulation mode).
"""
import json, os, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "methodlm_gui.html")
PORT = 8777

SKELETON = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<style>html,body{margin:0;background:#0a0603}</style></head><body>{body}</body></html>")


def _read_html():
    with open(HTML, encoding="utf-8") as f:
        return SKELETON.replace("{body}", f.read())


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, _read_html(), "text/html; charset=utf-8")
        elif self.path == "/api/ping":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        try:
            data = self._body()
        except Exception as e:
            return self._send(400, json.dumps({"error": f"bad json: {e}"}))
        if self.path == "/api/describe":
            from methodlm_io import validate
            r = validate(data.get("path", ""), data.get("target"),
                         table=data.get("table"), query=data.get("query"))
            self._send(200, json.dumps(r))
        elif self.path == "/api/run":
            self._send(200, json.dumps(self._run(data)))
        elif self.path == "/api/race":
            self._send(200, json.dumps(self._run(data, race=True)))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _run(self, data, race=False):
        import methodlm as M
        from methodlm_io import load_any, featurize, format_report
        path, target = data.get("path", ""), data.get("target", "")
        raw, notes = load_any(path, table=data.get("table"), query=data.get("query"))
        tbl, rep = featurize(raw, target)
        report = format_report(notes, rep, target)
        cols = [c for c in tbl if c != target]
        q = (f"Investigate what actually drives {target} in this dataset "
             f"(columns: {', '.join(cols[:12])}). Do not trust raw correlations.")
        name = os.path.splitext(os.path.basename(path))[0]
        res = M.investigate(name, tbl, target, q, False, ingest_report=report)
        with open(os.path.join(HERE, f"ledger_{name}.txt"), encoding="utf-8") as f:
            out = {"ledger": f.read(), "verdict": res["verdict"],
                   "tested": res["nrun"], "prereg": res["pre"], "gate": res["gate"]}
        if race:
            out["vanilla"] = M.vanilla_answer(q)
        return out


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}"
    print(f"MethodLM quantum console -> {url}  (Ctrl+C to stop)")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
