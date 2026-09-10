"""Local HTTP regression test with real OCR images and isolated test probabilities.

Usage: python tests/manual_e2e_smoke_test.py image1.png image2.png
The temporary synthetic probability projection tests wiring only. It is never
written to data/private and is not a prediction for the pictured games.
"""
import base64
import json
from pathlib import Path
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from mlb_decision_model import dashboard

def main(paths):
    with tempfile.TemporaryDirectory() as directory:
        projection = Path(directory) / "synthetic-test-only.json"
        with patch.object(dashboard, "MODEL_PROBABILITIES_PATH", projection):
            server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"
            def post(route, payload):
                request = urllib.request.Request(base+route, data=json.dumps(payload).encode(),
                    headers={"Content-Type":"application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.load(response)
            try:
                picks, predictions = [], []
                for path in paths:
                    result = post("/api/ocr-extract", {"image_data_url":"data:image/png;base64,"+base64.b64encode(Path(path).read_bytes()).decode()})
                    print(Path(path).name, len(result["drafts"]), "market rows", flush=True)
                    for draft in result["drafts"]:
                        if not draft["event_name"]:
                            continue
                        for selection in draft["selections"]:
                            row = dict(event_id=draft["event_name"],period=draft["period"],
                                market=draft["market"],selection=selection["label"],line=draft["market_line"])
                            picks.append({**row, "name":f'{row["event_id"]} {row["period"]} {row["market"]} {row["selection"]}',
                                "odds":selection["odds"]})
                            predictions.append({**row,"probability":0.4+0.02*(len(predictions)%10)})
                # No HTTP client-supplied probability can bypass missing model data.
                try:
                    post("/api/analyze", {"picks":[{**p,"probability":.99} for p in picks],"legs":2})
                    raise AssertionError("Client probability bypassed server projection")
                except urllib.error.HTTPError as error:
                    assert error.code == 409, error.code
                projection.write_text(json.dumps({"model_version":"SYNTHETIC_TEST_ONLY","predictions":predictions}),encoding="utf-8")
                result = post("/api/analyze", {"picks":picks,"legs":2,"top_n":5})
                assert result["combination_count"] >= 5, result
                assert len(result["survival_rank"]) == 5
                assert result["worst_survival"]["hit_probability"] <= result["survival_rank"][-1]["hit_probability"]
                print("PASS: HTTP OCR -> all candidate offers -> five ranked combinations; test projection removed")
            finally:
                server.shutdown()
                server.server_close()
                worker.join()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
