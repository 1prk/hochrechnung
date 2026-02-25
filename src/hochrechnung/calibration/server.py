"""
HTTP server for calibration station verification UI.

Similar to verification/server.py but adapted for calibration stations.
"""

import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from hochrechnung.calibration.loader import save_verified_calibration_counters
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


class CalibrationVerificationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for calibration verification UI."""

    # Class variables set by server
    verification_data_path: Path | None = None
    mbtiles_path: Path | None = None
    data_root: Path | None = None
    year: int | None = None
    region: str | None = None
    stations_df: Any = None  # pandas DataFrame
    dtv_column: str = "dtv"
    id_column: str = "id"

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/" or self.path == "/index.html":
            self._serve_file("index.html", "text/html")

        elif self.path == "/app.js":
            self._serve_file("app.js", "application/javascript")

        elif self.path == "/api/verification-data":
            self._serve_verification_data()

        elif self.path.startswith("/tiles/"):
            self._serve_tile()

        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/api/save-corrections":
            self._save_corrections()
        else:
            self.send_error(404, "Not Found")

    def _serve_file(self, filename: str, content_type: str) -> None:
        """Serve a static file from verification module's static dir."""
        # Use the same static files from verification module
        static_dir = Path(__file__).parent.parent / "verification" / "static"
        file_path = static_dir / filename

        if not file_path.exists():
            self.send_error(404, "File not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()

        with file_path.open("rb") as f:
            self.wfile.write(f.read())

    def _serve_verification_data(self) -> None:
        """Serve verification data JSON."""
        if not self.verification_data_path or not self.verification_data_path.exists():
            self.send_error(500, "Verification data not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        with self.verification_data_path.open() as f:
            data = json.load(f)
            self.wfile.write(json.dumps(data).encode())

    def _safe_send_error(self, code: int, message: str) -> None:
        """Send error response, ignoring connection errors."""
        try:
            self.send_error(code, message)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _serve_tile(self) -> None:
        """Serve vector tiles from MBTiles."""
        import sqlite3

        if not self.mbtiles_path or not self.mbtiles_path.exists():
            self._safe_send_error(500, "MBTiles not found")
            return

        parts = self.path.split("/")
        if len(parts) != 5:
            self._safe_send_error(400, "Invalid tile path")
            return

        try:
            z = int(parts[2])
            x = int(parts[3])
            y_pbf = parts[4]
            y = int(y_pbf.replace(".pbf", ""))
        except ValueError:
            self._safe_send_error(400, "Invalid tile coordinates")
            return

        tms_y = (2**z - 1) - y

        try:
            conn = sqlite3.connect(self.mbtiles_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?",
                (z, x, tms_y),
            )

            row = cursor.fetchone()
            conn.close()

            if row is None:
                self._safe_send_error(404, "Tile not found")
                return

            tile_data = row[0]

            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            self.wfile.write(tile_data)

        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            log.error("Failed to serve tile", error=str(e))
            self._safe_send_error(500, "Tile serving failed")

    def _save_corrections(self) -> None:
        """Save calibration station corrections."""
        if (
            self.data_root is None
            or self.year is None
            or self.region is None
            or self.stations_df is None
        ):
            self.send_error(500, "Server not properly configured")
            return

        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            changes = data.get("changes", [])

            if not changes:
                self.send_error(400, "No changes provided")
                return

            # Update stations DataFrame
            for change in changes:
                counter_id = str(change["counter_id"])
                mask = self.stations_df[self.id_column].astype(str) == counter_id

                if mask.any():
                    # Update verification status
                    self.stations_df.loc[mask, "verification_status"] = "verified"
                    self.stations_df.loc[mask, "verified_at"] = datetime.now().isoformat()
                    self.stations_df.loc[mask, "verification_metadata"] = change.get(
                        "metadata", ""
                    )
                    self.stations_df.loc[mask, "is_discarded"] = change.get(
                        "is_discarded", False
                    )

                    # Update DTV if provided (for corrections)
                    if change.get("dtv") is not None:
                        self.stations_df.loc[mask, self.dtv_column] = change["dtv"]

            # Save to verified CSV
            saved_path = save_verified_calibration_counters(
                self.stations_df,
                self.data_root,
                self.region,
                self.year,
            )

            log.info(
                "Saved calibration corrections",
                n_changes=len(changes),
                path=str(saved_path),
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            response = {"success": True, "n_changes": len(changes)}
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            log.error("Failed to save corrections", error=str(e))
            self.send_error(500, f"Save failed: {e}")

    def log_message(self, format: str, *args: Any) -> None:
        """Override to use our logger."""
        _ = format, args
        log.debug("HTTP request", method=self.command, path=self.path)


def start_calibration_verification_server(
    verification_data_path: Path,
    mbtiles_path: Path | None,
    data_root: Path,
    year: int,
    region: str,
    stations_df: Any,
    dtv_column: str = "dtv",
    id_column: str = "id",
    port: int = 8000,
) -> None:
    """
    Start calibration verification HTTP server.

    Args:
        verification_data_path: Path to verification JSON.
        mbtiles_path: Path to MBTiles file (optional).
        data_root: Root data directory.
        year: Campaign year.
        region: Region name.
        stations_df: DataFrame with calibration station data.
        dtv_column: Name of DTV column.
        id_column: Name of ID column.
        port: HTTP port.
    """
    CalibrationVerificationHandler.verification_data_path = verification_data_path
    CalibrationVerificationHandler.mbtiles_path = mbtiles_path
    CalibrationVerificationHandler.data_root = data_root
    CalibrationVerificationHandler.year = year
    CalibrationVerificationHandler.region = region
    CalibrationVerificationHandler.stations_df = stations_df
    CalibrationVerificationHandler.dtv_column = dtv_column
    CalibrationVerificationHandler.id_column = id_column

    server = HTTPServer(("localhost", port), CalibrationVerificationHandler)

    log.info(
        "Starting calibration verification server",
        url=f"http://localhost:{port}",
        year=year,
        region=region,
    )

    print(f"\n🎯 Calibration Verification UI: http://localhost:{port}")
    print(f"📊 Year: {year}")
    print(f"🗺️  Region: {region}")
    print(f"📁 Data: {verification_data_path}")
    print("\nVerify calibration stations, then press Ctrl+C to continue with calibration\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down calibration verification server")
        server.shutdown()
