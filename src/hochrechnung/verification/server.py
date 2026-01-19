"""
Simple HTTP server for verification UI.

Serves the static frontend and handles API requests for saving corrections.
"""

import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from hochrechnung.utils.logging import get_logger
from hochrechnung.verification.persistence import save_verified_counters

log = get_logger(__name__)


class VerificationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for verification UI."""

    # Class variables set by server
    verification_data_path: Path | None = None
    mbtiles_path: Path | None = None
    data_root: Path | None = None
    year: int | None = None
    counters_df: Any = None  # pandas DataFrame

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
        """Serve a static file."""
        static_dir = Path(__file__).parent / "static"
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
            # Client already disconnected, ignore
            pass

    def _serve_tile(self) -> None:
        """Serve vector tiles from MBTiles."""
        import sqlite3

        if not self.mbtiles_path or not self.mbtiles_path.exists():
            self._safe_send_error(500, "MBTiles not found")
            return

        # Parse tile coordinates from path: /tiles/{z}/{x}/{y}.pbf
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

        # Query MBTiles database
        # MBTiles uses TMS scheme (flip Y coordinate)
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
            # Client cancelled request (common during fast map panning)
            pass
        except Exception as e:
            log.error("Failed to serve tile", error=str(e))
            self._safe_send_error(500, "Tile serving failed")

    def _save_corrections(self) -> None:
        """Save counter corrections."""
        if self.data_root is None or self.year is None or self.counters_df is None:
            self.send_error(500, "Server not properly configured")
            return

        # Read request body
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            changes = data.get("changes", [])

            if not changes:
                self.send_error(400, "No changes provided")
                return

            # Update counters DataFrame

            for change in changes:
                counter_id = change["counter_id"]
                mask = self.counters_df["counter_id"] == counter_id

                if mask.any():
                    self.counters_df.loc[mask, "base_id"] = change["base_id"]

                    if change.get("count") is not None:
                        self.counters_df.loc[mask, "count"] = change["count"]

                    self.counters_df.loc[mask, "verification_status"] = "verified"
                    self.counters_df.loc[mask, "verified_at"] = datetime.now()
                    self.counters_df.loc[mask, "verification_metadata"] = change.get(
                        "metadata", ""
                    )
                    # Handle discard flag (defaults to False if not provided)
                    self.counters_df.loc[mask, "is_discarded"] = change.get(
                        "is_discarded", False
                    )

            # Save to CSV
            saved_path = save_verified_counters(
                self.counters_df, self.data_root, self.year
            )

            log.info(
                "Saved corrections",
                n_changes=len(changes),
                path=str(saved_path),
            )

            # Send success response
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
        # format and args are unused - we use our own logger
        _ = format, args
        log.debug("HTTP request", method=self.command, path=self.path)


def start_verification_server(
    verification_data_path: Path,
    mbtiles_path: Path,
    data_root: Path,
    year: int,
    counters_df: Any,
    port: int = 8000,
) -> None:
    """
    Start verification HTTP server.

    Args:
        verification_data_path: Path to verification JSON.
        mbtiles_path: Path to MBTiles file.
        data_root: Root data directory.
        year: Campaign year.
        counters_df: DataFrame with counter data.
        port: HTTP port.
    """
    # Set class variables for handler
    VerificationHandler.verification_data_path = verification_data_path
    VerificationHandler.mbtiles_path = mbtiles_path
    VerificationHandler.data_root = data_root
    VerificationHandler.year = year
    VerificationHandler.counters_df = counters_df

    server = HTTPServer(("localhost", port), VerificationHandler)

    log.info(
        "Starting verification server",
        url=f"http://localhost:{port}",
        year=year,
    )

    print(f"\n🌐 Verification UI: http://localhost:{port}")
    print(f"📊 Year: {year}")
    print(f"📁 Data: {verification_data_path}")
    print("\nPress Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down verification server")
        server.shutdown()
