"""Evaluate pump detector alerts 30 minutes after signal time.

This module reads logs/pump_alerts.csv, evaluates only PENDING alerts that are at
least 30 minutes old, fetches Binance kline data for the 30-minute window, then
writes results to logs/pump_alert_results.csv.

Run:
    python -m app.detectors.pump_result_evaluator
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://api.binance.com"
ALERT_LOG_PATH = Path("logs/pump_alerts.csv")
RESULT_LOG_PATH = Path("logs/pump_alert_results.csv")
EVALUATION_DELAY = timedelta(minutes=30)


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    high: float
    low: float
    close: float


def _http_get(path: str, params: Optional[dict] = None) -> Optional[object]:
    query = urlencode(params or {})
    url = f"{BASE_URL}{path}" + (f"?{query}" if query else "")
    req = Request(url, headers={"User-Agent": "pump-result-evaluator/1.0"})
    try:
        with urlopen(req, timeout=10) as res:
            return json.loads(res.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[WARN] API request failed: {url} ({exc})")
        return None


def fetch_klines_window(symbol: str, start_ms: int, end_ms: int) -> List[Candle]:
    payload = _http_get(
        "/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        },
    )

    if not isinstance(payload, list) or not payload:
        payload = _http_get(
            "/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": "5m",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )

    if not isinstance(payload, list):
        return []

    candles: List[Candle] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 5:
            continue
        try:
            candles.append(
                Candle(
                    open_time_ms=int(row[0]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                )
            )
        except (TypeError, ValueError):
            continue
    return candles


def _parse_timestamp(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _classification(max_return_30m: float, max_drawdown_30m: float) -> str:
    if max_return_30m >= 0.02:
        return "SUCCESS"
    if max_drawdown_30m <= -0.015:
        return "FAIL"
    return "NEUTRAL"


def _read_alert_rows() -> List[Dict[str, str]]:
    if not ALERT_LOG_PATH.exists():
        print(f"[INFO] alert log not found: {ALERT_LOG_PATH}")
        return []
    with ALERT_LOG_PATH.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _write_alert_rows(rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_LOG_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_results(rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return

    RESULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULT_LOG_PATH.exists()
    fieldnames = [
        "evaluated_at",
        "alert_id",
        "timestamp",
        "symbol",
        "entry_reference_price",
        "price_after_30m",
        "max_high_30m",
        "min_low_30m",
        "max_return_30m",
        "max_drawdown_30m",
        "result",
    ]

    with RESULT_LOG_PATH.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def evaluate_pending_alerts() -> None:
    now = datetime.now(timezone.utc)
    alert_rows = _read_alert_rows()
    if not alert_rows:
        return

    fieldnames = list(alert_rows[0].keys())
    if "status" not in fieldnames:
        print("[WARN] status column missing in pump_alerts.csv")
        return

    results_to_append: List[Dict[str, str]] = []
    evaluated_count = 0

    for row in alert_rows:
        status = (row.get("status") or "").strip().upper()
        if status != "PENDING":
            continue

        ts_raw = (row.get("timestamp") or "").strip()
        symbol = (row.get("symbol") or "").strip()
        alert_id = (row.get("alert_id") or "").strip()

        ts = _parse_timestamp(ts_raw)
        if ts is None or not symbol:
            row["status"] = "INVALID"
            continue

        if now - ts < EVALUATION_DELAY:
            continue

        try:
            entry_price = float((row.get("entry_reference_price") or "0").strip())
        except ValueError:
            row["status"] = "INVALID"
            continue

        if entry_price <= 0:
            row["status"] = "INVALID"
            continue

        start_ms = int(ts.timestamp() * 1000)
        end_ms = int((ts + EVALUATION_DELAY).timestamp() * 1000)
        candles = fetch_klines_window(symbol, start_ms, end_ms)
        if not candles:
            row["status"] = "DATA_UNAVAILABLE"
            continue

        max_high = max(c.high for c in candles)
        min_low = min(c.low for c in candles)
        price_after_30m = candles[-1].close

        max_return = (max_high - entry_price) / entry_price
        max_drawdown = (min_low - entry_price) / entry_price
        result = _classification(max_return, max_drawdown)

        results_to_append.append(
            {
                "evaluated_at": now.isoformat(),
                "alert_id": alert_id,
                "timestamp": ts_raw,
                "symbol": symbol,
                "entry_reference_price": f"{entry_price:.8f}",
                "price_after_30m": f"{price_after_30m:.8f}",
                "max_high_30m": f"{max_high:.8f}",
                "min_low_30m": f"{min_low:.8f}",
                "max_return_30m": f"{max_return:.6f}",
                "max_drawdown_30m": f"{max_drawdown:.6f}",
                "result": result,
            }
        )
        row["status"] = "EVALUATED"
        evaluated_count += 1

    _append_results(results_to_append)
    _write_alert_rows(alert_rows, fieldnames)
    print(f"[INFO] evaluation complete | evaluated={evaluated_count} | results_written={len(results_to_append)}")


def main() -> None:
    evaluate_pending_alerts()


if __name__ == "__main__":
    main()
