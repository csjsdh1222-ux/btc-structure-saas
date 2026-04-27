"""Independent pre-signal detector.

This module scans Binance USDT markets to detect symbols that look like they are
about to break out (pre-pump state). It only prints alerts and writes CSV logs;
it never places orders.

Run:
    python -m app.detectors.pre_signal_detector
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://api.binance.com"
LOG_PATH = Path("logs/pre_signal_alerts.csv")

MIN_QUOTE_VOLUME_USDT = 10_000_000.0
TOP_SYMBOL_LIMIT = 100

ALERT_SCORE_THRESHOLD = 0.4
ALERT_COOLDOWN_SECONDS = 10 * 60
REALERT_PRICE_INCREASE = 0.02

LAST_ALERTS: Dict[str, Tuple[float, float]] = {}


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class PreSignalMetrics:
    symbol: str
    current_price: float
    rise_15m: float
    volume_ratio_5m: float
    swing_high_gap: float
    low_rebound_ratio: float
    score: float
    reasons: List[str]


@dataclass(frozen=True)
class DetectorConfig:
    mode: str
    rise_15m_threshold: float
    volume_ratio_threshold: float
    swing_lookback: int
    swing_imminent_tolerance: float
    low_zone_ratio_max: float


REAL_CONFIG = DetectorConfig(
    mode="real",
    rise_15m_threshold=0.005,
    volume_ratio_threshold=1.2,
    swing_lookback=20,
    swing_imminent_tolerance=0.004,
    low_zone_ratio_max=0.4,
)

TEST_CONFIG = DetectorConfig(
    mode="test",
    rise_15m_threshold=0.003,
    volume_ratio_threshold=1.05,
    swing_lookback=12,
    swing_imminent_tolerance=0.007,
    low_zone_ratio_max=0.5,
)


def _http_get(path: str, params: Optional[dict] = None) -> Optional[object]:
    query = urlencode(params or {})
    url = f"{BASE_URL}{path}" + (f"?{query}" if query else "")
    req = Request(url, headers={"User-Agent": "pre-signal-detector/1.0"})
    try:
        with urlopen(req, timeout=10) as res:
            return json.loads(res.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[WARN] API request failed: {url} ({exc})")
        return None


def fetch_top_usdt_symbols(limit: int = TOP_SYMBOL_LIMIT) -> List[str]:
    ticker = _http_get("/api/v3/ticker/24hr")
    exchange_info = _http_get("/api/v3/exchangeInfo")
    if not isinstance(ticker, list) or not isinstance(exchange_info, dict):
        return []

    tradable_symbols = {
        s.get("symbol")
        for s in exchange_info.get("symbols", [])
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
    }

    ranked: List[Tuple[str, float]] = []
    for row in ticker:
        symbol = row.get("symbol")
        if symbol not in tradable_symbols:
            continue
        try:
            quote_volume = float(row.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            continue

        if quote_volume < MIN_QUOTE_VOLUME_USDT:
            continue
        ranked.append((symbol, quote_volume))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in ranked[:limit]]


def fetch_klines(symbol: str, interval: str, limit: int) -> List[Candle]:
    payload = _http_get(
        "/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit}
    )
    if not isinstance(payload, list):
        return []

    candles: List[Candle] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            candles.append(
                Candle(
                    open_time_ms=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        except (TypeError, ValueError):
            continue
    return candles


def calculate_price_change_15m(candles_5m: Sequence[Candle]) -> float:
    if len(candles_5m) < 4 or candles_5m[-4].close <= 0:
        return 0.0
    return (candles_5m[-1].close - candles_5m[-4].close) / candles_5m[-4].close


def calculate_volume_ratio_5m(candles_5m: Sequence[Candle], lookback: int = 20) -> float:
    if len(candles_5m) < lookback + 1:
        return 0.0
    recent = candles_5m[-1].volume
    avg = sum(c.volume for c in candles_5m[-(lookback + 1) : -1]) / lookback
    if avg <= 0:
        return 0.0
    return recent / avg


def swing_high_imminent(
    candles_5m: Sequence[Candle], lookback: int, tolerance: float
) -> Tuple[bool, float]:
    if len(candles_5m) < lookback + 1:
        return False, 1.0

    current = candles_5m[-1].close
    prev_high = max(c.high for c in candles_5m[-(lookback + 1) : -1])
    if prev_high <= 0:
        return False, 1.0

    gap_ratio = (prev_high - current) / prev_high
    # "돌파 직전": swing high 아래에 있으면서 tolerance 이내.
    imminent = current < prev_high and 0 <= gap_ratio <= tolerance
    return imminent, max(0.0, gap_ratio)


def low_rebound_initial_state(
    candles_5m: Sequence[Candle], lookback: int, low_zone_ratio_max: float
) -> Tuple[bool, float]:
    if len(candles_5m) < lookback + 1:
        return False, 1.0

    window = candles_5m[-(lookback + 1) :]
    current = window[-1].close
    local_low = min(c.low for c in window)
    local_high = max(c.high for c in window)

    if local_high <= local_low:
        return False, 1.0

    rebound_ratio = (current - local_low) / (local_high - local_low)

    # "저점 대비 상승 초기": 저점 위로 올라왔지만 상단까지는 아직 멂.
    in_initial_zone = 0.05 <= rebound_ratio <= low_zone_ratio_max
    return in_initial_zone, rebound_ratio


def calculate_pre_signal_score(
    rise_ok: bool,
    volume_ok: bool,
    swing_imminent_ok: bool,
    initial_rebound_ok: bool,
) -> float:
    score = 0.0
    if rise_ok:
        score += 0.30
    if volume_ok:
        score += 0.25
    if swing_imminent_ok:
        score += 0.25
    if initial_rebound_ok:
        score += 0.20
    return score


def should_alert(symbol: str, current_price: float, score: float, now_ts: float) -> bool:
    if score < ALERT_SCORE_THRESHOLD:
        return False

    last = LAST_ALERTS.get(symbol)
    if last is None:
        LAST_ALERTS[symbol] = (now_ts, current_price)
        return True

    last_ts, last_price = last
    elapsed = now_ts - last_ts
    price_rise = ((current_price - last_price) / last_price) if last_price > 0 else 0.0

    if elapsed >= ALERT_COOLDOWN_SECONDS or price_rise >= REALERT_PRICE_INCREASE:
        LAST_ALERTS[symbol] = (now_ts, current_price)
        return True

    return False


def write_alert_csv(
    timestamp_iso: str,
    symbol: str,
    rise_15m: float,
    volume_ratio_5m: float,
    swing_high_gap: float,
    low_rebound_ratio: float,
    score: float,
    reasons: str,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if new_file:
            writer.writerow(
                [
                    "timestamp",
                    "symbol",
                    "rise_15m",
                    "volume_ratio_5m",
                    "swing_high_gap",
                    "low_rebound_ratio",
                    "score",
                    "reasons",
                ]
            )
        writer.writerow(
            [
                timestamp_iso,
                symbol,
                f"{rise_15m:.6f}",
                f"{volume_ratio_5m:.6f}",
                f"{swing_high_gap:.6f}",
                f"{low_rebound_ratio:.6f}",
                f"{score:.6f}",
                reasons,
            ]
        )


def _build_reasons(
    rise_ok: bool,
    volume_ok: bool,
    swing_imminent_ok: bool,
    initial_rebound_ok: bool,
) -> List[str]:
    reasons: List[str] = []
    if rise_ok:
        reasons.append("rise_15m")
    if volume_ok:
        reasons.append("volume_5m")
    if swing_imminent_ok:
        reasons.append("swing_imminent")
    if initial_rebound_ok:
        reasons.append("initial_rebound")
    return reasons


def scan_symbol(symbol: str, config: DetectorConfig) -> Tuple[Optional[PreSignalMetrics], Optional[str]]:
    candles_5m = fetch_klines(symbol, "5m", 40)
    if len(candles_5m) < 25:
        return None, "insufficient_candles"

    rise_15m = calculate_price_change_15m(candles_5m)
    volume_ratio = calculate_volume_ratio_5m(candles_5m, lookback=20)
    swing_imminent_ok, swing_gap = swing_high_imminent(
        candles_5m,
        lookback=config.swing_lookback,
        tolerance=config.swing_imminent_tolerance,
    )
    initial_rebound_ok, rebound_ratio = low_rebound_initial_state(
        candles_5m,
        lookback=config.swing_lookback,
        low_zone_ratio_max=config.low_zone_ratio_max,
    )

    rise_ok = rise_15m >= config.rise_15m_threshold
    volume_ok = volume_ratio >= config.volume_ratio_threshold

    score = calculate_pre_signal_score(
        rise_ok=rise_ok,
        volume_ok=volume_ok,
        swing_imminent_ok=swing_imminent_ok,
        initial_rebound_ok=initial_rebound_ok,
    )

    if score < ALERT_SCORE_THRESHOLD:
        return None, (
            "score_below_threshold("
            f"score={score:.2f}, rise_ok={rise_ok}, volume_ok={volume_ok}, "
            f"swing_imminent_ok={swing_imminent_ok}, initial_rebound_ok={initial_rebound_ok}"
            ")"
        )

    reasons = _build_reasons(rise_ok, volume_ok, swing_imminent_ok, initial_rebound_ok)
    return PreSignalMetrics(
        symbol=symbol,
        current_price=candles_5m[-1].close,
        rise_15m=rise_15m,
        volume_ratio_5m=volume_ratio,
        swing_high_gap=swing_gap,
        low_rebound_ratio=rebound_ratio,
        score=score,
        reasons=reasons,
    ), None


def _print_alert(metric: PreSignalMetrics) -> None:
    reason = " + ".join(metric.reasons) if metric.reasons else "n/a"
    print("[PRE-SIGNAL 감지]")
    print(f"symbol: {metric.symbol}")
    print(f"rise_15m: {metric.rise_15m * 100:+.2f}%")
    print(f"volume_ratio_5m: {metric.volume_ratio_5m:.2f}x")
    print(f"swing_high_gap: {metric.swing_high_gap * 100:.3f}%")
    print(f"low_rebound_ratio: {metric.low_rebound_ratio:.3f}")
    print(f"score: {metric.score:.2f}")
    print(f"reason: {reason}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-signal detector.")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run detector in relaxed test mode thresholds.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print why each symbol was filtered out.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run repeated scans at a fixed interval until interrupted.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=120,
        help="Interval seconds between scans when --watch is enabled (default: 120).",
    )
    return parser.parse_args()


def _run_single_scan(args: argparse.Namespace, config: DetectorConfig) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    symbols = fetch_top_usdt_symbols(TOP_SYMBOL_LIMIT)
    if not symbols:
        print("[WARN] 스캔 대상 심볼을 가져오지 못했습니다.")
        return

    print(f"[INFO] scanning {len(symbols)} symbols | mode={config.mode}")

    alerts = 0
    filtered_counts: Dict[str, int] = {}
    now_ts = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()

    for symbol in symbols:
        if symbol == "BTCUSDT":
            continue
        try:
            metric, reject_reason = scan_symbol(symbol, config)
        except Exception as exc:
            print(f"[WARN] symbol scan failed: {symbol} ({exc})")
            continue

        if metric is None:
            if args.debug:
                reason = reject_reason or "filtered"
                print(f"[DEBUG] {symbol} filtered: {reason}")
            base_reason = (reject_reason or "filtered").split("(", 1)[0]
            filtered_counts[base_reason] = filtered_counts.get(base_reason, 0) + 1
            continue

        if should_alert(symbol, metric.current_price, metric.score, now_ts):
            _print_alert(metric)
            write_alert_csv(
                timestamp_iso=now_iso,
                symbol=metric.symbol,
                rise_15m=metric.rise_15m,
                volume_ratio_5m=metric.volume_ratio_5m,
                swing_high_gap=metric.swing_high_gap,
                low_rebound_ratio=metric.low_rebound_ratio,
                score=metric.score,
                reasons=" + ".join(metric.reasons),
            )
            alerts += 1

    elapsed = time.time() - start
    filtered_text = ", ".join(f"{k}: {v}" for k, v in sorted(filtered_counts.items()))
    print(
        f"[INFO] scan complete | alerts={alerts} | filtered={{{filtered_text}}} | elapsed={elapsed:.1f}s"
    )


def main() -> None:
    args = _parse_args()
    config = TEST_CONFIG if args.test_mode else REAL_CONFIG

    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be a positive integer.")

    if not args.watch:
        _run_single_scan(args, config)
        return

    print(
        f"[INFO] watch mode enabled | interval={args.interval_seconds}s "
        "(press Ctrl+C to stop safely)"
    )
    scan_count = 0
    try:
        while True:
            scan_count += 1
            print(f"[INFO] watch scan #{scan_count} started")
            _run_single_scan(args, config)
            print(f"[INFO] next scan in {args.interval_seconds}s")
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Exiting watch mode safely.")


if __name__ == "__main__":
    main()
