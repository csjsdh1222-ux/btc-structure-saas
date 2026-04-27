"""Independent pump-entry candidate detector.

This module scans Binance USDT markets and prints radar-style alerts for symbols
that satisfy early pump candidate conditions. It does NOT place orders and does
NOT connect to existing trading strategy/engine modules.

Run:
    python -m app.detectors.pump_detector
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
LOG_PATH = Path("logs/pump_alerts.csv")

MIN_QUOTE_VOLUME_USDT = 10_000_000.0
TOP_SYMBOL_LIMIT = 100

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
class PumpMetrics:
    symbol: str
    current_price: float
    price_change_5m: float
    volume_ratio_5m: float
    swing_breakout: bool
    fvg_like: bool
    rebound_like: bool
    btc_condition: str
    chasing_risk: bool
    volume_drop_risk: bool
    score: float
    reasons: List[str]


@dataclass(frozen=True)
class DetectorConfig:
    mode: str
    price_change_threshold: float
    volume_ratio_threshold: float
    score_threshold: float
    swing_high_tolerance: float


REAL_CONFIG = DetectorConfig(
    mode="real",
    price_change_threshold=0.03,
    volume_ratio_threshold=2.0,
    score_threshold=0.7,
    swing_high_tolerance=0.003,
)

TEST_CONFIG = DetectorConfig(
    mode="test",
    price_change_threshold=0.015,
    volume_ratio_threshold=1.3,
    score_threshold=0.5,
    swing_high_tolerance=0.007,
)


def _http_get(path: str, params: Optional[dict] = None) -> Optional[object]:
    query = urlencode(params or {})
    url = f"{BASE_URL}{path}" + (f"?{query}" if query else "")
    req = Request(url, headers={"User-Agent": "pump-detector/1.0"})
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

        # Exclude low liquidity symbols from radar candidates.
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


def calculate_price_change_5m(candles: Sequence[Candle]) -> float:
    if len(candles) < 2 or candles[-2].close <= 0:
        return 0.0
    return (candles[-1].close - candles[-2].close) / candles[-2].close


def calculate_volume_ratio(candles: Sequence[Candle], lookback: int = 20) -> float:
    if len(candles) < lookback + 1:
        return 0.0
    recent = candles[-1].volume
    avg = sum(c.volume for c in candles[-(lookback + 1) : -1]) / lookback
    if avg <= 0:
        return 0.0
    return recent / avg


def detect_swing_high_breakout(
    candles: Sequence[Candle], lookback: int = 20, tolerance: float = 0.003
) -> bool:
    if len(candles) < lookback + 1:
        return False
    current = candles[-1].close
    prev_high = max(c.high for c in candles[-(lookback + 1) : -1])
    if prev_high <= 0:
        return False
    # "근접" 허용: 직전 swing high 대비 tolerance 이내.
    return current >= prev_high or (prev_high - current) / prev_high <= tolerance


def detect_simple_fvg(candles: Sequence[Candle]) -> bool:
    if len(candles) < 3:
        return False
    a, _, c = candles[-3], candles[-2], candles[-1]
    # 3-candle bullish imbalance proxy:
    # candle[-1] low above candle[-3] high implies a gap-like inefficiency zone.
    return c.low > a.high


def detect_simple_rebound(candles: Sequence[Candle]) -> bool:
    if len(candles) < 4:
        return False
    c1, c2, c3, c4 = candles[-4], candles[-3], candles[-2], candles[-1]
    # Simple structure rebound: two-step decline then bullish reversal close.
    decline = c1.close > c2.close > c3.close
    bullish_reversal = c4.close > c4.open and c4.close > c3.high
    return decline and bullish_reversal


def evaluate_btc_condition() -> str:
    btc_5m = fetch_klines("BTCUSDT", "5m", 4)
    btc_15m = fetch_klines("BTCUSDT", "15m", 3)
    if len(btc_5m) < 4 or len(btc_15m) < 3:
        return "unknown"

    # 급락 필터 기준:
    # - 5m 기준 최근 3개 캔들(약 15분) 누적 수익률 <= -1.5% -> bearish_crash
    p0 = btc_5m[-4].close
    p1 = btc_5m[-1].close
    crash_15m = (p1 - p0) / p0 if p0 > 0 else 0.0
    if crash_15m <= -0.015:
        return "bearish_crash"

    # 시장 분류:
    # - 15m 최근 2개 캔들 수익률 절대값 < 0.3% -> sideways
    # - 그 외 양수면 bullish, 음수면 bearish
    b0 = btc_15m[-3].close
    b1 = btc_15m[-1].close
    move = (b1 - b0) / b0 if b0 > 0 else 0.0
    if abs(move) < 0.003:
        return "sideways"
    return "bullish" if move > 0 else "bearish"


def is_chasing_risk(candles: Sequence[Candle]) -> bool:
    if len(candles) < 4:
        return True

    # 15분 상승률 8% 이상.
    prior = candles[-4].close
    rise_15m = (candles[-1].close - prior) / prior if prior > 0 else 0.0
    if rise_15m >= 0.08:
        return True

    # 최근 3개 양봉 + 거래량 감소.
    last3 = candles[-3:]
    all_green = all(c.close > c.open for c in last3)
    vol_desc = last3[0].volume > last3[1].volume > last3[2].volume
    if all_green and vol_desc:
        return True

    # 긴 윗꼬리: wick >= 2 * body.
    c = candles[-1]
    body = abs(c.close - c.open)
    upper_wick = c.high - max(c.open, c.close)
    if body > 0 and upper_wick >= 2 * body:
        return True

    return False


def is_volume_drop_risk(one_min_candles: Sequence[Candle]) -> bool:
    if len(one_min_candles) < 7:
        return False

    current = one_min_candles[-1].volume
    avg_prev5 = sum(c.volume for c in one_min_candles[-6:-1]) / 5
    if avg_prev5 <= 0:
        return False
    return current <= 0.5 * avg_prev5


def calculate_pump_score(
    price_ok: bool,
    volume_ok: bool,
    swing_ok: bool,
    fvg_ok: bool,
    rebound_ok: bool,
    btc_condition: str,
) -> float:
    score = 0.0
    if price_ok:
        score += 0.25
    if volume_ok:
        score += 0.25
    if swing_ok:
        score += 0.20
    if fvg_ok:
        score += 0.10
    if rebound_ok:
        score += 0.10

    if btc_condition == "bullish":
        score += 0.10
    elif btc_condition == "sideways":
        score -= 0.10

    return min(1.0, max(0.0, score))


def should_alert(symbol: str, price: float, score: float, now_ts: float, score_threshold: float) -> bool:
    if score < score_threshold:
        return False

    prev = LAST_ALERTS.get(symbol)
    if prev is None:
        LAST_ALERTS[symbol] = (now_ts, price)
        return True

    prev_ts, prev_price = prev
    elapsed = now_ts - prev_ts
    if elapsed >= ALERT_COOLDOWN_SECONDS:
        LAST_ALERTS[symbol] = (now_ts, price)
        return True

    # 10분 내 재알림은 기본 금지, 단 +2% 추가 상승이면 허용.
    if prev_price > 0 and (price - prev_price) / prev_price >= REALERT_PRICE_INCREASE:
        LAST_ALERTS[symbol] = (now_ts, price)
        return True
    return False


def write_alert_csv(
    timestamp_iso: str,
    symbol: str,
    price_change_5m: float,
    volume_ratio: float,
    score: float,
    btc_condition: str,
    alert_reason: str,
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
                    "price_change_5m",
                    "volume_ratio",
                    "score",
                    "btc_condition",
                    "alert_reason",
                ]
            )
        writer.writerow(
            [
                timestamp_iso,
                symbol,
                f"{price_change_5m:.6f}",
                f"{volume_ratio:.6f}",
                f"{score:.6f}",
                btc_condition,
                alert_reason,
            ]
        )


def _build_reasons(price_ok: bool, volume_ok: bool, swing_ok: bool, fvg_ok: bool, rebound_ok: bool) -> List[str]:
    reasons: List[str] = []
    if price_ok:
        reasons.append("price")
    if volume_ok:
        reasons.append("volume")
    if swing_ok:
        reasons.append("breakout")
    if fvg_ok:
        reasons.append("fvg_like")
    if rebound_ok:
        reasons.append("rebound")
    return reasons


def scan_symbol(symbol: str, btc_condition: str, config: DetectorConfig) -> Tuple[Optional[PumpMetrics], Optional[str]]:
    candles_5m = fetch_klines(symbol, "5m", 40)
    candles_1m = fetch_klines(symbol, "1m", 12)
    if len(candles_5m) < 25 or len(candles_1m) < 7:
        return None, "insufficient_candles"

    price_change = calculate_price_change_5m(candles_5m)
    volume_ratio = calculate_volume_ratio(candles_5m, lookback=20)
    swing_ok = detect_swing_high_breakout(
        candles_5m,
        lookback=20,
        tolerance=config.swing_high_tolerance,
    )
    fvg_ok = detect_simple_fvg(candles_5m)
    rebound_ok = detect_simple_rebound(candles_5m)

    price_ok = price_change >= config.price_change_threshold
    volume_ok = volume_ratio >= config.volume_ratio_threshold

    chasing_risk = is_chasing_risk(candles_5m)
    volume_drop_risk = is_volume_drop_risk(candles_1m)

    # 무효화 조건.
    if btc_condition == "bearish_crash":
        return None, "btc_bearish_crash"
    if not volume_ok and price_ok:
        return None, "volume_below_threshold"
    if chasing_risk:
        return None, "chasing_risk"
    if volume_drop_risk:
        return None, "volume_drop_risk"

    # 필수 조건: 가격+거래량+swing
    if not (price_ok and volume_ok and swing_ok):
        failed: List[str] = []
        if not price_ok:
            failed.append("price")
        if not volume_ok:
            failed.append("volume")
        if not swing_ok:
            failed.append("swing")
        return None, f"required_conditions_not_met({','.join(failed)})"

    score = calculate_pump_score(price_ok, volume_ok, swing_ok, fvg_ok, rebound_ok, btc_condition)
    reasons = _build_reasons(price_ok, volume_ok, swing_ok, fvg_ok, rebound_ok)
    return PumpMetrics(
        symbol=symbol,
        current_price=candles_5m[-1].close,
        price_change_5m=price_change,
        volume_ratio_5m=volume_ratio,
        swing_breakout=swing_ok,
        fvg_like=fvg_ok,
        rebound_like=rebound_ok,
        btc_condition=btc_condition,
        chasing_risk=chasing_risk,
        volume_drop_risk=volume_drop_risk,
        score=score,
        reasons=reasons,
    ), None


def _print_alert(metric: PumpMetrics) -> None:
    reason = " + ".join(metric.reasons) if metric.reasons else "n/a"
    print("[급등 초입 후보 감지]")
    print(f"symbol: {metric.symbol}")
    print(f"price_change_5m: {metric.price_change_5m * 100:+.2f}%")
    print(f"volume_ratio: {metric.volume_ratio_5m:.2f}x")
    print(f"score: {metric.score:.2f}")
    print(f"reason: {reason}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pump entry candidate detector.")
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TEST_CONFIG if args.test_mode else REAL_CONFIG

    # alerts=0이어도 logs 디렉터리는 항상 생성.
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    btc_condition = evaluate_btc_condition()
    if btc_condition == "bearish_crash":
        print("[INFO] BTC 급락 조건 충족으로 이번 스캔 신호를 모두 무효화합니다.")
        return

    symbols = fetch_top_usdt_symbols(TOP_SYMBOL_LIMIT)
    if not symbols:
        print("[WARN] 스캔 대상 심볼을 가져오지 못했습니다.")
        return

    print(f"[INFO] scanning {len(symbols)} symbols | mode={config.mode} | btc_condition={btc_condition}")

    alerts = 0
    now_ts = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()

    for symbol in symbols:
        if symbol == "BTCUSDT":
            continue
        try:
            metric, reject_reason = scan_symbol(symbol, btc_condition, config)
        except Exception as exc:  # defensive: continue full scan
            print(f"[WARN] symbol scan failed: {symbol} ({exc})")
            continue

        if metric is None:
            if args.debug:
                reason = reject_reason or "filtered"
                print(f"[DEBUG] {symbol} filtered: {reason}")
            continue

        if should_alert(
            symbol,
            metric.current_price,
            metric.score,
            now_ts,
            config.score_threshold,
        ):
            _print_alert(metric)
            write_alert_csv(
                timestamp_iso=now_iso,
                symbol=metric.symbol,
                price_change_5m=metric.price_change_5m,
                volume_ratio=metric.volume_ratio_5m,
                score=metric.score,
                btc_condition=metric.btc_condition,
                alert_reason=" + ".join(metric.reasons),
            )
            alerts += 1

    elapsed = time.time() - start
    print(f"[INFO] scan complete | alerts={alerts} | elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
