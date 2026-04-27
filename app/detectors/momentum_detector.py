"""Momentum continuation detector.

This module scans Binance USDT markets and prints radar-style alerts for symbols
that are already rising (not early pump entry). It does NOT place orders and
does NOT modify or depend on pump_detector runtime state.

Run:
    python -m app.detectors.momentum_detector
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.detectors.pump_detector import (
    TOP_SYMBOL_LIMIT,
    PumpMetrics,
    assess_chasing_risk,
    calculate_pump_score,
    calculate_volume_ratio,
    detect_simple_fvg,
    detect_simple_rebound,
    detect_swing_high_breakout,
    evaluate_btc_condition,
    fetch_klines,
    fetch_top_usdt_symbols,
    should_alert,
)

LOG_PATH = Path("logs/momentum_alerts.csv")


@dataclass(frozen=True)
class MomentumConfig:
    mode: str
    rise_15m_threshold: float
    volume_ratio_threshold: float
    score_threshold: float
    swing_high_tolerance: float
    chasing_rise_15m_threshold: float


REAL_CONFIG = MomentumConfig(
    mode="real",
    rise_15m_threshold=0.02,
    volume_ratio_threshold=1.5,
    score_threshold=0.5,
    swing_high_tolerance=0.005,
    chasing_rise_15m_threshold=0.10,
)

TEST_CONFIG = MomentumConfig(
    mode="test",
    rise_15m_threshold=0.015,
    volume_ratio_threshold=1.2,
    score_threshold=0.45,
    swing_high_tolerance=0.008,
    chasing_rise_15m_threshold=0.14,
)


def calculate_price_change_15m(candles_5m) -> float:
    if len(candles_5m) < 4 or candles_5m[-4].close <= 0:
        return 0.0
    return (candles_5m[-1].close - candles_5m[-4].close) / candles_5m[-4].close


def _build_reasons(price_ok: bool, volume_ok: bool, swing_ok: bool, fvg_ok: bool, rebound_ok: bool) -> List[str]:
    reasons: List[str] = []
    if price_ok:
        reasons.append("momentum_15m")
    if volume_ok:
        reasons.append("volume")
    if swing_ok:
        reasons.append("breakout")
    if fvg_ok:
        reasons.append("fvg_like")
    if rebound_ok:
        reasons.append("rebound")
    return reasons


def write_alert_csv(
    timestamp_iso: str,
    symbol: str,
    price_change_15m: float,
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
                    "price_change_15m",
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
                f"{price_change_15m:.6f}",
                f"{volume_ratio:.6f}",
                f"{score:.6f}",
                btc_condition,
                alert_reason,
            ]
        )


def scan_symbol(symbol: str, btc_condition: str, config: MomentumConfig) -> Tuple[Optional[PumpMetrics], Optional[str]]:
    candles_5m = fetch_klines(symbol, "5m", 40)
    if len(candles_5m) < 25:
        return None, "insufficient_candles"

    rise_15m = calculate_price_change_15m(candles_5m)
    volume_ratio = calculate_volume_ratio(candles_5m, lookback=20)

    price_ok = rise_15m >= config.rise_15m_threshold
    volume_ok = volume_ratio >= config.volume_ratio_threshold

    swing_ok = detect_swing_high_breakout(
        candles_5m,
        lookback=20,
        tolerance=config.swing_high_tolerance,
    )
    fvg_ok = detect_simple_fvg(candles_5m)
    rebound_ok = detect_simple_rebound(candles_5m)

    chasing_risk, _ = assess_chasing_risk(candles_5m, config.chasing_rise_15m_threshold)

    if btc_condition == "bearish_crash":
        return None, "btc_bearish_crash"

    # 보조 레이더 필수 조건: 15분 모멘텀 + 거래량.
    if not (price_ok and volume_ok):
        return None, (
            "required_conditions_not_met("
            f"momentum_15m={price_ok} rise={rise_15m * 100:+.2f}%, "
            f"volume={volume_ok} ratio={volume_ratio:.2f}x"
            ")"
        )

    score = calculate_pump_score(price_ok, volume_ok, swing_ok, fvg_ok, rebound_ok, btc_condition)

    # chasing_risk는 일부 허용: 완전 제외 대신 점수 패널티만 부여.
    if chasing_risk:
        score = max(0.0, score - 0.10)

    reasons = _build_reasons(price_ok, volume_ok, swing_ok, fvg_ok, rebound_ok)
    if chasing_risk:
        reasons.append("chasing_risk_allowed")

    return PumpMetrics(
        symbol=symbol,
        current_price=candles_5m[-1].close,
        price_change_5m=rise_15m,
        volume_ratio_5m=volume_ratio,
        swing_breakout=swing_ok,
        fvg_like=fvg_ok,
        rebound_like=rebound_ok,
        btc_condition=btc_condition,
        chasing_risk=chasing_risk,
        volume_drop_risk=False,
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
    parser = argparse.ArgumentParser(description="Momentum continuation detector.")
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
    filtered_counts: Dict[str, int] = {}
    now_ts = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()

    for symbol in symbols:
        if symbol == "BTCUSDT":
            continue
        try:
            metric, reject_reason = scan_symbol(symbol, btc_condition, config)
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
                price_change_15m=metric.price_change_5m,
                volume_ratio=metric.volume_ratio_5m,
                score=metric.score,
                btc_condition=metric.btc_condition,
                alert_reason=" + ".join(metric.reasons),
            )
            alerts += 1

    elapsed = time.time() - start
    filtered_text = ", ".join(f"{k}: {v}" for k, v in sorted(filtered_counts.items()))
    print(
        f"[INFO] scan complete | alerts={alerts} | filtered={{{filtered_text}}} | elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
