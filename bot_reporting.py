"""Presentation and an observational journal. No trading decisions or broker calls.

Only allowlisted signal data enters the journal. Failures are reported to the
operator and never used to accept/reject an entry or change an active trade.
"""
import csv
import hashlib
import io
import json
import math
import os
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

VERSION = "presentation-v1"
TITLE = "دكتور محمد"


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def fmt(value):
    value = number(value)
    return "غير متاح" if value is None else f"{value:.1f}"


def clean(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    try:
        return number(value)
    except Exception:
        return None


def split_levels(levels, price):
    values = sorted(set(x for v in levels if (x := number(v)) is not None))
    return ([v for v in reversed(values) if v < price],
            [v for v in values if v > price], [v for v in values if v == price])


def levels_text(levels):
    return " / ".join(fmt(v) for v in levels) or "غير متاحة"


def key_levels_text(levels, price):
    supports, resistances, at_price = split_levels(levels, price)
    lines = ["📍 Key Levels — الأقرب أولًا",
             "مقاومات: " + levels_text(resistances),
             "السعر: " + fmt(price), "دعوم: " + levels_text(supports)]
    if at_price:
        lines.append("مستوى عند السعر: " + levels_text(at_price))
    return "\n".join(lines)


def scenarios(levels, price, bias, market_label, market_dir):
    supports, resistances, _ = split_levels(levels, price)
    support = supports[0] if supports else None
    resistance = resistances[0] if resistances else None
    up = "المحطات الأعلى: " + levels_text(resistances)
    down = "المحطات الأدنى: " + levels_text(supports)
    bullish = (bias == "Bullish" and market_dir != "Bearish"
               and market_label not in ("Range", "Messy", "Weak"))
    bearish = (bias == "Bearish" and market_dir != "Bullish"
               and market_label not in ("Range", "Messy", "Weak"))
    if bullish and support is not None:
        return ("ميل صاعد مشروط بالحفاظ على الدعم " + fmt(support),
                "إذا حافظ السعر على " + fmt(support) + " وظهر تأكيد ارتداد، نراقب الصعود. " + up,
                "إذا فقد الدعم " + fmt(support) + "، نراقب امتداد الهبوط. "
                + "المحطات التالية: " + levels_text(supports[1:]),
                "تداول السعر دون " + fmt(support) + " يستدعي إعادة تقييم سيناريو الصعود.")
    if bearish and resistance is not None:
        return ("ميل هابط مشروط بالبقاء دون المقاومة " + fmt(resistance),
                "إذا ظهر رفض عند " + fmt(resistance) + "، نراقب الهبوط. " + down,
                "إذا تجاوز المقاومة " + fmt(resistance) + "، نراقب امتداد الصعود. "
                + "المحطات التالية: " + levels_text(resistances[1:]),
                "تداول السعر فوق " + fmt(resistance) + " يستدعي إعادة تقييم سيناريو الهبوط.")
    if support is not None and resistance is not None:
        return ("لا أفضلية اتجاهية واضحة في هذه القراءة",
                "نراقب التفاعل بين الدعم " + fmt(support) + " والمقاومة " + fmt(resistance)
                + ". الارتداد أو الرفض يحتاج تأكيدًا؛ وجود السعر عند المستوى وحده لا يكفي.",
                "تجاوز أحد الطرفين يستدعي متابعة الاتجاه الجديد. " + up + "؛ " + down,
                "الخروج من المنطقة بين " + fmt(support) + " و" + fmt(resistance)
                + " أو تغير الميل يستدعي إعادة التقييم.")
    return ("المستويات المتاحة لا تكفي لبناء سيناريو متوازن",
            "نراقب أقرب مستوى متاح مع انتظار تأكيد مستقل. " + up + "؛ " + down,
            "تُعاد القراءة عند ظهور مستويات إضافية أو تغير الميل.",
            "لا يوجد شرط إلغاء محدد لسيناريو اتجاهي مكتمل في هذه القراءة.")


def hourly_text(session, symbol, bias, market, levels, price, trade, now,
                previous=None):
    supports, resistances, _ = split_levels(levels, price)
    lines = [TITLE + " | 🕐 تحديث الساعة", now.strftime("%Y-%m-%d %H:%M") + " بتوقيت الرياض",
             symbol + " | " + session, "", "🧭 قراءة السوق", "الحالة: " + market,
             "الميل 1H/4H: " + bias, "", key_levels_text(levels, price), "",
             "👀 ما نراقبه", "أقرب دعم: " + fmt(supports[0] if supports else None)
             + " | أقرب مقاومة: " + fmt(resistances[0] if resistances else None),
             "نراقب رد الفعل عند المستوى؛ تأكيد الدخول يخضع لشروط البوت الحالية."]
    if previous and previous.get("symbol") == symbol:
        changes = []
        delta = price - previous["price"]
        if abs(delta) >= 0.05:
            changes.append(f"السعر {delta:+.1f} نقطة")
        if previous["bias"] != bias:
            changes.append("الميل " + previous["bias"] + " ← " + bias)
        if previous["levels"] != list(levels):
            changes.append("تحدّثت Key Levels المعروضة")
        if changes:
            lines += ["", "التغير منذ التحديث السابق (" + previous["time"] + "): " + "؛ ".join(changes)]
    lines += ["", "🎯 متابعة الصفقة"]
    if not trade:
        lines.append("لا توجد صفقة نشطة.")
    else:
        status = {"pending": "بانتظار التفعيل", "live": "مفعّلة"}.get(trade.get("status"), "غير معروف")
        lines += [status + " | " + str(trade.get("direction", "")),
                  "الدخول المخطط: " + fmt(trade.get("entry")) + " | الوقف الحالي: " + fmt(trade.get("stop")),
                  "T1: " + fmt(trade.get("t1")) + " | T2: " + fmt(trade.get("t2"))
                  + " | T3: " + fmt(trade.get("t3"))]
        targets = trade.get("dynamic_targets", {})
        pending = [(name, t) for name, t in targets.items() if not t.get("hit")]
        pending.sort(key=lambda pair: int(pair[0][1:]))
        if pending:
            name, target = pending[0]
            lines.append("الهدف التالي المتابع: " + name + " عند " + fmt(target.get("price")))
        for t in trade.get("target_map", []):
            lines.append(str(t["name"]) + ": " + fmt(t["price"]) + " | تقدير خريطة الأهداف: "
                         + str(t.get("prob", "N/A")) + "% | " + str(t.get("zone", "")))
    return "\n".join(lines)


def daily_text(symbol, session, price, levels, bias, market_label, market_dir,
               adx, liquidity, atr, now):
    view, primary, alternative, invalidation = scenarios(levels, price, bias, market_label, market_dir)
    return "\n".join([
        TITLE + " | 🗓️ خطة اليوم", now.strftime("%Y-%m-%d %H:%M") + " بتوقيت الرياض",
        symbol + " | " + session, "السعر: " + fmt(price), "",
        "🧭 الصورة العامة", view,
        "الحالة: " + market_label + " | الاتجاه: " + market_dir + " | الميل 1H/4H: " + bias,
        "ADX: " + fmt(adx) + " | السيولة: " + liquidity,
        "مقياس التذبذب الساعي (ATR): " + fmt(atr) + " نقطة — مقياس تاريخي، وليس حدودًا متوقعة", "",
        key_levels_text(levels, price), "", "① السيناريو الأساسي", primary,
        "", "② السيناريو البديل", alternative, "", "↻ متى نعيد التقييم؟", invalidation,
        "", "هذه قراءة مشروطة للمستويات الحالية، وليست إشارة دخول أو توقعًا ثابتًا لنهاية اليوم."])


def signal_explanation(direction, level, plan, reasons, levels, pending_invalidate):
    translations = {"Wick Rejection near Level": "تجمّع ذيول رفض قرب المستوى",
                    "Break&Retest": "تحقق شرط الكسر وإعادة الاختبار",
                    "Rejection": "تحقق شرط شمعة الرفض", "Stoch RSI cross": "تقاطع Stoch RSI",
                    "Momentum shift": "تحول في الزخم", "Momentum breakout": "تحقق شرط اختراق الزخم"}
    evidence = [translations[r] for r in reasons if r in translations]
    entry, t1, stop = (number(plan.get(k)) for k in ("entry", "t1", "stop"))
    sign = 1 if direction == "BUY" else -1
    obstacles = sorted(set(v for x in levels if (v := number(x)) is not None
                           and entry is not None and t1 is not None
                           and 0 < sign * (v - entry) < sign * (t1 - entry)), reverse=sign < 0)
    activation = "عند أو فوق " if direction == "BUY" else "عند أو تحت "
    route = "الدخول " + fmt(entry)
    for name in ("t1", "t2", "t3"):
        if number(plan.get(name)) is not None:
            route += " ← " + name.upper() + " " + fmt(plan[name])
    cancel = ("وصول السعر إلى الوقف " + fmt(stop) + " قبل التفعيل يلغي الإشارة وفق قواعد البوت."
              if pending_invalidate else "الإلغاء التلقائي عند الوقف قبل التفعيل معطّل في الإعدادات.")
    return "\n".join(["🧠 شرح الإشارة المبكرة", "المستوى المرجعي: " + fmt(level),
                      "ما رصده البوت: " + ("؛ ".join(evidence) or "راجع قائمة الأسباب المسجلة أعلاه"),
                      "شرط التفعيل الحالي: رصد السعر " + activation + fmt(entry) + " في دورة المتابعة.",
                      "المسار المحتمل بعد التفعيل: " + route,
                      "مستويات معاكسة قبل T1: " + (levels_text(obstacles) if obstacles else "لا توجد ضمن Key Levels المتاحة"),
                      cancel, "بعد التفعيل تُطبّق قواعد الوقف وإدارة الصفقة الحالية.",
                      "تحقق الأهداف مشروط باستمرار الحركة؛ هذا الشرح لا يضيف شرط دخول جديدًا."])


TRADE_FIELDS = (
    "direction", "level", "entry", "stop", "initial_stop", "t1", "t2", "t3", "t4",
    "status", "created_utc", "last_price", "last_update_riyadh", "trade_type", "trigger",
    "score", "confidence", "bias_at_entry", "signal_rr", "level_quality_score",
    "level_touches", "level_strong", "level_tradable", "market_label_at_entry",
    "session_at_entry", "target_map", "all_target_candidates", "dynamic_targets",
    "activation_price", "activated_at_riyadh", "last_target_hit_idx", "stop_pending")


class EvaluationJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.last_error = None
        self.last_success = None
        self._last_observation = None

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=0.5)
        conn.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, at TEXT NOT NULL, "
                     "trade_id TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL)")
        conn.execute("CREATE INDEX IF NOT EXISTS events_at ON events(at)")
        return conn

    def trade_id(self, trade):
        identity = [str(trade.get(k, "")) for k in ("created_utc", "direction", "entry", "level")]
        return hashlib.sha256("|".join(identity).encode()).hexdigest()[:20]

    def record(self, event, trade, now, **details):
        if not trade:
            return False
        try:
            trade_id = self.trade_id(trade)
            sample_key = (trade_id, now.strftime("%Y-%m-%dT%H:%M"))
            with self.lock:
                if event == "observation" and sample_key == self._last_observation:
                    return True
                payload = clean({"schema": 1, "version": VERSION,
                                 "price_basis": "observed_index_not_broker_fill",
                                 "trade": {k: trade[k] for k in TRADE_FIELDS if k in trade},
                                 "details": details})
                conn = self._connect()
                try:
                    with conn:
                        conn.execute("INSERT INTO events(at,trade_id,event,payload) VALUES(?,?,?,?)",
                                     (now.isoformat(), trade_id, event,
                                      json.dumps(payload, ensure_ascii=False, allow_nan=False)))
                finally:
                    conn.close()
                if event == "observation":
                    self._last_observation = sample_key
                self.last_success = now.isoformat()
                return True
        except Exception as exc:
            self.last_error = now.isoformat() + " | " + type(exc).__name__
            # Do not print credentials, event payloads or database exception strings.
            print("[WARN] Evaluation journal write failed: " + type(exc).__name__)
            return False

    def rows(self, days, now):
        if not 1 <= days <= 90:
            raise ValueError("days must be between 1 and 90")
        since = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if not self.path.exists():
            return []
        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT at,trade_id,event,payload FROM events WHERE at>=? AND at<=? ORDER BY id LIMIT 100001",
                                    (since.isoformat(), now.isoformat())).fetchall()
            finally:
                conn.close()
        if len(rows) > 100000:
            raise ValueError("export too large; choose fewer days")
        return [(at, tid, event, json.loads(payload)) for at, tid, event, payload in rows]

    def summary(self, days, now):
        rows = self.rows(days, now)
        counts = {event: sum(row[2] == event for row in rows)
                  for event in ("signal", "activated", "cancelled", "closed")}
        results = [number(p["details"].get("r_result")) for _, _, event, p in rows if event == "closed"]
        results = [r for r in results if r is not None]
        wins, losses, be = (sum(r > 0 for r in results), sum(r < 0 for r in results), sum(r == 0 for r in results))
        return "\n".join([TITLE + " | 📒 سجل التقييم", f"الفترة: آخر {days} أيام تقويمية حتى الآن",
                           f"إشارات صدرت: {counts['signal']} | تفعّلت: {counts['activated']}",
                           f"أُلغيت قبل التفعيل: {counts['cancelled']} | أُغلقت: {counts['closed']}",
                           f"نتائج الإغلاق: ربح {wins} | خسارة {losses} | تعادل {be}",
                           f"المحصلة المرصودة للمؤشر: {sum(results):+.2f}R — قبل التكاليف",
                           "الإحصاء حسب وقت الحدث؛ قد تكون الصفقة المغلقة قد بدأت قبل الفترة.",
                           "لا يمثل ربح العقود الفعلي، ولا يعيد بناء تداولات ما قبل تشغيل السجل.",
                           "آخر تعثر كتابة في هذه الجلسة: " + (self.last_error or "لا يوجد"),
                           "تنزيل الملف: /export_evaluation " + str(days)])

    def export_csv(self, days, now):
        rows = self.rows(days, now)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["time_riyadh", "signal_id", "event", "direction", "status", "level",
                         "planned_entry", "observed_activation", "initial_stop", "current_stop",
                         "T1", "T2", "T3", "score", "quality_estimate", "market", "session",
                         "observed_price", "exit_price", "exit_reason", "result_R", "highest_target_hit",
                         "details_json", "snapshot_json"])
        for at, tid, event, payload in rows:
            tr, detail = payload["trade"], payload["details"]
            values = [at, tid, event, tr.get("direction"), tr.get("status"), tr.get("level"),
                      tr.get("entry"), tr.get("activation_price"), tr.get("initial_stop"), tr.get("stop"),
                      tr.get("t1"), tr.get("t2"), tr.get("t3"), tr.get("score"), tr.get("confidence"),
                      tr.get("market_label_at_entry"), tr.get("session_at_entry"),
                      detail.get("observed_price", tr.get("last_price")), detail.get("exit_price"),
                      detail.get("exit_reason"), detail.get("r_result"), detail.get("max_target_hit"),
                      json.dumps(detail, ensure_ascii=False), json.dumps(tr, ensure_ascii=False)]
            # Protect user-entered manual close reasons when opened by spreadsheet apps.
            values = [("'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
                      for v in values]
            writer.writerow(values)
        data = output.getvalue().encode("utf-8-sig")
        if len(data) > 24 * 1024 * 1024:
            raise ValueError("export too large; choose fewer days")
        return data


class Reports:
    def __init__(self):
        default = Path(__file__).resolve().parent / "data" / "evaluation.sqlite3"
        self.journal = EvaluationJournal(os.getenv("EVALUATION_DB_PATH", str(default)))
        self.previous_hour = None
        self.daily_anchor = None
        self.daily_sent_at = None

    def hourly(self, session, symbol, bias, market, levels, price, trade, now):
        message = hourly_text(session, symbol, bias, market, levels, price, trade, now, self.previous_hour)
        self.previous_hour = {"symbol": symbol, "bias": bias, "levels": list(levels),
                              "price": price, "time": now.strftime("%Y-%m-%d %H:%M")}
        return message

    def daily_due(self, symbol, levels, price, bias, label, market_dir, ny_now, now):
        # Presentation schedule only. Never alters BotState or the trading clock.
        if ny_now.weekday() >= 5 or not 9 * 60 + 15 <= ny_now.hour * 60 + ny_now.minute < 16 * 60:
            return False
        anchor = self.daily_anchor
        first = anchor is None or anchor["day"] != ny_now.date() or anchor["symbol"] != symbol
        changed = False
        if not first:
            support, resistance = anchor["support"], anchor["resistance"]
            changed = ((support is not None and price < support)
                       or (resistance is not None and price > resistance)
                       or (bias, label, market_dir) != anchor["regime"])
            changed = changed and (now - self.daily_sent_at).total_seconds() >= 1800
        if not first and not changed:
            return False
        supports, resistances, _ = split_levels(levels, price)
        self.daily_anchor = {"symbol": symbol, "day": ny_now.date(),
                             "support": supports[0] if supports else None,
                             "resistance": resistances[0] if resistances else None,
                             "regime": (bias, label, market_dir)}
        self.daily_sent_at = now
        return True


REPORTS = Reports()
