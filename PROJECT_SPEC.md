# PROJECT_SPEC.md

# AI-Trading-System für US-Aktien / FTMO / Polygon 1m-Daten / 5m-Backtest

## Projektziel

Ich möchte ein autonomes AI-Trading-System für US-Aktien entwickeln, ausgelegt auf FTMO-Regeln.

Das System soll später selbstständig Long- und Short-Trades handeln können. Der erste technische Schritt ist aber nur der Aufbau der Datenpipeline, danach Indikatoren, dann Backtest, dann AI-Modelle.

Wichtig: Nicht alles auf einmal bauen. Zuerst Datenpipeline, danach schrittweise erweitern.

---

# 1. Datenquelle und Datenpipeline

Die historischen Marktdaten sollen über Polygon/Massive per API heruntergeladen werden.

Geplant ist:

- 1-Minuten-OHLCV-Daten per API downloaden
- viele US-Aktien, vor allem S&P-500- und Nasdaq-Titel
- Daten lokal als Parquet speichern
- aus den 1-Minuten-Daten eigene 5-Minuten-Kerzen bauen
- nur reguläre Wall-Street-Handelszeiten verwenden
- Pre-Market und After-Hours nicht für Signale verwenden

Die Datenpipeline soll so aussehen:

Polygon/Massive API
→ 1-Minuten-OHLCV-Daten downloaden
→ lokal als Parquet speichern
→ nur reguläre Wall-Street-Session filtern
→ eigene 5-Minuten-Kerzen bauen
→ Indikatoren berechnen
→ Backtest / AI-Training / Walk-Forward-Test

Die Rohdaten sollen als 1-Minuten-Daten behalten werden, damit später auch andere Timeframes getestet werden können.

Die Handelssignale laufen aber zunächst auf 5-Minuten-Kerzen.

---

# 2. Lokale Datenstruktur

Gewünschte Projektstruktur:

data/
├── raw/
│   └── 1m/
├── processed/
│   └── 5m/
└── metadata/
    ├── symbol_universe.csv
    ├── earnings_calendar.csv
    └── corporate_actions.csv

Beispiele:

data/raw/1m/AAPL.parquet
data/raw/1m/MSFT.parquet

data/processed/5m/AAPL.parquet
data/processed/5m/MSFT.parquet

---

# 3. Aktienuniversum

Das Aktienuniversum soll später aus den wichtigsten S&P-500- und Nasdaq-Titeln bestehen.

Für den Start sollen ungefähr die Top 50 S&P-500-Titel und Top 30 Nasdaq-Titel verwendet werden, zusammengeführt und ohne Duplikate.

Die Symbole stehen in:

data/metadata/symbol_universe.csv

Format:

symbol
AAPL
MSFT
NVDA

Wichtig:
Falls einzelne Symbole bei Polygon anders geschrieben werden, z. B. BRK.B vs BRK-B, soll das Script Fehler sauber loggen und nicht komplett abbrechen.

---

# 4. Handelszeiten

Gehandelt beziehungsweise für Signale verwendet werden darf nur die reguläre Wall-Street-Session:

09:30 bis 16:00 New York Time

Regeln:

- Pre-Market: keine Signale
- After-Hours: keine Signale
- neue Trades: keine neuen Trades mehr 30 Minuten vor Marktschluss
- offene Trades: alle offenen Trades 10 Minuten vor Marktschluss schließen

Praktisch:

- keine neuen Trades nach 15:30 New York Time
- alle offenen Trades spätestens um 15:50 New York Time schließen

---

# 5. 1-Minuten- zu 5-Minuten-Kerzen

Aus den 1-Minuten-Daten sollen eigene 5-Minuten-Kerzen gebaut werden.

Beispiel:

09:30 bis 09:34 ergibt die abgeschlossene 09:35-Kerze.
09:35 bis 09:39 ergibt die abgeschlossene 09:40-Kerze.
15:55 bis 15:59 ergibt die abgeschlossene 16:00-Kerze.

Das System darf später immer erst nach Abschluss einer 5-Minuten-Kerze entscheiden.

OHLCV-Aggregation:

- open = erster Open-Wert
- high = höchster High-Wert
- low = niedrigster Low-Wert
- close = letzter Close-Wert
- volume = Summe des Volumens

Zeitzone:

- Rohdaten korrekt behandeln
- auf America/New_York konvertieren
- Session-Filter 09:30 bis 16:00 New York Time

---

# 6. Hauptindikator: 2-Session-VWAP

Der zentrale Indikator ist ein 2-Session-VWAP.

Dieser VWAP soll immer die letzten zwei regulären Handelssessions berücksichtigen.

Es sollen also nur Daten aus den regulären Handelszeiten verwendet werden.

Berechnung:

typical_price = (High + Low + Close) / 3

2-Session-VWAP =
Summe(typical_price * volume über die letzten 2 regulären Sessions)
/
Summe(volume über die letzten 2 regulären Sessions)

Beispiel:

Am Mittwoch basiert der VWAP auf Dienstag + Mittwoch bis zur aktuellen abgeschlossenen 5-Minuten-Kerze.

Am Donnerstag basiert der VWAP auf Mittwoch + Donnerstag bis zur aktuellen abgeschlossenen 5-Minuten-Kerze.

---

# 7. Weitere Indikatoren

Zusätzlich nutzt die Strategie:

- RSI
- Momentum
- Support/Resistance
- ATR / Volatilität

RSI wird verwendet für:

- Trendbestätigung
- Momentum-Zustand
- Überkauft-/Überverkauft-Situationen
- Mean-Reversion-Erkennung

Momentum wird verwendet für:

- Richtungsbestätigung
- Beschleunigung / Abschwächung einer Bewegung
- Exit bei Momentum-Verlust

Support und Resistance sollen algorithmisch erkannt werden, zum Beispiel über:

- Swing-Highs
- Swing-Lows
- Previous Day High
- Previous Day Low
- Previous Close
- 2-Day High
- 2-Day Low
- VWAP-Zonen

Ein separater harter Volumenfilter soll in der Hauptstrategie nicht verwendet werden, weil Volumen bereits indirekt im VWAP enthalten ist.

Aber:
Volumen darf vom unabhängigen AI-Modell als Feature berücksichtigt werden.

---

# 8. Setup-Arten

Das System soll vier Setup-Arten erkennen und separat markieren:

- TREND_LONG
- TREND_SHORT
- MEAN_REVERSION_LONG
- MEAN_REVERSION_SHORT

Trendfolge-Trades handeln in Richtung des VWAP-Bias.

Beispiel Trend Long:

- Kurs über 2-Session-VWAP
- RSI bestätigt Stärke
- Momentum positiv
- Support wurde gehalten oder Resistance wurde gebrochen
- erwarteter Move mindestens 0.2 %

Beispiel Trend Short:

- Kurs unter 2-Session-VWAP
- RSI bestätigt Schwäche
- Momentum negativ
- Resistance wurde gehalten oder Support wurde gebrochen
- erwarteter Move mindestens 0.2 %

Mean-Reversion-Trades sind ebenfalls erlaubt.

Beispiel Mean-Reversion Long:

- Kurs stark unter VWAP
- RSI überverkauft
- Momentum verliert Abwärtskraft
- Kurs erreicht Support-Zone
- AI erkennt erhöhte Rücklaufwahrscheinlichkeit

Beispiel Mean-Reversion Short:

- Kurs stark über VWAP
- RSI überkauft
- Momentum verliert Aufwärtskraft
- Kurs erreicht Resistance-Zone
- AI erkennt erhöhte Rücklaufwahrscheinlichkeit

Mean-Reversion-Trades müssen im Backtest separat ausgewertet werden, damit später entschieden werden kann, ob diese Setup-Art sinnvoll ist oder deaktiviert werden sollte.

---

# 9. Grundlegende System-Pipeline

Der grobe Ablauf des Systems soll später so sein:

5-Minuten-Kerze schließt
→ Handelszeit prüfen
→ Indikatoren berechnen
→ Setup erkennen
→ Strategie-Typ bestimmen
→ Haupt-AI-Modell bewertet Setup
→ unabhängiges AI-Modell bewertet Setup separat
→ Earnings-Regel prüfen
→ Mindestprofit prüfen
→ Stop-Distanz prüfen
→ Tagesverlust prüfen
→ Positionsgröße berechnen
→ ggf. Positionsgröße reduzieren
→ Mindestpositionsgröße prüfen
→ optionalen 1m-LTF-Filter prüfen
→ Trade eröffnen oder ablehnen
→ TP/Teilverkauf/Trailing/Exit-Logik setzen
→ Trade überwachen
→ spätestens 10 Minuten vor Marktschluss schließen
→ Trade vollständig loggen
→ abgelehnte Trades ebenfalls loggen
→ Audit-/Logbuch-Agent analysiert später optional

Ein Setup allein reicht nicht für einen Trade. Nach der Setup-Erkennung müssen beide AI-Modelle und alle harten Regeln erfüllt sein.

---

# 10. Haupt-AI-Modell

Das Haupt-AI-Modell soll für jedes erkannte Setup eine Wahrscheinlichkeit und einen erwarteten Move berechnen.

Die Wahrscheinlichkeit bedeutet konkret:

Wie wahrscheinlich ist es, dass dieser Trade mindestens den geforderten Mindestmove erreicht, bevor Stop-Loss, Zeit-Exit oder Gegensignal eintritt?

Das Modell soll ausgeben:

- main_ai_probability
- expected_move_pct

Für normale Trades gelten:

- main_ai_probability >= 70 %
- expected_move_pct >= 0.2 %

Für Earnings-Trades gelten strengere Regeln:

- main_ai_probability >= 85 %
- expected_move_pct >= 0.3 %

Earnings-Trading soll grundsätzlich vermieden werden.

Eine Ausnahme ist nur erlaubt, wenn die strengeren AI- und Profit-Anforderungen erfüllt sind.

---

# 11. Unabhängiges zweites AI-Modell

Zusätzlich zum Haupt-AI-Modell soll es ein unabhängiges zweites AI-Modell geben.

Dieses Modell soll wie ein separater Bewerter funktionieren und die Marktsituation eigenständig einschätzen.

Es soll nicht einfach dieselbe Strategie bestätigen, sondern mit anderem Feature-Set und anderem Blickwinkel arbeiten.

Die Art des unabhängigen Modells soll so aufgebaut werden:

- zweites ML-Modell
- anderes Feature-Set
- separater Blick auf die Marktsituation
- nicht bloß Kopie des Hauptmodells

Das unabhängige AI-Modell soll unter anderem diese Dinge bewerten:

- 2-Session-VWAP
- RSI
- Momentum
- Support/Resistance
- Preisstruktur
- Kerzenverhalten
- Volumen
- relative Volume
- Trendstärke
- Überdehnung
- VWAP-Abstand
- Volatilität
- Abstand zum Ziel
- Abstand zum Stop
- Tageszeit
- Mean-Reversion-Wahrscheinlichkeit

Für normale Trades gilt:

- independent_ai_probability >= 60 %

Für Earnings-Trades gilt:

- independent_ai_probability >= 70 %

Ein Trade darf final nur genommen werden, wenn sowohl das Haupt-AI-Modell als auch das unabhängige AI-Modell zustimmen und alle harten Regeln erfüllt sind.

---

# 12. Backtest des unabhängigen AI-Filters

Sehr wichtig:

Es sollen auch alle Trades gespeichert werden, die das Haupt-AI-Modell genommen hätte, aber die vom unabhängigen AI-Modell blockiert wurden.

Diese Trades sind besonders interessant, weil man damit später prüfen kann, ob der zweite AI-Check wirklich sinnvoll ist oder profitable Trades unnötig blockiert.

Diese Kategorie soll gesondert geloggt werden:

- main_ai_approved = true
- independent_ai_approved = false
- independent_ai_blocked = true

Ziel der Auswertung:

Performance nur mit Haupt-AI
vs.
Performance mit Haupt-AI + unabhängiger AI

---

# 13. Finale Trade-Freigabe

Die finale Trade-Freigabe sieht so aus:

Setup erkannt
+ Haupt-AI erfüllt Schwelle
+ unabhängige AI erfüllt Schwelle
+ Mindestprofit erfüllt
+ Earnings-Regel erfüllt
+ Stop-Distanz erlaubt
+ Zeitregel erfüllt
+ Tagesverlustregel erfüllt
+ Positionsgröße gültig
+ optionaler 1m-LTF-Filter erfüllt, falls aktiviert
= Trade erlaubt

Wenn eine Bedingung nicht erfüllt ist, wird der Trade abgelehnt und mit Ablehnungsgrund gespeichert.

---

# 14. Stop-Regel

Die maximale Stop-Distanz beträgt:

max_stop_distance_pct = 2 %

Wenn der technisch sinnvolle Stop weiter als 2 % vom aktuellen Kurs entfernt liegt, wird der Trade geskippt.

Beispiel:

Entry Long: 100 $
maximal erlaubter Stop: 98 $

Wenn technischer Stop bei 97.50 $ liegen müsste:
→ Trade wird geskippt.

Jeder Trade muss immer mit Stop-Loss eröffnet werden.

Kein Trade ohne Stop.

---

# 15. FTMO-Risk-Regeln und Tagesverlust

Das System soll nach FTMO-Gesichtspunkten konservativ arbeiten.

Der interne maximale Tagesverlust beträgt:

max_daily_loss = 3.5 % der Accountgröße

Dabei müssen berücksichtigt werden:

- geschlossener Tages-PnL
- offener Floating-PnL
- Gebühren
- Slippage
- offene Risiken

Wenn das Tagesverlustlimit gefährdet ist:

- keine neuen Trades
- offene Trades defensiv managen
- bei Erreichen des Limits Positionen schließen

---

# 16. Positionsgrößen-Regel

Die Positionsgröße soll anhand von Entry, Stop, Accountgröße und verbleibendem Tagesverlustpuffer berechnet werden.

Das System darf die Positionsgröße reduzieren, damit ein Stop-Loss den maximalen Tagesverlust von 3.5 % nicht verletzt.

Zusätzlich prüfen:

- Würde ein Stop-Loss den Tagesverlust zu nah an 3.5 % bringen?
- Sind bereits andere Positionen offen?
- Gibt es korrelierte Positionen?
- Ist genug Margin vorhanden?
- Ist das Symbol handelbar?

Dabei gilt eine Mindestpositionsgröße:

minimum_position_size = 5 % der Accountgröße

Wenn die notwendige Reduzierung dazu führen würde, dass die Positionsgröße unter 5 % der Accountgröße fällt, wird der Trade geskippt.

Beispiel:

Accountgröße: 100.000 $
Mindestpositionsgröße: 5.000 $

Wenn die Risk Engine wegen Tagesverlustpuffer nur noch 3.000 $ Positionsgröße erlauben würde:
→ Trade wird nicht genommen.

---

# 17. Take-Profit und Trade-Management

Das System soll den Take-Profit selbstständig dort setzen, wo es ihn anhand der Marktsituation für sinnvoll hält.

Es soll also keinen starren Pflicht-TP nur bei 0.2 % geben.

Der Trade muss aber mindestens das jeweilige Mindestziel erfüllen.

Normale Trades:

- Mindestziel / expected_move_pct >= 0.2 %

Earnings-Trades:

- Mindestziel / expected_move_pct >= 0.3 %

Mögliche TP- und Exit-Methoden:

- variabler TP an Support/Resistance
- Teilverkauf bei +0.2 %
- bei Earnings Mindestziel mindestens +0.3 %
- Trailing Stop nach +0.2 %
- Exit bei VWAP-Rücklauf
- Exit bei Momentum-Verlust
- Exit bei Gegensignal
- Zeit-Exit
- Marktschluss-Exit

Trade-Management:

- Stop-Loss Pflicht
- Take-Profit dynamisch
- Teilverkauf möglich
- Trailing Stop möglich
- VWAP-Exit möglich
- Momentum-Exit möglich
- Gegensignal-Exit möglich
- Marktschluss-Exit Pflicht

Alle offenen Trades müssen spätestens 10 Minuten vor Marktschluss geschlossen werden.

---

# 18. Gebühren und Slippage

Die Kostenannahmen für den Backtest sind:

- Gebühr = 0.005 %
- Slippage = 3 bps = 0.03 %

Im Backtest sollen sowohl Brutto- als auch Netto-Ergebnisse gespeichert werden:

- gross_pnl
- net_pnl
- gross_pnl_pct
- net_pnl_pct
- fees
- slippage

Slippage und Gebühren müssen in der Backtest-Auswertung berücksichtigt werden.

---

# 19. Optionaler Low-Timeframe-Entry-Filter auf 1-Minuten-Basis

Zusätzlich soll optional eine Low-Timeframe-Entry-Confirmation auf 1-Minuten-Basis eingebaut werden.

Wichtig:

- Das Hauptsignal entsteht weiterhin ausschließlich auf abgeschlossenen 5-Minuten-Kerzen.
- Die 1-Minuten-Kerze erzeugt keine eigenen Trades.
- Die 1-Minuten-Kerze erzeugt keine eigenen Setups.
- Sie dient nur als zusätzlicher Entry-Filter nach einem bereits vollständig gültigen 5-Minuten-Setup.

Wenn ein 5-Minuten-Setup inklusive Haupt-AI, unabhängiger AI, Earnings-Regel, Mindestprofit, Stop-Distanz, Tagesverlustregel und Positionsgrößenprüfung gültig ist, prüft das System optional die zuletzt abgeschlossene 1-Minuten-Kerze.

Wenn die 1-Minuten-Bestätigung nicht passt:

- Trade wird geskippt.
- Es wird nicht gewartet.
- Es wird kein späterer Einstieg gesucht.
- Der Trade wird als LTF-abgelehnt gespeichert.

---

# 20. Zwei Arten von 1-Minuten-Bestätigung

Es sollen grundsätzlich diese 1-Minuten-Filter möglich sein:

## 20.1 Micro-Structure Confirmation

Für Long:

- Higher Low
- Break über lokales 1m-Micro-High
- bullische Fortsetzung nach Pullback

Für Short:

- Lower High
- Break unter lokales 1m-Micro-Low
- bärische Fortsetzung nach Pullback

## 20.2 Bad-1m-Candle-Filter

Für Long wird der Trade geskippt, wenn die letzte 1m-Kerze zum Beispiel zeigt:

- klare bärische Rejection
- langer oberer Wick
- Close nahe Tief
- kurzfristige Überdehnung

Für Short wird der Trade geskippt, wenn die letzte 1m-Kerze zum Beispiel zeigt:

- klare bullische Rejection
- langer unterer Wick
- Close nahe Hoch
- kurzfristige Überdehnung

---

# 21. Backtest-Plan für den 1m-LTF-Filter

Theoretisch wären diese Varianten möglich:

Variante A:
Ohne 1m-Bestätigung

Variante B:
Mit Micro-Structure Confirmation

Variante C:
Mit Bad-1m-Candle-Filter

Variante D:
Mit beiden Filtern kombiniert

Für die erste Umsetzung sollen aber nur diese beiden Varianten relevant sein:

Variante A:
Ohne 1m-Bestätigung

Variante D:
Mit Micro-Structure Confirmation + Bad-1m-Candle-Filter kombiniert

Ziel ist erstmal nicht, jede Zwischenvariante zu optimieren, sondern nur zu prüfen:

Bringt die komplette 1m-Bestätigung einen Vorteil gegenüber keiner 1m-Bestätigung?

Im Backtest sollen daher mindestens diese zwei Modi vergleichbar sein:

use_ltf_confirmation = false

und:

use_ltf_confirmation = true
ltf_use_micro_structure = true
ltf_use_bad_candle_filter = true

Besonders wichtig ist die Auswertung aller Trades, die nach 5m-Setup, Haupt-AI, unabhängiger AI und Risk-Regeln eigentlich erlaubt gewesen wären, aber durch die kombinierte 1m-Bestätigung geblockt wurden.

Diese Trades sollen separat markiert werden:

- ltf_filter_enabled = true
- ltf_confirmed = false
- ltf_rejected = true
- ltf_rejection_reason = "micro_structure_not_confirmed"

oder:

- ltf_rejection_reason = "bad_1m_candle"

oder:

- ltf_rejection_reason = "micro_structure_not_confirmed_and_bad_1m_candle"

Zusätzliche Logging-Felder für den LTF-Filter:

- ltf_filter_enabled
- ltf_confirmed
- ltf_rejected
- ltf_rejection_reason
- ltf_candle_open
- ltf_candle_high
- ltf_candle_low
- ltf_candle_close
- ltf_candle_body_pct
- ltf_upper_wick_pct
- ltf_lower_wick_pct
- ltf_micro_structure_signal

In der Backtest-Auswertung soll zusätzlich verglichen werden:

- Performance ohne 1m-Bestätigung
- Performance mit kombinierter 1m-Bestätigung
- Trades, die durch 1m-Bestätigung geblockt wurden
- PnL dieser geblockten Trades, falls sie ohne LTF-Filter genommen worden wären
- Auswirkung auf Winrate
- Auswirkung auf Profit Factor
- Auswirkung auf Max Drawdown
- Auswirkung auf durchschnittlichen Netto-Trade
- Auswirkung auf Trade-Anzahl
- Auswirkung auf Trendfolge-Trades
- Auswirkung auf Mean-Reversion-Trades
- Auswirkung auf Long-Trades
- Auswirkung auf Short-Trades

Die finale erste Vergleichsfrage lautet:

Ist Variante D besser als Variante A?

Also:

Kein 1m-Filter
vs.
Micro-Structure Confirmation + Bad-1m-Candle-Filter

---

# 22. Training und Walk-Forward-Backtest

Das Modell soll monatlich neu trainiert werden.

Der Standard-Trainingszeitraum beträgt:

training_window_months = 6

Der Backtest soll über 18 Monate laufen:

backtest_period_months = 18

Die Methode soll ein Walk-Forward-Backtest sein.

Das bedeutet:

- Für jeden Testmonat wird nur mit Daten trainiert, die davor lagen.
- Kein Lookahead-Bias.

Für einen 18-Monate-Walk-Forward-Backtest mit 6 Monaten Trainingsfenster werden mindestens 24 Monate Kursdaten benötigt.

Da ungefähr 2.5 Jahre Daten geladen werden sollen, ist genug Puffer vorhanden.

Beispiel:

Monat 1:
Trainiere auf vorherige 6 Monate.
Teste auf nächsten Monat.

Monat 2:
Trainiere wieder auf die letzten 6 Monate.
Teste auf nächsten Monat.

Monat 3:
Trainiere wieder auf die letzten 6 Monate.
Teste auf nächsten Monat.

Optional soll zusätzlich geprüft werden, ob andere Trainingsfenster stabiler sind:

- 3 Monate
- 6 Monate
- 9 Monate
- 12 Monate

Der Standard bleibt aber erstmal:

- Trainingsfenster = 6 Monate
- Re-Training = monatlich
- Backtestzeitraum = 18 Monate

Die Entscheidung soll nicht nur nach höchstem Gewinn getroffen werden, sondern nach robuster Out-of-Sample-Performance.

Wichtige Kriterien:

- Profit Factor
- Max Drawdown
- Winrate
- durchschnittlicher Trade nach Kosten
- Stabilität über Monate
- Anzahl Trades
- Performance mit und ohne unabhängigen AI-Filter
- Performance mit und ohne LTF-Filter
- Trendfolge separat
- Mean Reversion separat

---

# 23. Logging

Jeder genommene und jeder abgelehnte Trade soll detailliert geloggt werden.

Besonders wichtig sind:

- Trades, die durch die unabhängige AI blockiert wurden
- Trades, die durch den 1m-LTF-Filter blockiert wurden
- Mean-Reversion-Trades
- Earnings-Trades

Wichtige Log-Felder:

- trade_id
- symbol
- side
- strategy_type
- entry_time
- exit_time
- entry_price
- exit_price
- stop_price
- target_price
- tp_method
- exit_reason
- position_size
- gross_pnl
- net_pnl
- gross_pnl_pct
- net_pnl_pct
- fees
- slippage
- main_ai_probability
- independent_ai_probability
- expected_move_pct
- is_earnings_day
- vwap_value
- price_distance_to_vwap
- rsi_value
- momentum_value
- nearest_support
- nearest_resistance
- stop_distance_pct
- time_of_day
- main_ai_approved
- independent_ai_approved
- independent_ai_blocked
- ltf_filter_enabled
- ltf_confirmed
- ltf_rejected
- ltf_rejection_reason
- rule_rejections

Alle abgelehnten Trades sollen ebenfalls gespeichert werden, damit später geprüft werden kann, ob die Filter sinnvoll waren.

---

# 24. Backtest-Auswertung

Die Backtest-Auswertung soll mindestens folgende Gruppen getrennt betrachten:

- Gesamtperformance
- Trendfolge-Trades
- Mean-Reversion-Trades
- Long-Trades
- Short-Trades
- Earnings-Trades
- Non-Earnings-Trades
- Trades, die beide AI-Modelle bestätigt haben
- Trades, die Haupt-AI genommen hätte, aber unabhängige AI blockiert hat
- Trades, die durch den 1m-LTF-Filter blockiert wurden
- Performance ohne unabhängigen AI-Filter
- Performance mit unabhängigem AI-Filter
- Performance ohne 1m-LTF-Filter
- Performance mit kombiniertem 1m-LTF-Filter
- Performance Variante A
- Performance Variante D

Besonders wichtige Fragen:

- Bringt das unabhängige AI-Modell echten Mehrwert?
- Blockiert das unabhängige AI-Modell zu viele Gewinner?
- Verbessert der 1m-LTF-Filter die Entry-Qualität?
- Blockiert der 1m-LTF-Filter zu viele Gewinner?
- Funktioniert Mean Reversion oder sollte sie deaktiviert werden?
- Funktionieren Longs und Shorts gleich gut?
- Sind Earnings-Trades trotz strenger Regeln sinnvoll?
- Welches Trainingsfenster ist am stabilsten?

---

# 25. Optionaler Audit-/Logbuch-Agent

Optional soll später ein Audit-/Logbuch-Agent eingebaut werden.

Dieser Agent darf keine Trades freigeben, blockieren oder ausführen.

Er ist nur ein intelligentes Logbuch und Analysewerkzeug.

Aufgaben des Audit-/Logbuch-Agenten:

- erklären, warum Trades genommen wurden
- erklären, warum Trades abgelehnt wurden
- besonders unabhängige-AI-Blockierungen analysieren
- besonders LTF-Blockierungen analysieren
- Tagesberichte erstellen
- Backtest-Auswertung kommentieren
- Regelverstöße erkennen
- Live-Verhalten mit Backtest-Verhalten vergleichen

Die eigentliche Entscheidungsgewalt liegt nicht beim Audit-Agenten, sondern bei:

Strategie-Setup
+ Haupt-AI-Modell
+ unabhängiges AI-Modell
+ harte Risiko- und Zeitregeln
+ optionaler 1m-LTF-Filter

Der Agent ist also eher ein intelligentes Logbuch, nicht Teil der finalen Entscheidungslogik.

---

# 26. Konfigurierbarkeit

Das gesamte System soll später so gebaut werden, dass alle Regeln konfigurierbar sind.

Wichtige Konfigurationswerte:

main_ai_threshold_normal = 0.70
main_ai_threshold_earnings = 0.85

independent_ai_threshold_normal = 0.60
independent_ai_threshold_earnings = 0.70

min_expected_move_normal_pct = 0.20
min_expected_move_earnings_pct = 0.30

max_stop_distance_pct = 2.00
max_daily_loss_pct = 3.50
minimum_position_size_pct = 5.00

fee_pct = 0.005
slippage_bps = 3

timeframe_signal = 5m
timeframe_raw_data = 1m

use_ltf_confirmation = true/false
ltf_use_micro_structure = true/false
ltf_use_bad_candle_filter = true/false

training_window_months = 6
retrain_frequency = monthly
backtest_period_months = 18

Dadurch sollen Schwellenwerte wie 70 %, 60 %, 85 %, 0.2 %, 0.3 %, 2 % Stop-Distanz, 3.5 % Tagesverlust, 5 % Mindestpositionsgröße, Gebühren, Slippage, Trainingsfenster und LTF-Filter im Backtest leicht angepasst und optimiert werden können.

---

# 27. Kompakte finale Zusammenfassung

Das System soll langfristig so funktionieren:

Polygon/Massive 1m-Daten per API downloaden
→ lokal als Parquet speichern
→ reguläre Wall-Street-Session filtern
→ eigene 5m-Kerzen bauen
→ 2-Session-VWAP, RSI, Momentum, Support/Resistance berechnen
→ 5m-Setup erkennen
→ Setup-Typ bestimmen: Trend oder Mean Reversion, Long oder Short
→ Haupt-AI bewertet Wahrscheinlichkeit und expected move
→ unabhängige AI bewertet Marktsituation separat
→ Earnings-Sonderregeln prüfen
→ Mindestmove prüfen
→ Stop-Distanz prüfen
→ FTMO-Tagesverlust prüfen
→ Positionsgröße berechnen und ggf. reduzieren
→ Mindestpositionsgröße 5 % prüfen
→ optional 1m-LTF-Filter Variante D prüfen
→ Trade eröffnen oder ablehnen
→ TP dynamisch setzen
→ Trade mit Stop, Teilverkauf, Trailing, VWAP-/Momentum-Exit managen
→ spätestens 10 Minuten vor Marktschluss schließen
→ genommene und abgelehnte Trades vollständig loggen
→ unabhängige-AI-Blockierungen besonders markieren
→ LTF-Blockierungen besonders markieren
→ Walk-Forward-Backtest über 18 Monate
→ monatliches Re-Training mit standardmäßig 6 Monaten Trainingsdaten
→ Auswertung mit/ohne unabhängige AI und mit/ohne 1m-LTF-Filter

---

# 28. Wichtig für die Umsetzung in Claude Code

Bitte nicht sofort das komplette Trading-System bauen.

Bitte in Etappen vorgehen.

## Etappe 1: Nur Datenpipeline

Zuerst bauen:

1. requirements.txt
2. config/settings.yaml
3. .gitignore
4. scripts/download_polygon_1m.py
5. scripts/build_5m_bars.py
6. scripts/validate_data.py

Anforderungen an download_polygon_1m.py:

- API-Key aus .env lesen
- Symbole aus data/metadata/symbol_universe.csv lesen
- Polygon/Massive 1-Minuten-Aggregates per API laden
- Daten in Monatsblöcken laden
- Downloads fortsetzbar machen
- Fehler loggen
- Daten als Parquet speichern
- pro Symbol eine Datei erzeugen

Anforderungen an build_5m_bars.py:

- 1-Minuten-Parquet-Dateien laden
- Zeitstempel korrekt behandeln
- auf America/New_York konvertieren
- nur 09:30 bis 16:00 New York Time verwenden
- eigene 5-Minuten-Kerzen bauen
- OHLCV korrekt aggregieren
- 5-Minuten-Daten als Parquet speichern

Anforderungen an validate_data.py:

- existieren 1-Minuten-Dateien?
- existieren 5-Minuten-Dateien?
- sind die Zeitstempel plausibel?
- liegen die 5-Minuten-Kerzen innerhalb regulärer Session?
- gibt es offensichtliche Datenlücken?
- Anzahl Zeilen pro Symbol anzeigen
- kurze Zusammenfassung ausgeben

## Etappe 2: Indikatoren

Erst wenn die Datenpipeline funktioniert:

- 2-Session-VWAP
- RSI
- Momentum
- ATR
- Support/Resistance
- Feature-Building

## Etappe 3: Regelbasierter Backtester ohne AI

Erst wenn Indikatoren funktionieren:

- Setups erkennen
- Trend Long/Short
- Mean Reversion Long/Short
- Entry/Exit
- Stop
- TP
- Kosten
- Slippage
- Marktschluss-Regeln
- Logging

## Etappe 4: AI-Trainingsdaten

Dann:

- Features speichern
- Labels erzeugen
- normaler Mindestmove 0.2 %
- Earnings-Mindestmove 0.3 %
- Label 1 = Mindestmove wird vor Stop/Zeit-Exit/Gegensignal erreicht
- Label 0 = Stop/Zeit-Exit/Gegensignal zuerst

## Etappe 5: AI-Modelle

Dann:

- Haupt-AI-Modell
- unabhängiges zweites AI-Modell
- unterschiedliche Feature-Sets
- Walk-Forward-Training
- monatliches Retraining

## Etappe 6: Voller 18-Monate-Walk-Forward-Backtest

Dann:

- 18 Monate Testzeitraum
- 6 Monate rollendes Trainingsfenster
- Vergleich mit und ohne unabhängige AI
- Vergleich Variante A und Variante D beim 1m-LTF-Filter
- Mean Reversion separat
- Earnings separat
- Long/Short separat