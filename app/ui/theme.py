from __future__ import annotations

import streamlit as st


def inject_global_css() -> None:
    st.markdown(
        """
<style>
:root {
  --bg: #0b0f14;
  --card-bg: #101826;
  --text: #e6edf3;
  --muted: #8b9bb3;
  --border: rgba(255, 255, 255, 0.06);
  --buy: #00c853;
  --hold: #607d8b;
  --sell: #ef5350;
  --chip: #1a2432;
}

.stApp {
  background: radial-gradient(circle at 5% 5%, #122036 0%, var(--bg) 45%, var(--bg) 100%);
}

[data-testid="stAppViewContainer"] > .main {
  padding-top: 0.7rem;
}

[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 100%;
  padding-top: 0.45rem;
  padding-left: 0.9rem;
  padding-right: 0.9rem;
  padding-bottom: 0.8rem;
}

[data-testid="stSidebar"] .block-container {
  padding-top: 1rem;
}

.topbar {
  margin-bottom: 12px;
}

.pulse-block-gap {
  height: 12px;
}

.app-title {
  font-size: 28px;
  font-weight: 700;
  color: #e6edf3;
  margin: 0 0 8px 2px;
  letter-spacing: 0.01em;
}

.card {
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(22, 32, 48, 0.96) 0%, rgba(15, 24, 38, 0.96) 100%);
  border: 1px solid var(--border);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
  padding: 12px 14px;
}

.section-title {
  font-size: 1.35rem;
  font-weight: 800;
  letter-spacing: 0.01em;
  color: #e6edf3;
  line-height: 1.15;
}

.brain-header-price {
  font-size: 2.05rem;
  font-weight: 800;
  line-height: 1.05;
  color: #e6edf3;
  white-space: nowrap;
}

.row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.row-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.row-right {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.pill {
  display: inline-block;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 700;
  padding: 0.25rem 0.62rem;
  letter-spacing: 0.03em;
}

.pill-buy {
  color: #001b0a;
  background: linear-gradient(180deg, #21e06d 0%, #00c853 100%);
}

.pill-hold {
  color: #d6dfea;
  background: linear-gradient(180deg, #546e7a 0%, #455a64 100%);
}

.pill-sell {
  color: #fff0f0;
  background: linear-gradient(180deg, #f46a6a 0%, #ef5350 100%);
}

.muted {
  color: var(--muted);
}

.tiny {
  font-size: 0.74rem;
  line-height: 1.15;
}

.chip {
  display: inline-block;
  margin: 2px 4px 2px 0;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 0.15rem 0.5rem;
  font-size: 0.71rem;
  color: #c7d2e0;
  background: var(--chip);
}

.badge-row {
  width: 170px;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
}

.pulse-card {
  border-radius: 18px;
  padding: 16px 16px 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: linear-gradient(180deg, rgba(20, 31, 47, 0.97) 0%, rgba(14, 23, 36, 0.97) 100%);
  margin-bottom: 0;
  position: relative;
  z-index: 1;
}

.pulse-card-shell {
  border-radius: 18px;
  padding: 16px 16px 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: linear-gradient(180deg, rgba(20, 31, 47, 0.97) 0%, rgba(14, 23, 36, 0.97) 100%);
  margin-bottom: 12px;
}

.pulse-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px 12px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 12px;
  letter-spacing: 0.2px;
}

.pulse-pill-buy {
  color: #001b0a;
  background: linear-gradient(180deg, #21e06d 0%, #00c853 100%);
}

.pulse-pill-hold {
  color: #d6dfea;
  background: linear-gradient(180deg, #546e7a 0%, #455a64 100%);
}

.pulse-pill-sell {
  color: #fff0f0;
  background: linear-gradient(180deg, #f46a6a 0%, #ef5350 100%);
}

.pulse-sparkline {
  width: 170px;
  height: 52px;
}

.pulse-sparkline-wrap {
  width: 170px;
  height: 52px;
}

.stButton button[kind="tertiary"],
.stButton button[data-testid="baseButton-tertiary"] {
  font-size: 1.36rem;
  font-weight: 800;
  line-height: 1;
  padding: 0.02rem 0;
  border: 0;
  background: transparent;
  color: #e6edf3;
  justify-content: flex-start;
}

.stButton button[kind="tertiary"]:hover,
.stButton button[data-testid="baseButton-tertiary"]:hover {
  border: 0;
  background: transparent;
  color: #f1f5fb;
}

.stButton button[data-testid="baseButton-tertiary"] p,
.stButton button[data-testid="baseButton-tertiary"] span {
  font-size: 1.36rem !important;
  font-weight: 800 !important;
  line-height: 1 !important;
}

.pulse-company {
  margin-top: 2px;
  font-size: 0.98rem;
  font-weight: 600;
  color: #8ea2bb;
  line-height: 1.2;
}

.pulse-center-zone {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  min-width: 0;
}

.pulse-price-main {
  font-size: 2.05rem;
  line-height: 1.05;
  font-weight: 800;
  color: #e6edf3;
}

.pulse-quote-line {
  font-size: 1rem;
  line-height: 1.23;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pos {
  color: #21e06d;
}

.neg {
  color: #ef5350;
}

.neu {
  color: #8ea2bb;
}

.pulse-block {
  margin-top: 10px;
}

.pulse-divider {
  height: 1px;
  background: rgba(255, 255, 255, 0.08);
  margin: 12px 0 11px 0;
}

.pulse-metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 16px;
}

.pulse-metric-item {
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.05);
  padding: 7px 9px;
  min-width: 0;
}

.pulse-metric-label {
  font-size: 0.8rem;
  line-height: 1.15;
  margin-bottom: 2px;
}

.pulse-metric-value {
  font-size: 0.95rem;
  line-height: 1.18;
  color: #dbe7f4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pulse-pill-row {
  display: flex;
  align-items: center;
  gap: 8px;
  justify-content: flex-end;
  width: 170px;
}

.pulse-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 7px;
  width: 100%;
}

.stButton > button[data-testid="baseButton-secondary"] {
  font-size: 0.8rem !important;
  line-height: 1 !important;
  min-height: 1.55rem !important;
  padding: 0.08rem 0.2rem !important;
}

.stButton > button[data-testid="baseButton-secondary"] span {
  font-size: 0.8rem !important;
}

@media (max-width: 900px) {
  .pulse-right {
    width: 100%;
  }
  .pulse-sparkline {
    width: 146px;
    height: 48px;
  }
  .pulse-sparkline-wrap {
    width: 146px;
    height: 48px;
  }
  .pulse-pill-row {
    width: 146px;
  }
  .pulse-confidence-row {
    width: 146px;
  }
  .badge-row {
    width: 146px;
  }
  .pulse-quote-line {
    font-size: 0.92rem;
  }
}

.badge-age {
  font-size: 12px;
  opacity: 0.75;
}

.badge-conf {
  font-size: 12px;
  opacity: 0.8;
}

.badge-icon {
  font-size: 13px;
  opacity: 0.85;
}

.pulse-confidence-row {
  width: 170px;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
  line-height: 1.2;
}

.pulse-info-icon {
  opacity: 0.82;
  font-size: 0.78rem;
}

.pulse-trash-topgap {
  height: 12px;
}

/* Pulse strict grid overrides */
.pulse-card {
  margin-bottom: 14px;
}

.pulse-card-wrap {
  position: relative;
  margin-bottom: 14px;
}

.pulse-card-open {
  position: absolute;
  inset: 0;
  z-index: 2;
  border-radius: 18px;
  display: block;
  cursor: pointer;
}

.pulse-top {
  display: grid;
  grid-template-columns: 1.2fr 2fr 1.4fr;
  column-gap: 14px;
  align-items: start;
}

.pulse-left {
  padding-top: 2px;
  min-width: 0;
}

.pulse-ticker {
  font-weight: 700;
  font-size: 20px;
  line-height: 1.1;
  color: #e6edf3;
}

.pulse-name {
  margin-top: 10px;
  font-size: 14px;
  opacity: 0.8;
  color: #8ea2bb;
}

.pulse-mid {
  align-self: start;
  min-width: 0;
}

.pulse-price {
  font-weight: 800;
  font-size: 34px;
  line-height: 1.05;
  color: #e6edf3;
  white-space: nowrap;
}

.pulse-quote-lines {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.quote-line {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 15px;
}

.quote-line .label {
  margin-left: 6px;
}

.moon {
  margin: 0 6px 0 2px;
}

.pulse-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  justify-content: flex-start;
  min-width: 160px;
  gap: 10px;
  position: relative;
}

.pulse-pill {
  align-self: flex-end;
}

.pulse-sparkline {
  width: 140px;
  height: 54px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.pulse-sparkline svg {
  width: 140px;
  height: 54px;
}

.pulse-meta {
  width: 140px;
  text-align: left;
  align-self: flex-end;
  font-size: 13px;
  opacity: 0.85;
  line-height: 1.2;
}

.pulse-conf {
  color: #dbe7f4;
}

.pulse-age {
  margin-top: 4px;
  color: #8ea2bb;
}

.pulse-trash {
  position: absolute;
  right: 0;
  bottom: 0;
  opacity: 0.78;
}

.pulse-stats {
  margin-top: 12px;
}

.pulse-meta-line {
  font-size: 13px;
  opacity: 0.85;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #dbe7f4;
}

.pulse-meta-line .info {
  cursor: help;
  margin: 0;
  position: relative;
  z-index: 3;
}

.pulse-meta-line .trash {
  margin-left: 0;
  text-decoration: none;
  color: #dbe7f4;
  position: relative;
  z-index: 3;
}

.pulse-meta-line .trash:hover {
  color: #ffffff;
}

.horizon-card-wrap {
  position: relative;
  margin-bottom: 10px;
}

.horizon-card-open {
  position: absolute;
  inset: 0;
  z-index: 2;
  border-radius: 16px;
  display: block;
  cursor: pointer;
}

.horizon-card-row {
  padding: 12px 14px;
}

.horizon-row {
  display: grid;
  grid-template-columns: 1.55fr 1.35fr 0.85fr;
  gap: 12px;
  align-items: center;
}

.horizon-col {
  min-width: 0;
}

.horizon-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.horizon-mid {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.horizon-right {
  display: flex;
  justify-content: flex-end;
}

.horizon-ticker {
  font-size: 1.7rem;
  font-weight: 800;
  line-height: 1.05;
  color: #e6edf3;
}

.horizon-open-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 88px;
  min-height: 36px;
  padding: 0 14px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.25);
  background: rgba(10, 18, 29, 0.65);
  color: #dce6f2;
  font-weight: 700;
  font-size: 0.8rem;
}

.brain-quote-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 14px;
  margin-top: 10px;
  margin-bottom: 10px;
}

.brain-quote-cell {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 8px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.brain-quote-k {
  font-size: 12px;
  color: #8ea2bb;
}

.brain-quote-v {
  font-size: 13px;
  font-weight: 700;
  color: #dbe7f4;
}

.brain-quote-compact {
  width: 100%;
  border-collapse: collapse;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  overflow: hidden;
}

.brain-quote-compact td {
  padding: 8px 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  border-right: 1px solid rgba(255, 255, 255, 0.06);
}

.brain-quote-compact tr:last-child td {
  border-bottom: 0;
}

.brain-quote-compact td:last-child {
  border-right: 0;
}

.brain-gauges {
  margin-top: 8px;
  margin-bottom: 10px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.brain-gauge {
  padding: 8px 6px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.05);
  text-align: center;
}

.brain-gauge.gauge-good .brain-gauge-needle { background: #00c853; }
.brain-gauge.gauge-good .brain-gauge-value { color: #21e06d; }
.brain-gauge.gauge-bad .brain-gauge-needle { background: #ef5350; }
.brain-gauge.gauge-bad .brain-gauge-value { color: #ef5350; }
.brain-gauge.gauge-neutral .brain-gauge-needle { background: #8fa3b8; }
.brain-gauge.gauge-neutral .brain-gauge-value { color: #dbe7f4; }

.brain-gauge-label {
  font-size: 10px;
  letter-spacing: 0.08em;
  color: #8ea2bb;
  margin-bottom: 6px;
}

.brain-gauge-dial {
  position: relative;
  width: 82px;
  height: 41px;
  margin: 0 auto 6px auto;
  border-top-left-radius: 82px;
  border-top-right-radius: 82px;
  border: 2px solid rgba(255, 255, 255, 0.22);
  border-bottom: 0;
  overflow: hidden;
}

.brain-gauge-needle {
  position: absolute;
  left: calc(50% - 1px);
  bottom: 0;
  width: 2px;
  height: 34px;
  background: #00c853;
  transform-origin: bottom center;
}

.brain-gauge-value {
  font-size: 12px;
  font-weight: 700;
  color: #dbe7f4;
}

.why-title {
  font-size: 1.15rem;
  font-weight: 800;
  color: #e8f0fb;
  margin: 0.2rem 0 0.35rem 0;
}

.why-ai-label {
  font-size: 0.74rem;
  font-weight: 700;
  color: #8ea2bb;
  margin-left: 0.35rem;
  letter-spacing: 0.03em;
}

.why-subline {
  margin-bottom: 0.35rem;
}
</style>
""",
        unsafe_allow_html=True,
    )
