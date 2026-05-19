
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import kruskal, mannwhitneyu

st.set_page_config(page_title="Dashboard Simulasi Waris Islam", page_icon="⚖️", layout="wide")

# ---------------------------------------------------------------------
# Theme-aware colors
# ---------------------------------------------------------------------
def is_dark_mode():
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return True

DARK = is_dark_mode()

if DARK:
    BG = "#111111"
    CARD = "#1A1A1A"
    TEXT = "#F2F2F2"
    MUTED = "#B6B6B6"
    GRID = "#2A2A2A"
    INPUT = "#222222"
else:
    BG = "#F8F5F6"
    CARD = "#FFFFFF"
    TEXT = "#24161A"
    MUTED = "#6E5A5F"
    GRID = "#E7DCDD"
    INPUT = "#FFFFFF"

MAROON = "#7B1E2B"
MAROON_2 = "#5E1620"
MAROON_3 = "#A33A4A"
GREEN = "#2E7D32"
BLUE = "#1565C0"
ORANGE = "#EF6C00"
PURPLE = "#6A1B9A"
RED = "#C62828"
GRAY = "#7A7A7A"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {MAROON_2}, {MAROON}); color: white; }}
    .block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; }}
    .title-card {{
        background: linear-gradient(135deg, {MAROON_2}, {MAROON});
        color: white; padding: 1.15rem 1.35rem; border-radius: 18px;
        box-shadow: 0 10px 25px rgba(123, 30, 43, 0.18);
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .sub-card {{
        background: {CARD}; color: {TEXT}; padding: 1rem 1.2rem; border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05); border: 1px solid {'#2A2A2A' if DARK else '#eadfe1'};
    }}
    .metric-box {{
        background: {CARD}; color: {TEXT}; padding: 0.9rem 1rem; border-radius: 14px;
        border-left: 6px solid {MAROON}; box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        border: 1px solid {'#2A2A2A' if DARK else '#eadfe1'};
    }}
    .small-note {{ color: {MUTED}; font-size: 0.92rem; }}
    div[data-testid="stDataFrame"] {{ background: {CARD}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-card">
        <h1 style="margin:0;">⚖️ Dashboard Simulasi Dinamika Konflik Waris Islam</h1>
        <p style="margin:0.35rem 0 0 0; opacity:0.95;">Agent-Based Modeling • Konseptualisasi • Koping • Monte Carlo • Analitik</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Dashboard ini otomatis membaca hasil Monte Carlo dari CSV dan menyesuaikan kontras saat mode gelap/terang berubah.")

# ---------------------------------------------------------------------
# Load CSV with fallback
# ---------------------------------------------------------------------
def load_results():
    candidates = [
        Path("hasil_monte_carlo.csv"),
        Path(__file__).with_name("hasil_monte_carlo.csv") if "__file__" in globals() else Path("hasil_monte_carlo.csv"),
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "Skenario" in df.columns:
                    return df, str(p)
            except Exception:
                pass
    return pd.DataFrame([
        {"Skenario": "Tanpa Koping", "T_recover_mean": 80.000, "T_recover_std": 0.000, "T_recover_p10": 80.000, "T_recover_p90": 80.000, "K_final_mean": 0.988, "K_final_std": 0.023, "K_max_mean": 1.000, "P_selesai": 0.000},
        {"Skenario": "Mediasi Preventif", "T_recover_mean": 72.249, "T_recover_std": 5.156, "T_recover_p10": 68.333, "T_recover_p90": 76.667, "K_final_mean": 0.229, "K_final_std": 0.123, "K_max_mean": 1.000, "P_selesai": 0.333},
        {"Skenario": "Mediasi Reaktif", "T_recover_mean": 65.890, "T_recover_std": 7.629, "T_recover_p10": 56.000, "T_recover_p90": 76.333, "K_final_mean": 0.634, "K_final_std": 0.067, "K_max_mean": 0.942, "P_selesai": 0.000},
        {"Skenario": "Mediator Eksternal", "T_recover_mean": 34.152, "T_recover_std": 6.799, "T_recover_p10": 25.667, "T_recover_p90": 42.700, "K_final_mean": 0.000, "K_final_std": 0.000, "K_max_mean": 1.000, "P_selesai": 1.000},
        {"Skenario": "Pendekatan Spiritual", "T_recover_mean": 54.652, "T_recover_std": 1.992, "T_recover_p10": 52.000, "T_recover_p90": 57.000, "K_final_mean": 0.000, "K_final_std": 0.000, "K_max_mean": 1.000, "P_selesai": 1.000},
    ]), "fallback"

summary_df, source_info = load_results()

# Normalize columns if necessary
expected_cols = ["Skenario", "T_recover_mean", "T_recover_std", "T_recover_p10", "T_recover_p90", "K_final_mean", "K_final_std", "K_max_mean", "P_selesai"]
for c in expected_cols:
    if c not in summary_df.columns:
        summary_df[c] = np.nan
summary_df = summary_df[expected_cols]

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.markdown("## Navigasi")
view = st.sidebar.radio("Pilih tampilan:", ["Ringkasan", "Visualisasi", "Analisis Statistik", "Data Mentah"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Kontrol Tampilan")
show_threshold = st.sidebar.checkbox("Tampilkan ambang konflik (0.7)", value=True)
show_values = st.sidebar.checkbox("Tampilkan label angka pada grafik", value=True)
show_theme_info = st.sidebar.checkbox("Tampilkan info theme", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### Sumber Data")
st.sidebar.write(f"- CSV: `{source_info}`")
st.sidebar.write("- Fallback: aktif bila CSV tidak ditemukan")

if show_theme_info:
    st.sidebar.write(f"- Mode terdeteksi: {'Dark' if DARK else 'Light'}")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def metric_card(label, value, delta=None):
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric(label, value, delta)
    st.markdown('</div>', unsafe_allow_html=True)

color_map = {
    "Tanpa Koping": GRAY,
    "Mediasi Preventif": MAROON,
    "Mediasi Reaktif": BLUE,
    "Mediator Eksternal": ORANGE,
    "Pendekatan Spiritual": PURPLE,
}

best_recover = summary_df.loc[summary_df["T_recover_mean"].idxmin(), "Skenario"]
best_finish = summary_df.loc[summary_df["K_final_mean"].idxmin(), "Skenario"]
best_success = summary_df.loc[summary_df["P_selesai"].idxmax(), "Skenario"]

plot_bg = CARD
paper_bg = CARD
font_color = TEXT

# ---------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------
if view == "Ringkasan":
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Skenario Tercepat", best_recover)
    with c2: metric_card("K Final Terendah", best_finish)
    with c3: metric_card("P Selesai Tertinggi", best_success)
    with c4: metric_card("Iterasi Monte Carlo", "1000")

    st.markdown('<div class="sub-card">', unsafe_allow_html=True)
    st.subheader("Inti Temuan")
    st.write(
        "Mediator Eksternal adalah skenario paling cepat menurunkan konflik, sedangkan Pendekatan Spiritual "
        "memberikan hasil yang paling konsisten. Tanpa koping menghasilkan konflik tertinggi dan tidak pernah "
        "pulih sepenuhnya."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=summary_df["Skenario"], y=summary_df["T_recover_mean"], name="T_recover_mean", marker_color=MAROON))
        fig.add_trace(go.Bar(x=summary_df["Skenario"], y=summary_df["K_final_mean"], name="K_final_mean", marker_color=ORANGE))
        fig.add_trace(go.Bar(x=summary_df["Skenario"], y=summary_df["P_selesai"], name="P_selesai", marker_color=GREEN))
        if show_values:
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig.update_layout(
            barmode="group",
            title="Perbandingan Metrik Utama",
            height=520,
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            font=dict(color=font_color),
            legend_title_text="Metrik",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        if show_threshold:
            fig.add_hline(y=0.7, line_dash="dash", line_color=RED)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=summary_df["Skenario"],
            y=summary_df["T_recover_mean"],
            mode='lines+markers+text' if show_values else 'lines+markers',
            text=[f'{v:.1f}' for v in summary_df["T_recover_mean"]] if show_values else None,
            textposition='top center',
            line=dict(color=MAROON, width=3),
            marker=dict(size=10),
            name="T_recover_mean"
        ))
        if show_threshold:
            fig2.add_hline(y=0.7, line_dash="dash", line_color=RED)
        fig2.update_layout(
            title="Rata-rata Waktu Pemulihan",
            height=520,
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            font=dict(color=font_color),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sub-card">', unsafe_allow_html=True)
    st.subheader("Ringkasan Cepat")
    a, b, c = st.columns(3)
    a.markdown(f"**Tercepat:** {best_recover}")
    b.markdown(f"**Paling Stabil:** {best_success}")
    c.markdown(f"**Risiko Tertinggi:** Tanpa Koping")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------
elif view == "Visualisasi":
    st.subheader("Visualisasi Hasil Simulasi")
    tab1, tab2, tab3 = st.tabs(["Monte Carlo", "Single Run", "Konseptualisasi"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(
                summary_df.melt(id_vars="Skenario", value_vars=["T_recover_mean", "K_final_mean", "K_max_mean"], var_name="Metrik", value_name="Nilai"),
                x="Skenario", y="Nilai", color="Metrik",
                color_discrete_sequence=[MAROON, ORANGE, BLUE],
                title="Ringkasan Distribusi Metrik"
            )
            fig.update_layout(height=520, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, font=dict(color=font_color))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(
                summary_df,
                x="Skenario", y="P_selesai", color="Skenario",
                color_discrete_sequence=[MAROON, BLUE, ORANGE, PURPLE, GREEN],
                text="P_selesai",
                title="Probabilitas Resolusi (P_selesai)"
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(height=520, showlegend=False, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, font=dict(color=font_color))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("Di laporan, bagian ini cocok untuk grafik 4 skenario koping dan grafik efektivitas single run.")
        fig = go.Figure()
        for idx, sc in enumerate(summary_df["Skenario"]):
            fig.add_trace(go.Bar(
                name=sc,
                x=["T_rec_mean", "K_final_mean", "K_max_mean"],
                y=[summary_df.loc[idx, "T_recover_mean"], summary_df.loc[idx, "K_final_mean"], summary_df.loc[idx, "K_max_mean"]],
                marker_color=color_map.get(sc, GRAY)
            ))
        fig.update_layout(barmode='group', title='Perbandingan Metrik Utama per Skenario', height=500,
                          plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, font=dict(color=font_color))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.info("State chart dan tabel atribut agen paling pas ditaruh di BAB III sebagai visual konseptual.")
        fig = go.Figure()
        fig.add_annotation(text="Gunakan gambar state_chart.png dan tabel_atribut.png di laporan", x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=MAROON))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title='Konseptualisasi', height=360, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, font=dict(color=font_color))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Analisis Statistik
# ---------------------------------------------------------------------
elif view == "Analisis Statistik":
    st.subheader("Analisis Statistik")
    st.write("Berdasarkan output Monte Carlo, perbedaan antar skenario sangat signifikan.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Kruskal-Wallis H", "4501.846")
        st.metric("p-value", "0.00e+00")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Kesimpulan", "Signifikan")
        st.metric("Alpha", "0.05")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Interpretasi")
    st.write(
        "Seluruh pasangan skenario pada uji post-hoc Mann-Whitney U menunjukkan p-value sangat kecil, "
        "yang berarti distribusi waktu pemulihan antar skenario berbeda secara statistik."
    )

    fig = go.Figure()
    for sc in summary_df["Skenario"]:
        mean_val = summary_df.loc[summary_df['Skenario'] == sc, 'T_recover_mean'].values[0]
        std_val = max(summary_df.loc[summary_df['Skenario'] == sc, 'T_recover_std'].values[0], 0.8)
        y = np.random.normal(mean_val, std_val, 120)
        fig.add_trace(go.Box(y=y, name=sc, marker_color=color_map.get(sc, GRAY)))
    fig.update_layout(title='Ilustrasi Distribusi Waktu Pemulihan', height=520, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, font=dict(color=font_color))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabel Ringkasan")
    st.dataframe(
        summary_df.style.format({
            "T_recover_mean": "{:.3f}",
            "T_recover_std": "{:.3f}",
            "T_recover_p10": "{:.3f}",
            "T_recover_p90": "{:.3f}",
            "K_final_mean": "{:.3f}",
            "K_final_std": "{:.3f}",
            "K_max_mean": "{:.3f}",
            "P_selesai": "{:.3f}",
        }),
        use_container_width=True
    )

# ---------------------------------------------------------------------
# Data Mentah
# ---------------------------------------------------------------------
elif view == "Data Mentah":
    st.subheader("Data Ringkasan dari CSV")
    st.dataframe(summary_df, use_container_width=True)
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name="hasil_monte_carlo_ringkas.csv", mime="text/csv")

    st.markdown("### Catatan Penempatan Visual dalam Laporan")
    st.write("- State chart: BAB III, bagian metodologi/konseptualisasi.")
    st.write("- Tabel atribut: BAB III, setelah penjelasan state chart.")
    st.write("- Grafik 4 skenario koping: BAB IV, hasil simulasi single run.")
    st.write("- Grafik Monte Carlo: BAB IV, setelah pembahasan simulasi iteratif.")
    st.write("- Grafik analisis statistik: BAB IV, bagian analitik dan trade-off.")

st.markdown("---")
st.caption("Streamlit Dashboard • Tema Maroon • ABM Waris Islam")
