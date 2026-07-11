import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import kruskal, mannwhitneyu

st.set_page_config(
    page_title="Simulasi Waris Islam — Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# PALET WARNA — Solid Maroon Theme (akademik, minim aksen)
# =============================================================
MAROON  = "#7B1E2B"
GRAY    = "#6b6b6b"
LIGHTBG = "#f5e9ea"

SCENARIO_COLORS = {
    "Tanpa Koping":         "#6b6b6b",
    "Mediasi Preventif":    "#7B1E2B",
    "Mediasi Reaktif":      "#1565C0",
    "Mediator Eksternal":   "#B4682E",
    "Pendekatan Spiritual": "#5A3E85",
}

RAW_COLS_MIN = {"skenario", "iterasi", "t_recover", "k_final", "k_max", "selesai"}

# =============================================================
# STYLE — mengikuti tema dashboard data mining (solid maroon,
# hero-header, metric-card, tanpa emoji, tipografi konsisten)
# =============================================================
st.markdown(f"""
<style>
  [data-testid="stSidebar"] {{
      background: {MAROON};
  }}
  [data-testid="stSidebar"] * {{ color: #fff !important; }}
  [data-testid="stSidebar"] .stRadio label {{ color: #fff !important; }}

  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
      color: {MAROON} !important;
      font-weight: 700 !important;
  }}
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
  .hero-header h1, .hero-header h2, .hero-header h3 {{
      color: white !important;
  }}

  .hero-header {{
      background: {MAROON};
      color: white;
      padding: 28px 32px;
      border-radius: 12px;
      margin-bottom: 24px;
  }}
  .hero-header h1 {{ margin: 0 0 6px 0; font-size: 26px; }}
  .hero-header p  {{ color: #e7c9cd; margin: 0; font-size: 14px; opacity: 0.95; }}

  .metric-card {{
      background: white;
      border-radius: 10px;
      padding: 18px 20px;
      border-left: 5px solid {MAROON};
      box-shadow: 0 2px 8px rgba(0,0,0,.08);
      margin-bottom: 10px;
  }}
  .metric-label {{ font-size: 12px; color: {GRAY}; text-transform: uppercase; font-weight: 600; }}
  .metric-value {{ font-size: 24px; font-weight: 700; color: {MAROON}; }}

  .section-title {{
      font-size: 18px; font-weight: 700;
      color: {MAROON}; margin-bottom: 12px;
      border-bottom: 2px solid {MAROON}; padding-bottom: 4px;
  }}

  .info-card {{
      background: {LIGHTBG};
      border: 1px solid #e0c3c6;
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 8px;
      font-size: 14px;
      color: #333;
  }}

  .badge-source {{
      background: {MAROON}; color: white;
      padding: 2px 10px; border-radius: 20px;
      font-size: 11px; font-weight: bold;
  }}

  div[data-testid="stMetric"] {{
      background: white;
      border-radius: 8px;
      padding: 12px 16px;
      border-left: 4px solid {MAROON};
      box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-header">
  <h1>Dashboard Simulasi Dinamika Konflik Waris Islam</h1>
  <p>Agent-Based Modeling · Simulasi Monte Carlo · Analisis Statistik · Universitas Muhammadiyah Malang</p>
</div>
""", unsafe_allow_html=True)

# =============================================================
# LOAD DATA (Ringkasan + opsional Data Mentah)
# =============================================================
def _norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_summary():
    if "uploaded_summary" in st.session_state and st.session_state.uploaded_summary is not None:
        return st.session_state.uploaded_summary, "upload"
    try:
        script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        csv_path = script_dir / "hasil_monte_carlo.csv"
        if csv_path.exists():
            df = _norm_cols(pd.read_csv(csv_path))
            if "Skenario" in df.columns:
                return df, str(csv_path)
    except Exception as e:
        st.warning(f"Gagal membaca CSV ringkasan lokal: {e}")

    df = pd.DataFrame([
        {"Skenario": "Tanpa Koping", "T_recover_mean": 80.000, "T_recover_std": 0.000, "T_recover_p10": 80.000, "T_recover_p90": 80.000, "K_final_mean": 0.988, "K_final_std": 0.023, "K_max_mean": 1.000, "P_selesai": 0.000},
        {"Skenario": "Mediasi Preventif", "T_recover_mean": 72.249, "T_recover_std": 5.156, "T_recover_p10": 68.333, "T_recover_p90": 76.667, "K_final_mean": 0.229, "K_final_std": 0.123, "K_max_mean": 1.000, "P_selesai": 0.333},
        {"Skenario": "Mediasi Reaktif", "T_recover_mean": 65.890, "T_recover_std": 7.629, "T_recover_p10": 56.000, "T_recover_p90": 76.333, "K_final_mean": 0.634, "K_final_std": 0.067, "K_max_mean": 0.942, "P_selesai": 0.000},
        {"Skenario": "Mediator Eksternal", "T_recover_mean": 34.152, "T_recover_std": 6.799, "T_recover_p10": 25.667, "T_recover_p90": 42.700, "K_final_mean": 0.000, "K_final_std": 0.000, "K_max_mean": 1.000, "P_selesai": 1.000},
        {"Skenario": "Pendekatan Spiritual", "T_recover_mean": 54.652, "T_recover_std": 1.992, "T_recover_p10": 52.000, "T_recover_p90": 57.000, "K_final_mean": 0.000, "K_final_std": 0.000, "K_max_mean": 1.000, "P_selesai": 1.000},
    ])
    return df, "fallback"

def load_raw():
    """Dataset mentah (per-iterasi) — opsional. Dipakai untuk histogram distribusi."""
    if "uploaded_raw" in st.session_state and st.session_state.uploaded_raw is not None:
        return st.session_state.uploaded_raw
    try:
        script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        csv_path = script_dir / "dataset_monte_carlo_mentah.csv"
        if csv_path.exists():
            df = _norm_cols(pd.read_csv(csv_path))
            if {"skenario", "T_recover"}.issubset(set(df.columns)) or {"skenario", "t_recover"}.issubset(set(c.lower() for c in df.columns)):
                return df
    except Exception:
        pass
    return None

def detect_csv_type(df):
    """Mendeteksi apakah CSV yang diupload adalah dataset RINGKASAN atau MENTAH."""
    cols_lower = {c.strip().lower() for c in df.columns}
    if "skenario" in cols_lower and "iterasi" in cols_lower:
        return "raw"
    if "skenario" in cols_lower and "t_recover_mean" in cols_lower:
        return "summary"
    return "unknown"

# =============================================================
# SIDEBAR
# =============================================================
st.sidebar.markdown("## Navigasi")
view = st.sidebar.radio("Pilih tampilan:", ["Ringkasan", "Visualisasi", "Analisis Statistik", "Data Mentah"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Unggah Dataset (CSV)")
st.sidebar.caption("Terima dataset ringkasan (5 baris/skenario) maupun dataset mentah (1000 iterasi/skenario).")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df_upload = _norm_cols(pd.read_csv(uploaded_file))
        csv_type = detect_csv_type(df_upload)
        if csv_type == "summary":
            st.session_state.uploaded_summary = df_upload
            st.sidebar.success("Dataset ringkasan berhasil diunggah.")
        elif csv_type == "raw":
            # samakan nama kolom jadi title-case standar
            rename_map = {c: c for c in df_upload.columns}
            for c in df_upload.columns:
                if c.strip().lower() == "skenario":
                    rename_map[c] = "skenario"
                if c.strip().lower() == "t_recover":
                    rename_map[c] = "T_recover"
                if c.strip().lower() == "k_final":
                    rename_map[c] = "K_final"
                if c.strip().lower() == "k_max":
                    rename_map[c] = "K_max"
            df_upload = df_upload.rename(columns=rename_map)
            st.session_state.uploaded_raw = df_upload
            st.sidebar.success(f"Dataset mentah berhasil diunggah ({len(df_upload):,} baris).")
        else:
            st.sidebar.error(
                "Kolom yang diharapkan tidak ditemukan. Dataset ringkasan harus memiliki kolom "
                "'Skenario' dan 'T_recover_mean'; dataset mentah harus memiliki kolom 'skenario' dan 'iterasi'."
            )
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Kontrol Tampilan")
show_threshold = st.sidebar.checkbox("Tampilkan ambang konflik (0.7)", value=True)
show_values = st.sidebar.checkbox("Tampilkan label angka pada grafik", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Sumber Data")
summary_df, source = load_summary()
raw_df = load_raw()

if source == "upload":
    st.sidebar.success("Ringkasan: dari data yang diunggah")
elif source == "fallback":
    st.sidebar.info("Ringkasan: data bawaan (CSV tidak ditemukan)")
else:
    st.sidebar.info(f"Ringkasan: `{Path(source).name}`")

if raw_df is not None:
    st.sidebar.success(f"Data mentah tersedia ({len(raw_df):,} baris)")
else:
    st.sidebar.info("Data mentah belum diunggah")

# Normalisasi kolom ringkasan
expected_cols = ["Skenario", "T_recover_mean", "T_recover_std", "T_recover_p10", "T_recover_p90", "K_final_mean", "K_final_std", "K_max_mean", "P_selesai"]
for c in expected_cols:
    if c not in summary_df.columns:
        summary_df[c] = np.nan
summary_df = summary_df[expected_cols]

# =============================================================
# HELPER
# =============================================================
def metric_card(label, value, delta=None):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

best_recover = summary_df.loc[summary_df["T_recover_mean"].idxmin(), "Skenario"] if not summary_df.empty else "-"
best_finish = summary_df.loc[summary_df["K_final_mean"].idxmin(), "Skenario"] if not summary_df.empty else "-"
best_success = summary_df.loc[summary_df["P_selesai"].idxmax(), "Skenario"] if not summary_df.empty else "-"

# =============================================================
# RINGKASAN
# =============================================================
if view == "Ringkasan":
    if summary_df.empty:
        st.error("Tidak ada data untuk ditampilkan. Silakan unggah CSV.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Skenario Tercepat", best_recover)
        with c2: metric_card("K Final Terendah", best_finish)
        with c3: metric_card("P Selesai Tertinggi", best_success)
        with c4: metric_card("Iterasi Monte Carlo", "1.000")

        st.markdown('<p class="section-title">Inti Temuan</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
        Mediator Eksternal adalah skenario paling cepat menurunkan konflik, sedangkan Pendekatan Spiritual
        memberikan hasil yang paling konsisten. Tanpa koping menghasilkan konflik tertinggi dan tidak pernah
        pulih sepenuhnya.
        </div>
        """, unsafe_allow_html=True)

        left, right = st.columns([1.15, 0.85])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=summary_df["Skenario"], y=summary_df["T_recover_mean"], name="T_recover_mean", marker_color=MAROON))
            fig.add_trace(go.Bar(x=summary_df["Skenario"], y=summary_df["K_final_mean"], name="K_final_mean", marker_color="#B4682E"))
            fig.add_trace(go.Bar(x=summary_df["Skenario"], y=summary_df["P_selesai"], name="P_selesai", marker_color="#5A3E85"))
            if show_values:
                fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
            fig.update_layout(
                barmode="group",
                title="Perbandingan Metrik Utama",
                height=480,
                plot_bgcolor="white",
                legend_title_text="Metrik",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            if show_threshold:
                fig.add_hline(y=0.7, line_dash="dash", line_color="#A13544")
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
                fig2.add_hline(y=0.7, line_dash="dash", line_color="#A13544")
            fig2.update_layout(
                title="Rata-rata Waktu Pemulihan",
                height=480,
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-title">Ringkasan Cepat</p>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a: st.markdown(f"<div class='info-card'><b>Tercepat:</b><br>{best_recover}</div>", unsafe_allow_html=True)
        with b: st.markdown(f"<div class='info-card'><b>Paling Stabil:</b><br>{best_success}</div>", unsafe_allow_html=True)
        with c: st.markdown(f"<div class='info-card'><b>Risiko Tertinggi:</b><br>Tanpa Koping</div>", unsafe_allow_html=True)

# =============================================================
# VISUALISASI
# =============================================================
elif view == "Visualisasi":
    if summary_df.empty:
        st.error("Tidak ada data untuk ditampilkan. Silakan unggah CSV.")
    else:
        st.markdown('<p class="section-title">Visualisasi Hasil Simulasi</p>', unsafe_allow_html=True)
        tab_labels = ["Ringkasan Monte Carlo", "Distribusi (Data Mentah)", "Single Run", "Konseptualisasi"]
        tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.box(
                    summary_df.melt(id_vars="Skenario", value_vars=["T_recover_mean", "K_final_mean", "K_max_mean"], var_name="Metrik", value_name="Nilai"),
                    x="Skenario", y="Nilai", color="Metrik",
                    color_discrete_sequence=[MAROON, "#B4682E", "#1565C0"],
                    title="Ringkasan Distribusi Metrik"
                )
                fig.update_layout(height=480, plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(
                    summary_df,
                    x="Skenario", y="P_selesai", color="Skenario",
                    color_discrete_map=SCENARIO_COLORS,
                    text="P_selesai",
                    title="Probabilitas Resolusi (P_selesai)"
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(height=480, showlegend=False, plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if raw_df is None:
                st.markdown(
                    "<div class='info-card'>Dataset mentah belum tersedia. Unggah file "
                    "<code>dataset_monte_carlo_mentah.csv</code> (kolom: skenario, iterasi, T_recover, "
                    "K_final, K_max, selesai) pada panel unggah di sidebar untuk melihat distribusi asli "
                    "1.000 iterasi per skenario.</div>",
                    unsafe_allow_html=True,
                )
            else:
                metric_choice = st.selectbox("Pilih metrik", ["T_recover", "K_final", "K_max"], index=0)
                fig = px.histogram(
                    raw_df, x=metric_choice, color="skenario",
                    color_discrete_map=SCENARIO_COLORS,
                    barmode="overlay", opacity=0.65, nbins=40,
                    title=f"Distribusi {metric_choice} — 1.000 Iterasi per Skenario"
                )
                fig.update_layout(height=500, plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.violin(
                    raw_df, x="skenario", y=metric_choice, color="skenario",
                    color_discrete_map=SCENARIO_COLORS, box=True,
                    title=f"Distribusi {metric_choice} per Skenario (Violin Plot)"
                )
                fig2.update_layout(height=500, plot_bgcolor="white", showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.markdown(
                "<div class='info-card'>Bagian ini menampilkan grafik perbandingan metrik utama antar "
                "skenario koping berdasarkan hasil rata-rata simulasi.</div>",
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            for idx, sc in enumerate(summary_df["Skenario"]):
                fig.add_trace(go.Bar(
                    name=sc,
                    x=["T_rec_mean", "K_final_mean", "K_max_mean"],
                    y=[summary_df.loc[idx, "T_recover_mean"], summary_df.loc[idx, "K_final_mean"], summary_df.loc[idx, "K_max_mean"]],
                    marker_color=SCENARIO_COLORS.get(sc, GRAY)
                ))
            fig.update_layout(barmode='group', title='Perbandingan Metrik Utama per Skenario', height=460, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown(
                "<div class='info-card'>State chart dan tabel atribut agen ditampilkan pada laporan "
                "(jurnal) bagian metodologi.</div>",
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            fig.add_annotation(text="Lihat state_chart.png dan tabel_atribut.png pada laporan", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=MAROON))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(title='Konseptualisasi Agen', height=320, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

# =============================================================
# ANALISIS STATISTIK
# =============================================================
elif view == "Analisis Statistik":
    if summary_df.empty:
        st.error("Tidak ada data untuk ditampilkan. Silakan unggah CSV.")
    else:
        st.markdown('<p class="section-title">Analisis Statistik</p>', unsafe_allow_html=True)
        st.markdown(
            "<div class='info-card'>Berdasarkan output Monte Carlo, perbedaan antar skenario koping "
            "signifikan secara statistik.</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            metric_card("Kruskal-Wallis H", "4501.846")
            metric_card("p-value", "&lt; 0.001")
        with c2:
            metric_card("Kesimpulan", "Signifikan")
            metric_card("Alpha", "0.05")

        st.markdown('<p class="section-title">Interpretasi</p>', unsafe_allow_html=True)
        st.markdown(
            "<div class='info-card'>Seluruh pasangan skenario pada uji post-hoc Mann-Whitney U menunjukkan "
            "p-value sangat kecil, yang berarti distribusi waktu pemulihan antar skenario berbeda secara "
            "statistik dan bukan disebabkan oleh kebetulan.</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-title">Distribusi Waktu Pemulihan</p>', unsafe_allow_html=True)
        fig = go.Figure()
        if raw_df is not None:
            for sc in summary_df["Skenario"]:
                y = raw_df.loc[raw_df["skenario"] == sc, "T_recover"]
                if len(y) > 0:
                    fig.add_trace(go.Box(y=y, name=sc, marker_color=SCENARIO_COLORS.get(sc, GRAY)))
            fig.update_layout(title='Distribusi Waktu Pemulihan (Data Mentah, N=1.000 per Skenario)', height=500, plot_bgcolor="white")
        else:
            for sc in summary_df["Skenario"]:
                mean_val = summary_df.loc[summary_df['Skenario'] == sc, 'T_recover_mean'].values[0]
                std_val = max(summary_df.loc[summary_df['Skenario'] == sc, 'T_recover_std'].values[0], 0.8)
                y = np.random.normal(mean_val, std_val, 120)
                fig.add_trace(go.Box(y=y, name=sc, marker_color=SCENARIO_COLORS.get(sc, GRAY)))
            fig.update_layout(title='Ilustrasi Distribusi Waktu Pemulihan (dari statistik ringkasan)', height=500, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        # Uji Kruskal-Wallis & Mann-Whitney langsung jika data mentah tersedia
        if raw_df is not None:
            st.markdown('<p class="section-title">Uji Signifikansi (dihitung dari data mentah)</p>', unsafe_allow_html=True)
            groups = [raw_df.loc[raw_df["skenario"] == sc, "T_recover"].values for sc in summary_df["Skenario"]]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                h_stat, p_val = kruskal(*groups)
                c1, c2 = st.columns(2)
                with c1: metric_card("Kruskal-Wallis H (aktual)", f"{h_stat:.3f}")
                with c2: metric_card("p-value (aktual)", f"{p_val:.3e}")

        st.markdown('<p class="section-title">Tabel Ringkasan</p>', unsafe_allow_html=True)
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

# =============================================================
# DATA MENTAH
# =============================================================
elif view == "Data Mentah":
    st.markdown('<p class="section-title">Data Ringkasan</p>', unsafe_allow_html=True)
    st.dataframe(summary_df, use_container_width=True)
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Unduh CSV Ringkasan", csv, file_name="hasil_monte_carlo_ringkas.csv", mime="text/csv")

    st.markdown('<p class="section-title">Data Mentah (Per Iterasi)</p>', unsafe_allow_html=True)
    if raw_df is not None:
        st.dataframe(raw_df, use_container_width=True)
        csv_raw = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button("Unduh CSV Data Mentah", csv_raw, file_name="dataset_monte_carlo_mentah.csv", mime="text/csv")
    else:
        st.markdown(
            "<div class='info-card'>Belum ada data mentah yang diunggah. Unggah file "
            "<code>dataset_monte_carlo_mentah.csv</code> melalui sidebar untuk menampilkannya di sini.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-title">Catatan Penempatan Visual dalam Laporan</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    - State chart: Bab III, bagian metodologi/konseptualisasi.<br>
    - Tabel atribut: Bab III, setelah penjelasan state chart.<br>
    - Grafik 4 skenario koping: Bab IV, hasil simulasi single run.<br>
    - Grafik Monte Carlo: Bab IV, setelah pembahasan simulasi iteratif.<br>
    - Grafik analisis statistik: Bab IV, bagian analitik dan trade-off.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Dashboard Simulasi Waris Islam — Agent-Based Modeling · Universitas Muhammadiyah Malang")
