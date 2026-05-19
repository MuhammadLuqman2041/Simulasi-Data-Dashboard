
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import kruskal, mannwhitneyu

st.set_page_config(page_title="Dashboard Simulasi Waris Islam", page_icon="⚖️", layout="wide")

MAROON = "#7B1E2B"
MAROON_2 = "#5E1620"
MAROON_3 = "#A33A4A"
BG = "#F8F5F6"
CARD = "#FFFFFF"
TEXT = "#2B1A1E"
MUTED = "#6E5A5F"
GREEN = "#2E7D32"
BLUE = "#1565C0"
ORANGE = "#EF6C00"
PURPLE = "#6A1B9A"
RED = "#C62828"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {MAROON_2}, {MAROON}); color: white; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
    .title-card {{
        background: linear-gradient(135deg, {MAROON_2}, {MAROON});
        color: white; padding: 1.2rem 1.4rem; border-radius: 18px;
        box-shadow: 0 10px 25px rgba(123, 30, 43, 0.18);
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .sub-card {{
        background: {CARD}; padding: 1rem 1.2rem; border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05); border: 1px solid #eadfe1;
    }}
    .metric-box {{
        background: {CARD}; padding: 0.9rem 1rem; border-radius: 14px;
        border-left: 6px solid {MAROON}; box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    }}
    .small-note {{ color: {MUTED}; font-size: 0.92rem; }}
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

st.caption("Dashboard ini merangkum hasil simulasi konflik pembagian waris dengan warna utama maroon untuk identitas visual yang konsisten.")

# ---------------------------------------------------------------------
# Data default dari hasil simulasi yang kamu kasih
# ---------------------------------------------------------------------
summary_df = pd.DataFrame([
    {"Skenario": "Tanpa Koping", "T_rec_mean": 80.000, "T_rec_std": 0.000, "K_final": 0.988, "K_max": 1.000, "P_selesai": 0.000},
    {"Skenario": "Mediasi Preventif", "T_rec_mean": 72.249, "T_rec_std": 5.156, "K_final": 0.229, "K_max": 1.000, "P_selesai": 0.333},
    {"Skenario": "Mediasi Reaktif", "T_rec_mean": 65.890, "T_rec_std": 7.629, "K_final": 0.634, "K_max": 0.942, "P_selesai": 0.000},
    {"Skenario": "Mediator Eksternal", "T_rec_mean": 34.152, "T_rec_std": 6.799, "K_final": 0.000, "K_max": 1.000, "P_selesai": 1.000},
    {"Skenario": "Pendekatan Spiritual", "T_rec_mean": 54.652, "T_rec_std": 1.992, "K_final": 0.000, "K_max": 1.000, "P_selesai": 1.000},
])

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.markdown("## Navigasi")
view = st.sidebar.radio(
    "Pilih tampilan:",
    ["Ringkasan", "Visualisasi", "Analisis Statistik", "Data Mentah"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Kontrol Tampilan")
show_threshold = st.sidebar.checkbox("Tampilkan ambang konflik (0.7)", value=True)
show_values = st.sidebar.checkbox("Tampilkan label angka pada grafik", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Informasi Model")
st.sidebar.write("- **State**: Tenang → Cemas → Konflik → Pulih")
st.sidebar.write("- **Parameter**: S, D, R, M")
st.sidebar.write("- **Uji**: Monte Carlo 1000 iterasi")
st.sidebar.write("- **Warna khas**: Maroon")

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
color_map = {
    "Tanpa Koping": "#7A7A7A",
    "Mediasi Preventif": MAROON,
    "Mediasi Reaktif": BLUE,
    "Mediator Eksternal": ORANGE,
    "Pendekatan Spiritual": PURPLE,
}

pretty = {
    "Tanpa Koping": "Tanpa Koping",
    "Mediasi Preventif": "Mediasi Preventif",
    "Mediasi Reaktif": "Mediasi Reaktif",
    "Mediator Eksternal": "Mediator Eksternal",
    "Pendekatan Spiritual": "Pendekatan Spiritual",
}

best_recover = summary_df.loc[summary_df["T_rec_mean"].idxmin(), "Skenario"]
best_finish = summary_df.loc[summary_df["K_final"].idxmin(), "Skenario"]
best_success = summary_df.loc[summary_df["P_selesai"].idxmax(), "Skenario"]

# ---------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------
if view == "Ringkasan":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Skenario Tercepat", best_recover)
    c2.metric("K Final Terendah", best_finish)
    c3.metric("P Selesai Tertinggi", best_success)
    c4.metric("Iterasi Monte Carlo", "1000")

    st.markdown('<div class="sub-card">', unsafe_allow_html=True)
    st.subheader("Inti Temuan")
    st.write(
        "Mediator Eksternal adalah skenario paling cepat menurunkan konflik, sementara Pendekatan Spiritual "
        "dan Mediator Eksternal sama-sama berhasil mencapai resolusi penuh pada simulasi Monte Carlo. "
        "Tanpa koping menghasilkan konflik paling tinggi dan tidak pernah pulih sepenuhnya."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        fig = px.bar(
            summary_df,
            x="Skenario",
            y=["T_rec_mean", "K_final", "P_selesai"],
            barmode="group",
            color_discrete_sequence=[MAROON, ORANGE, GREEN],
            title="Perbandingan Metrik Utama"
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_font_color=TEXT,
            legend_title_text="Metrik",
            height=500,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=summary_df["Skenario"],
            y=summary_df["T_rec_mean"],
            mode="lines+markers",
            line=dict(color=MAROON, width=3),
            marker=dict(size=10),
            name="T_rec_mean"
        ))
        if show_threshold:
            fig2.add_hline(y=0.7, line_dash="dash", line_color=RED)
        fig2.update_layout(
            title="Rata-rata Waktu Pemulihan",
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20)
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
            fig = px.violin(
                summary_df.melt(id_vars="Skenario", value_vars=["T_rec_mean", "K_final", "K_max"], var_name="Metrik", value_name="Nilai"),
                x="Skenario", y="Nilai", color="Metrik", box=True, points="all",
                color_discrete_sequence=[MAROON, ORANGE, BLUE],
                title="Distribusi Metrik Ringkas"
            )
            fig.update_layout(height=520, plot_bgcolor="white", paper_bgcolor="white")
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
            fig.update_layout(height=520, showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("Pada notebook, visual single run yang cocok ditaruh di sini adalah grafik 4 skenario koping dan grafik analisis efektivitas.")
        fig = go.Figure()
        for i, sc in enumerate(summary_df["Skenario"]):
            fig.add_trace(go.Bar(
                name=sc,
                x=["T_rec_mean", "K_final", "K_max"],
                y=[summary_df.loc[i, "T_rec_mean"], summary_df.loc[i, "K_final"], summary_df.loc[i, "K_max"]],
                marker_color=color_map[sc]
            ))
        fig.update_layout(barmode='group', title='Perbandingan Metrik Utama per Skenario', height=500,
                          plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.info("Untuk bagian konseptualisasi, taruh state chart dan tabel atribut sebagai gambar di BAB III.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1,2,3,4], y=[1,2,3,4], mode='lines', line=dict(color=MAROON)))
        fig.update_layout(title='Placeholder Konseptualisasi (gunakan gambar state chart di laporan)', height=400,
                          plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Analisis Statistik
# ---------------------------------------------------------------------
elif view == "Analisis Statistik":
    st.subheader("Analisis Statistik")
    st.write("Hasil yang diinput menunjukkan perbedaan signifikan antar skenario pada T_recover.")

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
        fig.add_trace(go.Box(y=np.random.normal(summary_df.loc[summary_df['Skenario']==sc, 'T_rec_mean'].values[0], 1.5, 120), name=sc, marker_color=color_map[sc]))
    fig.update_layout(title='Ilustrasi Distribusi Waktu Pemulihan', height=520, plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabel Ringkasan")
    st.dataframe(summary_df.style.format({"T_rec_mean":"{:.3f}","T_rec_std":"{:.3f}","K_final":"{:.3f}","K_max":"{:.3f}","P_selesai":"{:.3f}"}), use_container_width=True)

# ---------------------------------------------------------------------
# Data Mentah
# ---------------------------------------------------------------------
elif view == "Data Mentah":
    st.subheader("Data Ringkasan")
    st.dataframe(summary_df, use_container_width=True)
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name="hasil_monte_carlo_ringkas.csv", mime="text/csv")

    st.markdown("### Catatan Penempatan Visual dalam Laporan")
    st.write("- State chart: BAB III, bagian metode/konseptualisasi.")
    st.write("- Tabel atribut: BAB III, setelah state chart.")
    st.write("- Grafik 4 skenario koping: BAB IV, hasil simulasi single run.")
    st.write("- Grafik Monte Carlo: BAB IV, setelah pembahasan simulasi iteratif.")
    st.write("- Grafik analisis statistik: BAB IV, bagian analitik dan trade-off.")

st.markdown("---")
st.caption("Streamlit Dashboard • Tema Maroon • ABM Waris Islam")
