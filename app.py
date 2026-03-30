import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go

# === КОНСТАНТЫ ===
C = 299_792_458.0
R_EARTH = 6_371_000.0
G = 6.67430e-11
M_EARTH = 5.9726e24

# === ФИЗИКА ===
def earth_rotation_rate():
    return 2 * math.pi / (24 * 3600)

W_EARTH = earth_rotation_rate()

def orbital_velocity_and_rate(alt_m):
    v = math.sqrt(G * M_EARTH / (R_EARTH + alt_m))
    w = v / (R_EARTH + alt_m)
    return v, w

def horizon_central_angle(alt_m):
    rho = R_EARTH / (R_EARTH + alt_m)
    gamma_h = math.acos(rho)
    return gamma_h, rho

def elevation_from_gamma(gamma_rad, rho):
    num = np.cos(gamma_rad) - rho
    den = np.sin(gamma_rad)
    return np.arctan2(num, den)

def doppler_offset_rate(t, freq_hz, v_sat, w_sat, rho, dt):
    gamma = (w_sat - W_EARTH) * t
    el_rad = elevation_from_gamma(gamma, rho)
    f_t = (v_sat * freq_hz / C) * np.cos(el_rad)
    dfdt = np.gradient(f_t, dt)
    return np.abs(dfdt)

# === LoRa ===
def Ts(SF, BW_hz): return (2**SF) / BW_hz

def Tpreamble(SF, BW_hz, npreamble=12.0, add_tail=True):
    return (npreamble + (4.25 if add_tail else 0.0)) * Ts(SF, BW_hz)

def npayload_symbols(SF, PL_bytes, CR, IH=1, DE=0):
    num = (8*PL_bytes - 4*SF + 28 + 16*CR - 20*IH)
    den = 4*(SF - 2*DE)
    add = math.ceil(num/den) if den>0 else 0
    add = max(add, 0) * (CR + 4)
    return 8 + int(add)

def Tpacket(SF, BW_hz, PL_bytes, CR, npreamble=12.0, add_tail=True, IH=1, DE=0):
    return Tpreamble(SF, BW_hz, npreamble, add_tail) + \
           npayload_symbols(SF, PL_bytes, CR, IH, DE)*Ts(SF, BW_hz)

def fp_max(BW_hz, SF): return BW_hz / (3 * (2**SF))

# === ОСНОВНОЙ РАСЧЁТ ===
def compute(freq, alt, sfs, bws, cr, pl_min, pl_max, dt):
    v_sat, w_sat = orbital_velocity_and_rate(alt)
    gamma_h, rho = horizon_central_angle(alt)
    t_half = gamma_h / w_sat
    t = np.arange(0.0, t_half+dt, dt)

    abs_dfdt = doppler_offset_rate(t, freq, v_sat, w_sat, rho, dt)
    pl_values = np.arange(pl_min, pl_max+1)

    results = {}

    for SF in sfs:
        curves = {}
        for BW in bws:
            tol = fp_max(BW, SF)
            elkrit_deg = []

            for PL in pl_values:
                Tpkt = Tpacket(SF, BW, int(PL), cr)

                y = tol - abs_dfdt * Tpkt
                signs = np.sign(y)
                sign_changes = np.where(np.diff(signs))[0]

                if len(sign_changes) > 0:
                    j = sign_changes[0]
                    t0, t1 = t[j], t[j+1]
                    y0, y1 = y[j], y[j+1]

                    tc = t0 if y1 == y0 else t0 - y0*(t1-t0)/(y1-y0)

                    gamma_cross = (w_sat - W_EARTH) * tc
                    e_rad = elevation_from_gamma(gamma_cross, rho)
                    elkrit_deg.append(np.degrees(e_rad))

                else:
                    elkrit_deg.append(90.0 if y[0] >= 0 else 0.0)

            curves[BW] = elkrit_deg

        results[SF] = (pl_values, curves)

    return results

# === UI ===
st.title("LoRa Doppler Critical Elevation Calculator")

freq = st.number_input("Частота (Гц)", value=868e6)
alt = st.number_input("Высота орбиты (м)", value=500e3)

sfs = st.multiselect("Spreading Factor", [7,8,9,10,11,12], default=[12])
bws = st.multiselect("Bandwidth (Hz)", [41700, 62500, 125000, 250000, 500000], default=[125000])

pl_min, pl_max = st.slider("Payload (bytes)", 1, 255, (1, 120))
cr = st.selectbox("Coding Rate", [1,2,3,4], index=0)

# === РАСЧЁТ ===
if st.button("Рассчитать"):
    results = compute(freq, alt, sfs, bws, cr, pl_min, pl_max, dt=0.1)

    for SF, (pl, curves) in results.items():
        st.subheader(f"SF = {SF}")

        fig = go.Figure()

        for BW, elkrit in curves.items():
            fig.add_trace(go.Scatter(
                x=pl,
                y=elkrit,
                mode='lines',
                name=f"BW={BW/1000:.1f} kHz"
            ))

        fig.update_layout(
            xaxis_title="Payload (bytes)",
            yaxis_title="Critical elevation (deg)",
            hovermode="x unified"
        )

        st.plotly_chart(fig)

        # Excel экспорт
        df = pd.DataFrame({"PL": pl})
        for BW, elkrit in curves.items():
            df[f"BW_{BW}"] = elkrit

        st.download_button(
            f"Скачать Excel (SF{SF})",
            df.to_csv(index=False).encode(),
            file_name=f"SF{SF}.csv"
        )