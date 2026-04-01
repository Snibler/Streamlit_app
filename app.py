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

def earth_rotation_rate():
    return 2 * math.pi / (24 * 3600)

W_EARTH = earth_rotation_rate()

# === ФИЗИКА ===
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
def compute(freq, alt, sfs, bws, cr, pl_min, pl_max, dt, mode):
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

            elkrit_packet = []
            elkrit_symbol = []

            for PL in pl_values:
                Tpkt = Tpacket(SF, BW, int(PL), cr)
                Tsym = Ts(SF, BW)

                # === Packet ===
                y_packet = tol - abs_dfdt * Tpkt

                # === Symbol ===
                y_symbol = tol - abs_dfdt * Tsym

                def solve(y):
                    signs = np.sign(y)
                    idx = np.where(np.diff(signs))[0]

                    if len(idx) > 0:
                        j = idx[0]
                        t0, t1 = t[j], t[j+1]
                        y0, y1 = y[j], y[j+1]

                        tc = t0 if y1 == y0 else t0 - y0*(t1-t0)/(y1-y0)

                        gamma_cross = (w_sat - W_EARTH) * tc
                        e_rad = elevation_from_gamma(gamma_cross, rho)
                        return np.degrees(e_rad)

                    return 90.0 if y[0] >= 0 else 0.0

                elkrit_packet.append(solve(y_packet))
                elkrit_symbol.append(solve(y_symbol))

            curves[BW] = {
                "packet": elkrit_packet,
                "symbol": elkrit_symbol
            }

        results[SF] = (pl_values, curves)

    return results

# === UI ===
st.title("LoRa Doppler Critical Elevation Calculator")

freq = st.number_input("Frequency (Hz)", value=868e6)
alt = st.number_input("Orbit height (m)", value=500e3)

sfs = st.multiselect("SF", [7,8,9,10,11,12], default=[12])
bws = st.multiselect("BW (Hz)", [41700, 62500, 125000, 250000, 500000], default=[125000])

pl_min, pl_max = st.slider("Payload", 1, 255, (1, 120))
cr = st.selectbox("CR", [1,2,3,4], index=0)

mode = st.selectbox(
    "Model mode",
    ["Packet (SX12xx, SX1301 and older)", "Symbol (LR11xx, SX1302 and newer)", "Compare"]
)

# === РАСЧЁТ ===
if st.button("Рассчитать"):

    results = compute(freq, alt, sfs, bws, cr, pl_min, pl_max, dt=0.1, mode=mode)

    for SF, (pl, curves) in results.items():
        st.subheader(f"SF = {SF}")
        fig = go.Figure()

        for BW, data in curves.items():

            if mode in ["Packet", "Compare"]:
                fig.add_trace(go.Scatter(
                    x=pl,
                    y=data["packet"],
                    mode='lines',
                    name=f"BW={BW/1000:.1f}kHz (Packet)"
                ))

            if mode in ["Symbol", "Compare"]:
                fig.add_trace(go.Scatter(
                    x=pl,
                    y=data["symbol"],
                    mode='lines',
                    line=dict(dash='dash'),
                    name=f"BW={BW/1000:.1f}kHz (Symbol)"
                ))

        fig.update_layout(
            xaxis_title="Payload (bytes)",
            yaxis_title="ELkrit (deg)",
            hovermode="x unified"
        )

        st.plotly_chart(fig)

        # === экспорт ===
        df = pd.DataFrame({"PL": pl})
        for BW, data in curves.items():
            df[f"BW{BW}_packet"] = data["packet"]
            df[f"BW{BW}_symbol"] = data["symbol"]

        st.download_button(
            f"Download CSV (SF{SF})",
            df.to_csv(index=False).encode(),
            file_name=f"SF{SF}.csv"
        )