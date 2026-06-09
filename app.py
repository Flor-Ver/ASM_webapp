from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


ASSET_DIR = Path(__file__).parent / "assets"

st.set_page_config(page_title="Asynchrone Motor - Equivalent Schema", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none}
    .block-container {max-width:1500px; padding-top:.65rem; padding-bottom:2rem}
    h1 {font-size:1.55rem!important; margin-bottom:.2rem!important}
    div[data-testid="stTabs"] button {font-size:1rem; padding-left:1.5rem; padding-right:1.5rem}
    div[data-testid="stNumberInput"] {margin-bottom:-.48rem}
    div[data-testid="stNumberInput"] label p {font-size:.82rem}
    .section-title {font-size:1rem; font-weight:600; margin:.4rem 0 .25rem}
    .rule {border-top:2px solid #555; margin:1rem 0 .7rem}
    .result-table {width:100%; border-collapse:collapse; font-size:.86rem; margin-bottom:.8rem}
    .result-table td {padding:.17rem .15rem; vertical-align:middle}
    .result-table td:first-child {width:35%}
    .result-table td:nth-child(2) {width:50%; font-weight:700; white-space:nowrap}
    .result-table td:last-child {width:15%; color:#444}
    .schema-note {font-size:.88rem; line-height:1.35; margin:.2rem 0 .7rem}
    .help-copy {max-width:850px; font-size:.98rem; line-height:1.55}
    </style>
    """,
    unsafe_allow_html=True,
)


def motor_model(p, r1, l1, r2, l2, rg, lm, u, f, n):
    f_calc = max(float(f), 1e-7)
    p_calc = max(float(p), 1e-7)
    ns = 60 * f_calc / p_calc
    n_calc = min(max(float(n), 0.001), ns * 0.999999)
    slip = max((ns - n_calc) / ns, 1e-6)
    omega = 2 * np.pi * f_calc
    z1 = complex(r1, omega * l1)
    zmag = 1 / (1 / complex(rg, 0) + 1 / complex(0, omega * lm))
    zrotor = complex(r2 / slip, omega * l2)
    zp = 1 / (1 / zmag + 1 / zrotor)
    i1 = u / (z1 + zp)
    v_r1 = i1 * r1
    v_l1 = i1 * complex(0, omega * l1)
    e1 = u - v_r1 - v_l1
    im = e1 / complex(0, omega * lm)
    i2 = e1 / zrotor
    pin = u * np.real(i1)
    qin = u * np.imag(i1)
    p_iron = abs(e1) ** 2 / rg
    p_cu_r1 = r1 * abs(i1) ** 2
    p_cu_r2 = r2 * abs(i2) ** 2
    p_loss = p_iron + p_cu_r1 + p_cu_r2
    p_mech = pin - p_loss
    omega_m = 2 * np.pi * n_calc / 60
    torque = 3 * p_mech / omega_m if omega_m > 0 else 0
    efficiency = 100 * p_mech / pin if pin > 0 else 0
    pf = pin / (u * abs(i1)) if u * abs(i1) else 0
    return {
        "I_1": i1, "|I_1|": abs(i1), "V_R1": v_r1, "V_L1": v_l1,
        "E_1": e1, "I_m": im, "I_2": i2, "|I_2|": abs(i2),
        "P_in": pin, "Q_in": qin, "PF": pf, "P_iron": p_iron,
        "P_cu_R1": p_cu_r1, "P_cu_R2": p_cu_r2, "P_loss": p_loss,
        "P_mech": p_mech, "Torque": torque, "Efficiency": efficiency, "ns": ns,
    }


def fmt_complex(value):
    sign = "+" if value.imag >= 0 else "-"
    return f"{value.real:.2f} {sign} {abs(value.imag):.2f}j"


def asset(name):
    path = ASSET_DIR / name
    return path if path.exists() else None


def result_table(title, rows):
    html = [f'<div class="section-title">{title}</div><table class="result-table">']
    for label, value, unit, tip in rows:
        html.append(f'<tr title="{tip}"><td>{label}</td><td>{value}</td><td>{unit}</td></tr>')
    st.markdown("".join(html) + "</table>", unsafe_allow_html=True)


def schema_for(f, n, p):
    ns = 60 * f / p if p and f else 0
    if f == 0:
        return "DC_proef.png", (
            "Door te testen met een DC-spanning worden de spoelen overbrugd, "
            "waardoor het schema vereenvoudigt tot het getoonde schema."
        )
    if ns > 0 and abs(n - ns) < 1e-3:
        return "Nullastproef.png", (
            "Door de machine op synchrone snelheid te laten draaien is de rotor "
            "een open keten. Bijgevolg valt dat deel weg uit het schema."
        )
    if n == 0:
        return "Kortsluitproef.png", (
            "De magnetisatietak wordt verwaarloosd omdat de stroom door deze tak "
            "veel kleiner is dan de rotorstroom. Hierdoor ontstaat een kleine fout."
        )
    return "Volledig_schema.png", ""


def image_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def meter_image(is_dc, values, r1):
    path = asset("Weerstand_meting.png" if is_dc else "Vermogen_meting.png")
    if not path:
        return None
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    if is_dc:
        center = (w * (.714 + .814) / 2, h * (.18 + .48) / 2)
        draw.text(center, f"{r1:.2f} Ω", fill="#1a1a1a",
                  font=image_font(max(12, int(h * .045))), anchor="mm")
    else:
        x0, x1 = int(w * .625), int(w * .765)
        y0, y1 = int(h * .41), int(h * .62)
        rows = [
            ("Vrms", f"{values['U']:.2f}", "V", "white"),
            ("Irms", f"{values['|I_1|']:.2f}", "A", "#ffff00"),
            ("P1", f"{values['P_in']:.2f}", "W", "#00ff88"),
            ("PF", f"{values['PF']:.2f}", "", "white"),
        ]
        row_h = (y1 - y0) / 4
        text_font = image_font(max(10, int(row_h * .52)))
        for idx, (label, value, unit, color) in enumerate(rows):
            y = y0 + (idx + .5) * row_h
            draw.text((x0, y), label, fill=color, font=text_font, anchor="lm")
            draw.text((x0 + (x1 - x0) * .82, y), value, fill=color, font=text_font, anchor="rm")
            if unit:
                draw.text((x0 + (x1 - x0) * .84, y), unit, fill=color, font=text_font, anchor="lm")
    return image


def trial_parameters(r1_dc, u_nl, i_nl, p_nl, u_ks, i_ks, p_ks, f, l1):
    result = {}
    omega = 2 * np.pi * (f if f > 0 else 50)
    if r1_dc is not None:
        result["R1"] = (r1_dc, "Ω")
    if all(v is not None for v in (r1_dc, u_nl, i_nl, p_nl)):
        cos_phi = np.clip(p_nl / (u_nl * i_nl) if u_nl * i_nl else 0, -1, 1)
        i10 = i_nl * complex(cos_phi, -np.sqrt(max(1 - cos_phi**2, 0)))
        e10 = complex(u_nl, 0) - complex(r1_dc, omega * l1) * i10
        p_fe = max(p_nl - r1_dc * i_nl**2, 1e-9)
        rg_calc = abs(e10) ** 2 / p_fe
        im_abs = abs(i10 - e10 / rg_calc)
        result["E10"] = (e10, "V")
        result["Rg"] = (rg_calc, "Ω")
        if im_abs:
            result["Lm"] = (abs(e10) / (omega * im_abs), "H")
    if all(v is not None for v in (u_ks, i_ks, p_ks)) and i_ks > 0:
        zk = u_ks / i_ks
        rk = p_ks / i_ks**2
        xk = np.sqrt(max(zk**2 - rk**2, 0))
        result["Xk"] = (xk, "Ω")
        result["L1L2"] = (xk / (2 * omega), "H")
        if r1_dc is not None:
            result["R2"] = (rk - r1_dc, "Ω")
    return result


st.title("Asynchrone Motor - Equivalent Schema")
tab_help, tab_calc, tab_graph = st.tabs(["Handleiding", "Berekeningen", "Grafieken"])

with tab_help:
    st.markdown(
        """
        <div class="help-copy">
        <h2>Doel van de applicatie</h2>
        <p>Deze applicatie is ontwikkeld als interactief leermiddel voor het
        bestuderen van de werking van een asynchrone machine. Ze maakt gebruik
        van het equivalente schema om elektrische en mechanische grootheden te
        berekenen, zoals stromen, vermogens, rendement en elektromagnetisch koppel.</p>
        <p>Het schema bevat de statorparameters <i>R<sub>1</sub></i> en
        <i>L<sub>1</sub></i>, de magnetisatietak <i>R<sub>g</sub></i> en
        <i>L<sub>m</sub></i>, en de herleide rotorparameters
        <i>R<sub>2</sub>'</i> en <i>L<sub>2</sub>'</i>.</p>
        <h2>Algemene werking van de applicatie</h2>
        <p>In <b>Berekeningen</b> kunnen de parameters, bedrijfsomstandigheden
        en meetwaarden van de DC-, nullast- en kortsluitproef worden aangepast.
        De resultaten, het schema en het meettoestel worden automatisch bijgewerkt.</p>
        <p>In <b>Grafieken</b> kunnen de belangrijkste karakteristieken van de
        machine worden onderzocht.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_calc:
    input_col, output_col, visual_col = st.columns([1.05, 1.35, 3.4], gap="large")
    with input_col:
        st.markdown('<div class="section-title">Constanten</div>', unsafe_allow_html=True)
        p = st.number_input("p  [-]", min_value=0.1, value=2.0, step=1.0, help="Aantal poolparen")
        r1 = st.number_input("R₁  [Ω]", min_value=0.0001, value=8.2, help="Statorweerstand")
        l1 = st.number_input("L₁  [H]", min_value=0.000001, value=0.0364, format="%.4f", help="Statorlekinductantie")
        r2 = st.number_input("R₂'  [Ω]", min_value=0.0001, value=5.7, help="Herleide rotorweerstand")
        l2 = st.number_input("L₂'  [H]", min_value=0.000001, value=0.0364, format="%.4f", help="Herleide rotorlekinductantie")
        rg = st.number_input("R_g  [Ω]", min_value=0.0001, value=4576.0, help="IJzerverliesweerstand")
        lm = st.number_input("L_m  [H]", min_value=0.000001, value=0.624, format="%.4f", help="Magnetiseringsinductantie")
        st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
        u = st.number_input("V₁  [V]", min_value=0.0, value=230.0, help="Fasespanning RMS")
        f = st.number_input("f  [Hz]", min_value=0.0, value=50.0, help="Voedingsfrequentie")
        ns_display = 60 * f / p if p and f else 0
        n = st.number_input("n  [rpm]", min_value=0.0, max_value=max(ns_display, .01),
                            value=min(1490.0, max(ns_display, .01)), help="Rotorsnelheid")
        st.markdown('<div class="rule"></div><div class="section-title">DC-proef</div>', unsafe_allow_html=True)
        r1_dc = st.number_input("R₁,DC  [Ω]", min_value=0.0, value=None, placeholder="…")
        st.markdown('<div class="section-title">Nullastproef</div>', unsafe_allow_html=True)
        u_nl = st.number_input("U₁₀  [V]", min_value=0.0, value=None, placeholder="…")
        i_nl = st.number_input("I₁₀  [A]", min_value=0.0, value=None, placeholder="…")
        p_nl = st.number_input("P₁₀  [W]", min_value=0.0, value=None, placeholder="…")
        st.markdown('<div class="section-title">Kortsluitproef</div>', unsafe_allow_html=True)
        u_ks = st.number_input("U₁k  [V]", min_value=0.0, value=None, placeholder="…")
        i_ks = st.number_input("I₁k  [A]", min_value=0.0, value=None, placeholder="…")
        p_ks = st.number_input("P₁k  [W]", min_value=0.0, value=None, placeholder="…")

    results = motor_model(p, r1, l1, r2, l2, rg, lm, u, f, n)
    results["U"] = u
    calculated = trial_parameters(r1_dc, u_nl, i_nl, p_nl, u_ks, i_ks, p_ks, f, l1)

    with output_col:
        result_table("Stromen en spanningen", [
            ("I<sub>1</sub>", fmt_complex(results["I_1"]), "A", "Complexe statorstroom"),
            ("|I<sub>1</sub>|", f"{results['|I_1|']:.2f}", "A", "Effectieve statorstroom"),
            ("V<sub>R1</sub>", fmt_complex(results["V_R1"]), "V", "Spanningsval over R1"),
            ("V<sub>L1</sub>", fmt_complex(results["V_L1"]), "V", "Spanningsval over L1"),
            ("E<sub>1</sub>", fmt_complex(results["E_1"]), "V", "Interne spanning"),
            ("I<sub>m</sub>", fmt_complex(results["I_m"]), "A", "Magnetiseringsstroom"),
            ("I<sub>2</sub>'", fmt_complex(results["I_2"]), "A", "Herleide rotorstroom"),
            ("|I<sub>2</sub>'|", f"{results['|I_2|']:.2f}", "A", "Effectieve rotorstroom"),
        ])
        result_table("Vermogens", [
            ("P<sub>in</sub>", f"{results['P_in']:.2f}", "W", "Actief ingangsvermogen per fase"),
            ("Q<sub>in</sub>", f"{results['Q_in']:.2f}", "var", "Reactief ingangsvermogen per fase"),
            ("PF", f"{results['PF']:.2f}", "-", "Power factor"),
            ("P<sub>fe</sub>", f"{results['P_iron']:.2f}", "W", "IJzerverliesvermogen"),
            ("P<sub>cu,R1</sub>", f"{results['P_cu_R1']:.2f}", "W", "Statorkoperverlies"),
            ("P<sub>cu,R2</sub>", f"{results['P_cu_R2']:.2f}", "W", "Rotorkoperverlies"),
            ("P<sub>L</sub>", f"{results['P_loss']:.2f}", "W", "Totaal verliesvermogen"),
            ("P<sub>m</sub>", f"{results['P_mech']:.2f}", "W", "Mechanisch vermogen per fase"),
        ])
        result_table("Koppel en efficiëntie", [
            ("T", f"{results['Torque']:.2f}", "Nm", "Elektromagnetisch koppel"),
            ("η", f"{results['Efficiency']:.2f}", "%", "Rendement"),
        ])
        trial_spec = [
            ("R1", "R<sub>1</sub>", "Ω"), ("E10", "E̲<sub>10</sub>", "V"),
            ("Rg", "R<sub>g</sub>", "Ω"), ("Lm", "L<sub>m</sub>", "H"),
            ("R2", "R<sub>2</sub>'", "Ω"), ("Xk", "X<sub>k</sub>", "Ω"),
            ("L1L2", "L<sub>1</sub> = L<sub>2</sub>'", "H"),
        ]
        trial_rows = []
        for key, label, unit in trial_spec:
            if key in calculated:
                value = calculated[key][0]
                text = fmt_complex(value) if isinstance(value, complex) else (
                    f"{value:.4f}" if unit == "H" else f"{value:.2f}"
                )
            else:
                text = "—"
            trial_rows.append((label, text, unit, "Berekend uit de proefmetingen"))
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        result_table("Berekende parameters (uit proeven)", trial_rows)

    with visual_col:
        schema_name, note = schema_for(f, n, p)
        schema = asset(schema_name)
        if schema:
            st.image(str(schema), width="stretch")
        if note:
            st.markdown(f'<div class="schema-note">{note}</div>', unsafe_allow_html=True)
        meter = meter_image(f == 0, results, r1)
        if meter:
            st.image(meter, width="stretch")

with tab_graph:
    graph_controls, graph_area = st.columns([1.15, 4], gap="large")
    with graph_controls:
        graph_type = st.selectbox("Selecteer grafiek", [
            "Koppel vs snelheid",
            "Koppel vs snelheid met variabele R₂'",
            "Efficiëntie vs snelheid",
            "Power factor vs koppel",
        ])
        r2_pct = 100
        fan_curve = False
        if graph_type == "Koppel vs snelheid met variabele R₂'":
            r2_pct = st.slider("R₂' [% van nominale R₂']", 50, 300, 100)
            st.caption(f"R₂' = {r2 * r2_pct / 100:.2f} Ω ({r2_pct}% van nominale R₂')")
            fan_curve = st.checkbox("Ventilator karakteristiek tonen")

    ns = results["ns"]
    speeds = np.linspace(0, ns * .999999, 300)
    selected_r2 = r2 * r2_pct / 100
    curve_results = [motor_model(p, r1, l1, selected_r2, l2, rg, lm, u, f, speed) for speed in speeds]
    torques = np.array([x["Torque"] for x in curve_results])
    marker_min = float(speeds[int(np.argmax(torques))]) if graph_type == "Power factor vs koppel" else 0.0
    with graph_controls:
        prompts = {
            "Efficiëntie vs snelheid": "Snelheid [rpm] waarvoor we de efficiëntie [%] berekenen",
            "Power factor vs koppel": "Snelheid [rpm] en dus indirect het koppel, waarvoor we de power factor berekenen",
        }
        st.write(prompts.get(graph_type, "Snelheid [rpm] waarvoor we het koppel [Nm] berekenen"))
        marker_speed = st.number_input(
            "Snelheid [rpm]", min_value=marker_min, max_value=float(ns * .999999),
            value=min(max(float(n), marker_min), float(ns * .999999)), label_visibility="collapsed",
        )
    marker = motor_model(p, r1, l1, selected_r2, l2, rg, lm, u, f, marker_speed)
    fig = go.Figure()
    if graph_type in ("Koppel vs snelheid", "Koppel vs snelheid met variabele R₂'"):
        fig.add_scatter(x=speeds, y=torques, mode="lines", name="Motorkarakteristiek")
        fig.add_scatter(x=[marker_speed], y=[marker["Torque"]], mode="markers", name="Bedrijfspunt", marker_size=9)
        if fan_curve:
            fan = .5 * np.max(torques) / ns**3 * speeds**3
            fig.add_scatter(x=speeds, y=fan, mode="lines", name="Ventilator karakteristiek", line_dash="dash")
        fig.update_layout(xaxis_title="Snelheid [rpm]", yaxis_title="Koppel [Nm]",
                          title=f"Koppel in functie van snelheid bij R₂' = {selected_r2:.2f} Ω")
        with graph_controls:
            st.markdown(f"**Koppel: {marker['Torque']:.2f} Nm bij {marker_speed:.2f} rpm**")
    elif graph_type == "Efficiëntie vs snelheid":
        efficiency = np.array([x["Efficiency"] for x in curve_results])
        fig.add_scatter(x=speeds, y=efficiency, mode="lines")
        fig.add_scatter(x=[marker_speed], y=[marker["Efficiency"]], mode="markers", marker_size=9)
        fig.update_layout(xaxis_title="Snelheid [rpm]", yaxis_title="Efficiëntie [%]",
                          title="Efficiëntie in functie van snelheid")
        with graph_controls:
            st.markdown(f"**Efficiëntie: {marker['Efficiency']:.2f} % bij {marker_speed:.2f} rpm**")
    else:
        pf = np.array([x["PF"] for x in curve_results])
        kip = int(np.argmax(torques))
        fig.add_scatter(x=torques[kip:], y=pf[kip:], mode="lines", name="PF (stabiel gebied)")
        fig.add_scatter(x=[torques[kip]], y=[pf[kip]], mode="markers", name="Kipkoppel", marker_color="red", marker_size=9)
        fig.add_scatter(x=[marker["Torque"]], y=[marker["PF"]], mode="markers", marker_size=9)
        fig.update_layout(xaxis_title="Koppel [Nm]", yaxis_title="Power factor [-]",
                          title="Power factor in functie van het koppel (stabiel gebied)", yaxis_range=[0, 1])
        with graph_controls:
            st.markdown(f"**PF: {marker['PF']:.3f} bij koppel {marker['Torque']:.2f} Nm**")
    fig.update_layout(height=620, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_xaxes(rangemode="tozero", showgrid=True)
    fig.update_yaxes(rangemode="tozero", showgrid=True)
    with graph_area:
        st.plotly_chart(fig, width="stretch")
