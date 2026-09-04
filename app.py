"""Interactive Cp(T) database for engineering materials. Run: py -m streamlit run app.py"""

# log, started aug 20 2026
# aug20-24: db. five days. just organizing the cp csv. v3 v4 v5 ... v24. never again
# aug25: pixelation fix, words stacking over each other on the plot
# aug26: units. molar vs mass. refusal instead of guessing masses
# aug27: transitions. 14. uranium gets 2
# aug28: mullite stitch. smooth join != transition
# aug29: latex eqns + captions
# aug30: families / relatives dropdown
# aug31: extremes + ranking slider
# sep1: backdrop. red room
# sep2: exports
# sep3: tests green
# sep4: restyle pass, this one

import base64
import math  # debye estimate experiment. kept the import, dropped the idea
import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

R = 8.314462618  # J/(mol*K)
CSV_NAME = "cp_database_v24.csv"
BACKDROP_NAME = 'red_room.jpg'
HERE = os.path.dirname(os.path.abspath(__file__))

MOLAR_UNIT = "J/mol\u00b7K"
MASS_UNIT = 'J/g\u00b7K'

ABSENT_TOKEN = 'NaN'
SEGMENT_SUFFIX = re.compile(r"\s*\(\s*\d+\s*-\s*\d+\s*K\s*\)\s*$")

FIT_MEASURED = "measured range"  # leftover, only FIT_POINT gets checked now
FIT_POINT = 'point value, range by convention'

MAX_OVERLAY = 10
REFERENCE_T = 298.15

SERIES_COLORS = [
    "#F2C14E", '#7fb7be', "#E88D67", '#b8a9c9', "#9BC995",
    "#f28482", "#84A9C0", '#d9bf77', "#C89B9B", "#e5e1d8",
]
BRIDGE_COLOR = "#ffffff"  # only compose_figure uses this. fine where it is
GOLD = "#F2C14E"

plotly_cfg = {
    'scrollZoom': True,
    "displayModeBar": True,
    "displaylogo": False,
    'toImageButtonOptions': {"format": "png", 'filename': "cp_vs_T", "scale": 2},
}

NUMERIC_COLUMNS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "T_min", "T_max"]
OPTIONAL_TEXT_COLUMNS = ['Formula', "Composition", 'Molar_Mass', "Molar_Basis"]


@st.cache_data(show_spinner=False)
def load_database() -> pd.DataFrame:
    path = os.path.join(HERE, CSV_NAME)
    if not os.path.exists(path):
        st.error(f"{CSV_NAME} not found next to app.py.")
        st.stop()

    # keep_default_na=False: the literal 'NaN' token must survive to be handled below.
    frame = pd.read_csv(path, keep_default_na=False, na_values=[], dtype=str)
    # print(frame.shape)
    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in OPTIONAL_TEXT_COLUMNS:
        frame[col] = pd.Series(
            [None if v is None or str(v).strip() in ("", ABSENT_TOKEN) else v.strip() for v in frame[col]],
            dtype=object,
        )
    frame['molar_mass'] = pd.Series([float(v) if v is not None else None for v in frame["Molar_Mass"]], dtype=object)
    frame["base_name"] = [SEGMENT_SUFFIX.sub("", name) for name in frame["Material"]]
    return frame


def _peek(frame):
    # half done debug helper from the v19 audit. leaving it, costs nothing
    for col in frame.columns:
        pass
    # print(frame.head())


def _nk_estimate(comp):
    # neumann-kopp attempt. validation said no. keeping the stub as a warning
    # for elem, frac in comp:
    #     ...
    return None  # TODO delete someday


def evaluate_cp(row, T):
    """Cp in the row's native units for NASA9, Shomate, or Constant fits."""
    gas_const = R
    c = [float(row[f"C{i}"]) for i in range(1, 8)]
    kind = row['Polynomial_Type']
    if kind == "NASA9":
        return gas_const * (c[0] / (T * T) + c[1] / T + c[2] + c[3] * T + c[4] * T**2 + c[5] * T**3 + c[6] * T**4)
    if kind == "Shomate":
        t = T / 1000.0
        return c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] / (t * t)
    if kind == "Constant":
        return c[0]
    raise ValueError("Unknown fit type: {}".format(kind))


def in_range(row, T):
    return float(row["T_min"]) <= T <= float(row["T_max"])


def can_show_unit(row, unit):
    # gate on molar_mass only. never derive a mass from Formula/Composition
    return row["Units"] == unit or (row["Units"] == MOLAR_UNIT and unit == MASS_UNIT)


def convert_unit(value, row, unit):
    if row["Units"] == unit:
        return value
    if row['Units'] == MOLAR_UNIT and unit == MASS_UNIT:
        return value / row["molar_mass"]
    raise ValueError("Undefined conversion: material has no molar mass.")


def find_transitions(rows):
    """Phase-label change or Cp jump > threshold at a segment join."""
    transitions = []
    for i in range(len(rows) - 1):
        lower, upper = rows.iloc[i], rows.iloc[i + 1]
        boundary = float(lower["T_max"])
        if abs(boundary - float(upper["T_min"])) > 1e-6:
            continue
        cp_below = evaluate_cp(lower, boundary)
        cp_above = evaluate_cp(upper, boundary)
        jump_fraction = abs(cp_above - cp_below) / max(abs(cp_below), 1e-12)
        phase_changed = lower["Phase"] != upper["Phase"]
        # 2% cutoff. why does this threshold work? idk. mullite says thanks
        if not phase_changed and jump_fraction <= 0.02:
            continue
        transitions.append({
            "temperature": boundary,
            'from_phase': lower["Phase"],
            "to_phase": upper['Phase'],
            "cp_below": cp_below,
            'cp_above': cp_above,
            "delta_cp": cp_above - cp_below,
            "jump_percent": 100.0 * (cp_above - cp_below) / cp_below,
            'phase_changed': phase_changed,
            "row_below": lower,
        })
    return transitions


# latex bits
def _latex_number(value, scientific):
    if scientific:
        mantissa, exponent = f"{abs(value):.4e}".split("e")
        return f"{mantissa}\\times10^{{{int(exponent)}}}"
    return f"{abs(value):.4g}"


def _latex_terms(coefs, symbols, scientific):
    text = ""
    for value, symbol in zip(coefs, symbols):
        if value == 0.0:
            continue
        number = _latex_number(value, scientific)
        body = number if symbol == "" else f"{number}\\,{symbol}"
        text += (("-" if value < 0 else "") if text == "" else (" - " if value < 0 else " + ")) + body
    return text or "0"


def format_equation_latex(row):
    c = [float(row[f"C{i}"]) for i in range(1, 8)]
    kind = row["Polynomial_Type"]

    if kind == "Constant":
        return f"C_p = {c[0]:.4g}"
    if kind == "Shomate":
        symbols = ["", "t", "t^{2}", "t^{3}", "t^{-2}"]
        return "C_p = " + _latex_terms(c[:5], symbols, False)
    if kind == "NASA9":
        symbols = ["T^{-2}", "T^{-1}", "", "T", "T^{2}", "T^{3}", "T^{4}"]
        return "C_p = R\\left[\\," + _latex_terms(c, symbols, True) + "\\,\\right]"
    return "?"


def equation_caption(row):
    lo, hi = float(row["T_min"]), float(row["T_max"])
    kind = row["Polynomial_Type"]
    units = row['Units']
    if kind == "Constant":
        return f"{units}. Single measured point, held constant over {lo:.0f}\u2013{hi:.0f} K."
    if kind == "Shomate":
        return f"{units}. t = T/1000. Valid {lo:.0f}\u2013{hi:.0f} K."
    return f"{units}. R = {R:.4f}. Valid {lo:.0f}\u2013{hi:.0f} K."


class MaterialCatalog:
    """Groups CSV rows into materials."""

    def __init__(self, frame):
        self.frame = frame
        self.materials = sorted(set(frame["base_name"]))
        self._segments = {
            mat: frame[frame["base_name"] == mat].sort_values("T_min").reset_index(drop=True)
            for mat in self.materials
        }

    def segments_of(self, material):
        return self._segments[material]

    def first_row(self, material):
        return self._segments[material].iloc[0]

    def valid_range(self, material):
        rows = self._segments[material]
        return float(rows["T_min"].min()), float(rows["T_max"].max())

    def is_point_value(self, material):
        return FIT_POINT in set(self._segments[material]["Fit_Type"])

    def categories(self):
        return sorted(set(self.frame["Category"]))

    def search(self, query, categories, fit_filter):
        needle = query.strip().lower()
        result = []
        for mat in self.materials:
            row = self.first_row(mat)
            if categories and row["Category"] not in categories:
                continue
            if fit_filter == "Fitted curves only" and self.is_point_value(mat):
                continue
            if fit_filter == "Point values only" and not self.is_point_value(mat):
                continue
            haystack = " ".join(x.lower() for x in [mat, row["Formula"] or "", row["Composition"] or ""])
            if needle and needle not in haystack:
                continue
            result.append(mat)
        return result


class RelativeFinder:
    """Entries sharing a formula (other phases, allotropes, second sources)."""
    # FIXME: revisit this if we add more sources

    def __init__(self, catalog):
        self.catalog = catalog
        self._by_formula = {}
        for mat in catalog.materials:
            formula = catalog.first_row(mat)["Formula"]
            if formula is None:
                continue
            self._by_formula.setdefault(formula, []).append(mat)

    def families(self):
        result = []
        for formula in sorted(self._by_formula):
            members = self._by_formula[formula]
            if len(members) < 2:
                continue
            # print(formula, members)
            phases = list(dict.fromkeys(self.catalog.first_row(m)["Phase"] for m in members))
            kind = "phases" if len(phases) > 1 else "forms"
            label = f"{formula}: " + ", ".join(members) + f"  ({len(members)} {kind})"
            result.append({"formula": formula, 'members': list(members), "label": label})
        return result

    def relatives_of(self, material):
        row = self.catalog.first_row(material)
        formula = row["Formula"]
        if formula is None:
            return []
        return [
            {
                "material": other,
                'phase': self.catalog.first_row(other)["Phase"],
                "kind": "another form or source"
                if self.catalog.first_row(other)["Phase"] == row["Phase"]
                else "another phase",
            }
            for other in self._by_formula.get(formula, [])
            if other != material
        ]


class CurveBuilder:
    """Samples curves per segment, clipped to each fitted range; never extrapolates."""

    def __init__(self, catalog):
        self.catalog = catalog

    def build(self, material, t_low, t_high, unit):
        cat = self.catalog
        rows = cat.segments_of(material)
        first = rows.iloc[0]
        valid_min, valid_max = cat.valid_range(material)
        info = {
            "material": material,
            'category': first["Category"],
            "valid_min": valid_min,
            "valid_max": valid_max,
            'unit_refused': not can_show_unit(first, unit),
            "clipped_low": t_low < valid_min,
            "clipped_high": t_high > valid_max,
            "has_point_value": False,
        }
        if info["unit_refused"]:
            return [], info

        total_span = max(valid_max - valid_min, 1e-9)
        pieces = []
        for _, row in rows.iterrows():
            lo = max(float(row["T_min"]), t_low)
            hi = min(float(row["T_max"]), t_high)
            if hi <= lo:
                continue
            # 400 pts across the whole span, floor 12 per seg. looked fine, kept it
            num = max(int(400 * (float(row["T_max"]) - float(row["T_min"])) / total_span), 12)
            tempList = [lo + (hi - lo) * k / (num - 1) for k in range(num)]
            values = [convert_unit(evaluate_cp(row, T), row, unit) for T in tempList]
            # print(material, lo, hi, num)
            is_point = row["Fit_Type"] == FIT_POINT
            info["has_point_value"] = info["has_point_value"] or is_point
            pieces.append({
                "t": tempList,
                "cp": values,
                'phase': row["Phase"],
                "fit": row["Polynomial_Type"],
                'point_value': is_point,
            })
        return pieces, info

    def cp_at(self, material, T, unit):
        """Cp at one temperature, or None if no segment covers it."""
        for _, row in self.catalog.segments_of(material).iterrows():
            if not in_range(row, T):
                continue
            if not can_show_unit(row, unit):
                return None
            return convert_unit(evaluate_cp(row, T), row, unit)
        return None

    def extrema(self, material, unit):
        best_max = best_min = None
        for _, row in self.catalog.segments_of(material).iterrows():
            if not can_show_unit(row, unit):
                return None
            lo, hi = float(row["T_min"]), float(row["T_max"])
            for k in range(400):
                T = lo + (hi - lo) * k / 399
                value = convert_unit(evaluate_cp(row, T), row, unit)
                if best_max is None or value > best_max[0]:
                    best_max = (value, T)
                if best_min is None or value < best_min[0]:
                    best_min = (value, T)
        return {"max": best_max, "min": best_min}


def apply_backdrop():
    """Optional red_room.jpg backdrop; the only injected style rule."""
    # lazy on purpose. no jpg, no backdrop, move on
    try:
        with open(os.path.join(HERE, BACKDROP_NAME), "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
    except:
        return
    st.markdown(
        "<style>.stApp{background-image:"
        "linear-gradient(rgba(11,4,6,0.82),rgba(11,4,6,0.82)),"
        f'url("data:image/jpeg;base64,{encoded}");'
        "background-size:cover;background-position:center;background-attachment:fixed}</style>",
        unsafe_allow_html=True,
    )


def find_extremes(entries):
    """(value, T, material) of the highest and lowest plotted sample."""
    samples = [
        (cp, T, info["material"])
        for pieces, info, _ in entries
        for piece in pieces
        for T, cp in zip(piece["t"], piece["cp"])
    ]
    if not samples:
        return None, None
    return max(samples, key=lambda s: s[0]), min(samples, key=lambda s: s[0])


def mark_extremes(fig, entries, unit):
    highest, lowest = find_extremes(entries)
    if highest is None:
        return

    # max marker
    cp, T, material = highest
    fig.add_trace(go.Scatter(
        x=[T], y=[cp], mode="markers", showlegend=False, name="max",
        marker=dict(color=GOLD, size=11, symbol="circle-open", line=dict(width=2)),
        hovertemplate=f"<b>max</b><br>{material}<br>{T:.0f} K &nbsp; {cp:.4g} {unit}<extra></extra>",
    ))
    fig.add_annotation(
        x=T, y=cp, ax=0, ay=-34, showarrow=True, arrowhead=0, arrowcolor=GOLD,
        text=f"max: {cp:.4g} {unit}<br>{material}, {T:.0f} K",
        font=dict(color=GOLD, size=10),
        bgcolor="rgba(11,4,6,0.9)", bordercolor=GOLD, borderwidth=1, borderpad=3,
    )

    # min marker. yes, copy pasted from above. two blocks, a loop buys nothing
    cp, T, material = lowest
    fig.add_trace(go.Scatter(
        x=[T], y=[cp], mode="markers", showlegend=False, name='min',
        marker=dict(color=GOLD, size=11, symbol="circle-open", line=dict(width=2)),
        hovertemplate=f"<b>min</b><br>{material}<br>{T:.0f} K &nbsp; {cp:.4g} {unit}<extra></extra>",
    ))
    fig.add_annotation(
        x=T, y=cp, ax=0, ay=34, showarrow=True, arrowhead=0, arrowcolor=GOLD,
        text=f"min: {cp:.4g} {unit}<br>{material}, {T:.0f} K",
        font=dict(color=GOLD, size=10),
        bgcolor="rgba(11,4,6,0.9)", bordercolor=GOLD, borderwidth=1, borderpad=3,
    )


def compose_figure(entries, unit, t_low, t_high):
    fig = go.Figure()

    for i, (pieces, info, transitions) in enumerate(entries):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        legend_done = False

        for piece in pieces:
            fig.add_trace(go.Scatter(
                x=piece["t"],
                y=piece["cp"],
                mode="lines",
                name=info["material"],
                legendgroup=info["material"],
                showlegend=not legend_done,
                line=dict(color=color, width=2.2, dash="dash" if piece["point_value"] else "solid"),
                hovertemplate=(
                    f"<b>{info['material']}</b><br>%{{x:.1f}} K &nbsp; %{{y:.4g}} {unit}"
                    f"<br><i>{piece['phase']} \u00b7 {piece['fit']}</i><extra></extra>"
                ),
            ))
            legend_done = True

        for tr in transitions:
            T = tr["temperature"]
            if not (t_low <= T <= t_high):
                continue
            below = convert_unit(tr["cp_below"], tr["row_below"], unit)
            above = convert_unit(tr["cp_above"], tr["row_below"], unit)
            delta = above - below
            sign = "+" if delta >= 0 else "\u2212"
            label = f"{tr['from_phase']} \u2192 {tr['to_phase']}" if tr["phase_changed"] else "structural change"
            fig.add_trace(go.Scatter(
                x=[T, T],
                y=[below, above],
                mode="lines+markers",
                name="phase change",
                legendgroup='bridge',
                showlegend=False,
                line=dict(color=BRIDGE_COLOR, width=2.4),
                marker=dict(color=BRIDGE_COLOR, size=6, symbol="diamond"),
                hovertemplate=(
                    f"<b>{info['material']}</b> \u00b7 {label}<br>{T:.0f} K"
                    f"<br>\u0394Cp = {sign}{abs(delta):.4g} {unit} ({tr['jump_percent']:+.1f}%)"
                    "<extra></extra>"
                ),
            ))
            fig.add_annotation(
                x=T, y=(below + above) / 2.0,
                text=f"\u0394Cp {sign}{abs(delta):.3g}",
                showarrow=True, arrowhead=0, arrowcolor=BRIDGE_COLOR, ax=46, ay=0, xanchor="left",
                font=dict(color=BRIDGE_COLOR, size=11),
                bgcolor="rgba(11,4,6,0.9)", bordercolor=BRIDGE_COLOR, borderwidth=1, borderpad=3,
            )

    mark_extremes(fig, entries, unit)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.35)",
        font=dict(color="#efe7dc", size=12),
        margin=dict(l=80, r=30, t=40, b=60),
        height=560,
        hovermode="closest",
        legend=dict(
            bgcolor="rgba(11,4,6,0.7)", bordercolor="rgba(242,193,78,0.3)", borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        xaxis=dict(
            title=dict(text="Temperature, T (K)", standoff=12),
            gridcolor="rgba(242,193,78,0.12)", range=[t_low, t_high],
            automargin=True, ticks="outside", tickcolor="rgba(242,193,78,0.4)",
        ),
        yaxis=dict(
            title=dict(text=f"Heat capacity, Cp ({unit})", standoff=16),
            gridcolor="rgba(242,193,78,0.12)", rangemode="tozero",
            automargin=True, ticks="outside", tickcolor="rgba(242,193,78,0.4)",
        ),
    )
    return fig


class ExplorerApp:

    def __init__(self):
        self.frame = load_database()
        self.catalog = MaterialCatalog(self.frame)
        self.curves = CurveBuilder(self.catalog)
        self.relatives = RelativeFinder(self.catalog)

    # this lived at the bottom. moved it up during the sep4 pass, too lazy to move back
    def render_why_hard_tab(self):
        nist = "Boron Carbide (NIST-JANAF)"
        cea = "Boron Carbide (NASA CEA)"
        if nist in self.catalog.materials and cea in self.catalog.materials:
            gaps = []
            for T in [500, 1000, 1500, 2000, 2500, 2700]:
                a = self.curves.cp_at(nist, T, MOLAR_UNIT)
                b = self.curves.cp_at(cea, T, MOLAR_UNIT)
                if a is not None and b is not None:
                    gaps.append((T, a, b, 100.0 * (b - a) / a))
            worst = max(gaps, key=lambda g: abs(g[3]))
            st.subheader("1. No single authoritative source")
            with st.container(border=True):
                st.markdown(
                    f"Boron carbide from two sources. NIST-JANAF and NASA CEA agree within "
                    f"{abs(gaps[0][3]):.1f}% at {gaps[0][0]} K and diverge to {abs(worst[3]):.0f}% at {worst[0]} K. "
                    "Neither is wrong: they fit different measurements with different functional forms. "
                    "Every material in this database required a source decision. This pair is retained to show the cost of that decision."
                )
                entries = []
                for name in [nist, cea]:
                    pieces, info = self.curves.build(name, 300, 2743, MOLAR_UNIT)
                    entries.append((pieces, info, []))
                st.plotly_chart(compose_figure(entries, MOLAR_UNIT, 300, 2743), width="stretch", theme=None, config=plotly_cfg)

        n_no_mass = sum(self.catalog.first_row(m)["molar_mass"] is None for m in self.catalog.materials)
        st.subheader("2. Molar mass is not always defined")
        with st.container(border=True):
            st.markdown(
                f"{n_no_mass} of {len(self.catalog.materials)} materials are alloys, polymers, composites, glasses, "
                "rocks or mixtures. They have no molecule, so per-mole Cp is undefined and values are given per gram. "
                "Parsing a formula would have produced a number anyway: sandstone lists SiO\u2082 in its composition. "
                "The unit toggle refuses instead."
            )

        n_point = sum(self.catalog.is_point_value(m) for m in self.catalog.materials)
        st.subheader("3. Not every material has a fitted curve")
        with st.container(border=True):
            st.markdown(
                f"{n_point} materials exist in the literature only as a single value near room temperature. "
                "They are drawn as flat dashed lines over a conventional window and labelled as such. "
                "Estimation models (Debye, Neumann\u2013Kopp, van Krevelen) were tested and rejected: each deviated "
                "from measured data by more than the flat line does."
            )

    def render_header(self):
        apply_backdrop()
        st.title("Cp\u2013T materials database")
        st.caption(
            "{} materials, {} categories. ".format(len(self.catalog.materials), len(self.catalog.categories()))
            + "Sources: NIST-JANAF, NASA CEA, handbooks. Curves are drawn only within each fit's valid range."
        )

    def render_sidebar(self):
        if "plotted" not in st.session_state:
            st.session_state["plotted"] = [m for m in ["Copper", "Aluminum", "Titanium"] if m in self.catalog.materials]

        with st.sidebar:
            st.subheader("Materials")
            category_choice = st.selectbox("Category", ["All"] + self.catalog.categories())
            fit_filter = st.radio("Fit type", ["All", "Fitted curves only", "Point values only"], index=0)

            categories = [] if category_choice == "All" else [category_choice]
            options = self.catalog.search("", categories, fit_filter)
            material_choice = st.selectbox(
                "Material", options, index=None, placeholder=f"{len(options)} available"
            )
            if st.button("Add", disabled=material_choice is None):
                if material_choice not in st.session_state["plotted"]:
                    st.session_state["plotted"].append(material_choice)
                if len(st.session_state["plotted"]) > MAX_OVERLAY:
                    st.session_state["plotted"] = st.session_state["plotted"][-MAX_OVERLAY:]
                st.rerun()

            st.subheader("Families")
            families = self.relatives.families()
            family_labels = [f["label"] for f in families]
            family_choice = st.selectbox(
                "Allotropes and phases", family_labels, index=None, placeholder="Species with multiple forms"
            )
            if st.button("Add all forms", disabled=family_choice is None):
                for f in families:
                    if f["label"] == family_choice:
                        for member in f["members"]:
                            if member not in st.session_state["plotted"]:
                                st.session_state["plotted"].append(member)
                st.rerun()

            st.subheader("Plotted")
            st.session_state["plotted"] = st.multiselect(
                "Plotted materials",
                self.catalog.materials,
                default=st.session_state["plotted"],
                placeholder="None",
                label_visibility="collapsed",
            )
            plotted = list(st.session_state["plotted"])

            st.subheader("Conditions")
            if plotted:
                ranges = [self.catalog.valid_range(m) for m in plotted]
                span_low, span_high = min(r[0] for r in ranges), max(r[1] for r in ranges)
            else:
                span_low, span_high = 100.0, 3000.0
            # 5% pad, min 10 K. eyeballed, looks right
            pad = max((span_high - span_low) * 0.05, 10.0)
            step = float(max(round((span_high - span_low) / 200.0), 1))
            t_low, t_high = st.slider(
                "Temperature window (K)",
                min_value=float(max(0.0, span_low - pad)),
                max_value=float(span_high + pad),
                value=(span_low, span_high),
                step=step,
            )
            basis = st.radio("Basis", [MOLAR_UNIT, MASS_UNIT], help="Per-mole requires a molar mass.")
            st.caption("Dashed: single-point values. White verticals: phase transitions with \u0394Cp.")
        return plotted, t_low, t_high, basis

    def render_plot_tab(self, entries, refused, basis, t_low, t_high):
        if refused:
            st.error(
                "No molar mass, not shown per mole: {}. ".format(", ".join(refused))
                + "Switch basis to {} to include.".format(MASS_UNIT)
            )

        clipped = [
            f"{info['material']} ({info['valid_min']:.0f}\u2013{info['valid_max']:.0f} K)"
            for _, info, _ in entries
            if info["clipped_low"] or info["clipped_high"]
        ]
        if clipped:
            st.warning("Window exceeds fitted range, curve truncated: " + "; ".join(clipped))

        if not entries:
            st.info("Nothing to plot for this window and basis.")
            return

        fig = compose_figure(entries, basis, t_low, t_high)
        st.plotly_chart(fig, width="stretch", theme=None, config=plotly_cfg)
        st.caption("Drag to zoom, scroll to zoom, double-click to reset, hover for values.")

        highest, lowest = find_extremes(entries)
        c1, c2 = st.columns(2)
        # indexing twice instead of naming things, sue me
        c1.metric("Max Cp on plot", f"{highest[0]:.4g} {basis}", f"{highest[2]}, {highest[1]:.0f} K", delta_color="off")
        c2.metric("Min Cp on plot", f"{lowest[0]:.4g} {basis}", f"{lowest[2]}, {lowest[1]:.0f} K", delta_color='off')

        st.subheader("Equations")
        for i, (_, info, _) in enumerate(entries, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {info['material']}**")
                for _, row in self.catalog.segments_of(info["material"]).iterrows():
                    st.latex(format_equation_latex(row))
                    st.caption(equation_caption(row))

        st.subheader("Ranking")
        default_T = float(min(max(REFERENCE_T, t_low), t_high))
        T_rank = st.slider("Temperature (K)", float(t_low), float(t_high), default_T)
        value_key = f"Cp at {T_rank:.0f} K ({basis})"
        ranked = []
        outside = []
        for _, info, _ in entries:
            value = self.curves.cp_at(info["material"], T_rank, basis)
            if value is None:
                outside.append(f"{info['material']} ({info['valid_min']:.0f}\u2013{info['valid_max']:.0f} K)")
                continue
            ranked.append({"Material": info["material"], 'Category': info["category"], value_key: round(value, 4)})
        ranked.sort(key=lambda r: r[value_key], reverse=True)
        for position, r in enumerate(ranked, start=1):
            r["#"] = position
        if ranked:
            st.dataframe(pd.DataFrame(ranked)[["#", "Material", "Category", value_key]], hide_index=True, width="stretch")
        if outside:
            st.caption("Not ranked (outside fitted range): %s" % "; ".join(outside))

        st.subheader("Export")
        export_rows = [
            {"Material": info["material"], "T_K": T, f"Cp_{basis}": cp, 'Phase': piece["phase"]}
            for pieces, info, _ in entries
            for piece in pieces
            for T, cp in zip(piece["t"], piece["cp"])
        ]
        csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        c1, c2 = st.columns(2)
        c1.download_button("Plotted data (CSV)", csv_bytes, "cp_vs_T_data.csv", "text/csv")
        c2.download_button("Interactive figure (HTML)", html_bytes, "cp_vs_T.html", "text/html")
        st.caption("PNG: camera icon on the plot toolbar.")

    def render_detail_tab(self, entries, basis):
        if not entries:
            st.info("No materials plotted.")
            return
        names = [info["material"] for _, info, _ in entries]
        chosen = st.selectbox("Material", names)
        for _, info, transitions in entries:
            if info["material"] == chosen:
                self.render_detail_card(info, transitions, basis)

    def render_detail_card(self, info, transitions, basis):
        material = info["material"]
        rows = self.catalog.segments_of(material)
        first = rows.iloc[0]

        ref = self.curves.cp_at(material, REFERENCE_T, basis)
        ref_text = f"{ref:.4g}" if ref is not None else "n/a"
        ref_note = basis if ref is not None else "range starts %.0f K" % info["valid_min"]
        mass_text = f"{first['molar_mass']:.4g} g/mol" if first["molar_mass"] is not None else "undefined"

        c1, c2, c3 = st.columns(3)
        c1.metric("Cp at 298.15 K", ref_text, ref_note, delta_color="off")
        c2.metric("Molar mass", mass_text, first["Molar_Basis"], delta_color="off")
        c3.metric("Valid range", f"{info['valid_min']:.0f}\u2013{info['valid_max']:.0f} K",
                  f"{len(rows)} segment{'s' if len(rows) > 1 else ''}", delta_color="off")

        ext = self.curves.extrema(material, basis)
        if transitions:
            tr_lines = []
            for tr in transitions:
                d = convert_unit(tr["delta_cp"], tr["row_below"], basis)
                what = f"{tr['from_phase']} \u2192 {tr['to_phase']}" if tr["phase_changed"] else "structural"
                tr_lines.append(f"{tr['temperature']:.0f} K, {what}, \u0394Cp {d:+.4g} ({tr['jump_percent']:+.1f}%)")
            tr_text = "; ".join(tr_lines)
        else:
            tr_text = "none"

        with st.container(border=True):
            left, right = st.columns(2)
            left.markdown(
                f"**Formula:** {first['Formula'] or '\u2014'}  \n"
                f"**Composition:** {first['Composition'] or '\u2014'}  \n"
                f"**Category:** {first['Category']}  \n"
                f"**Phase:** {' \u2192 '.join(dict.fromkeys(rows['Phase'].tolist()))}  \n"
                f"**Native units:** {first['Units']}"
            )
            right.markdown(
                f"**Fit:** {' / '.join(dict.fromkeys(rows['Polynomial_Type'].tolist()))}, {first['Fit_Type']}  \n"
                f"**Source:** {' / '.join(dict.fromkeys(rows['Data_Source'].tolist()))}  \n"
                f"**Extremes:** "
                + (f"max {ext['max'][0]:.4g} at {ext['max'][1]:.0f} K, min {ext['min'][0]:.4g} at {ext['min'][1]:.0f} K"
                   if ext else "n/a on this basis") + "  \n"
                f"**Transitions:** {tr_text}"
            )

        with st.container(border=True):
            for _, row in rows.iterrows():
                st.latex(format_equation_latex(row))
                st.caption(equation_caption(row))
        if info["has_point_value"]:
            st.caption("Point value: T_max is a plotting convention, not a sourced bound. Drawn dashed.")

    def execute_pipeline(self):
        st.set_page_config(page_title="Cp\u2013T materials database", page_icon="\u26b1", layout="wide")
        self.render_header()
        plotted, t_low, t_high, basis = self.render_sidebar()

        entries, refused = [], []
        for mat in plotted:
            pieces, info = self.curves.build(mat, t_low, t_high, basis)
            if info["unit_refused"]:
                refused.append(mat)
                continue
            if not pieces:
                continue
            entries.append((pieces, info, find_transitions(self.catalog.segments_of(mat))))

        tab_plot, tab_detail, tab_why = st.tabs(["Plot", "Details", "Why this is hard"])
        with tab_plot:
            self.render_plot_tab(entries, refused, basis, t_low, t_high)
        with tab_detail:
            self.render_detail_tab(entries, basis)
        with tab_why:
            self.render_why_hard_tab()


def main():
    session_runner = ExplorerApp()
    session_runner.execute_pipeline()


if __name__ == "__main__":
    main()
