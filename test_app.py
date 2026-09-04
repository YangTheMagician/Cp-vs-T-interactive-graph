"""Tests for the Cp(T) explorer. Run:  py -m pytest test_app.py -q"""
import os
import pytest
from streamlit.testing.v1 import AppTest
import app as A

frame = A.load_database.__wrapped__()
catalog = A.MaterialCatalog(frame)
curves = A.CurveBuilder(catalog)
relatives = A.RelativeFinder(catalog)


def test_database_shape():
    assert len(frame) == 282
    assert len(catalog.materials) == 209
    assert len(catalog.categories()) == 11


def test_nan_token_is_absent():
    for column in ["Formula", "Composition", "Molar_Mass", "Molar_Basis"]:
        for value in frame[column]:
            assert value != "NaN"


def test_unit_gate_invariant():
    for _, row in frame.iterrows():
        has_formula = row["Formula"] is not None
        has_mass = row["molar_mass"] is not None
        assert has_formula == has_mass, row["Material"]
        assert (row["Units"] == A.MOLAR_UNIT) == has_mass, row["Material"]


@pytest.mark.parametrize("name,T,expected", [
    ("Helium", 300.0, 20.79), ("Nitrogen", 298.15, 29.12),
    ("Water (Liquid)", 298.15, 75.37), ("Aluminum", 298.15, 24.21),
    ("Copper", 298.15, 24.47), ("Iron", 298.15, 25.10),
    ("Graphite", 298.15, 8.53), ("Tungsten Carbide", 298.15, 39.8),
])
def test_known_values(name, T, expected):
    got = curves.cp_at(name, T, A.MOLAR_UNIT)
    assert got is not None and abs(got - expected) / expected < 0.02, (name, got)


def test_no_nonphysical_values_anywhere():
    for _, row in frame.iterrows():
        lo, hi = float(row["T_min"]), float(row["T_max"])
        for k in range(100):
            T = lo + (hi - lo) * k / 99
            v = A.evaluate_cp(row, T)
            assert v > 0, (row["Material"], T, v)
            assert v < (5.0 if row["Units"] == A.MASS_UNIT else 800.0), (row["Material"], T, v)


def test_never_extrapolates():
    pieces, info = curves.build("Copper", 1.0, 9000.0, A.MOLAR_UNIT)
    lo = min(min(p["t"]) for p in pieces)
    hi = max(max(p["t"]) for p in pieces)
    assert lo >= info["valid_min"] and hi <= info["valid_max"]
    assert info["clipped_low"] and info["clipped_high"]
    assert curves.cp_at("Copper", 5000.0, A.MOLAR_UNIT) is None


def test_molar_refused_without_molar_mass():
    _, info = curves.build("Stainless Steel 304", 300, 800, A.MOLAR_UNIT)
    assert info["unit_refused"]
    pieces, info = curves.build("Stainless Steel 304", 300, 800, A.MASS_UNIT)
    assert not info["unit_refused"] and pieces


def test_mass_conversion_matches_molar_mass():
    m = catalog.first_row("Copper")["molar_mass"]
    per_g = curves.cp_at("Copper", 500, A.MASS_UNIT)
    per_mol = curves.cp_at("Copper", 500, A.MOLAR_UNIT)
    assert abs(per_g * m - per_mol) < 1e-9


def test_transitions_fourteen_and_placed():
    total = 0
    for m in catalog.materials:
        total += len(A.find_transitions(catalog.segments_of(m)))
    assert total == 14
    ti = A.find_transitions(catalog.segments_of("Titanium"))
    assert [round(t["temperature"]) for t in ti] == [1155]
    assert abs(ti[0]["jump_percent"]) > 15
    u = A.find_transitions(catalog.segments_of("Uranium"))
    assert [round(t["temperature"]) for t in u] == [942, 1049]


def test_mullite_now_stitches():
    rows = catalog.segments_of("Mullite")
    assert len(rows) == 2
    assert A.find_transitions(rows) == []  # smooth join, not a transition


def test_segments_contiguous():
    for m in catalog.materials:
        rows = catalog.segments_of(m)
        for i in range(len(rows) - 1):
            assert rows.iloc[i]["T_max"] == rows.iloc[i + 1]["T_min"], m


def test_relatives_water_and_boron_carbide():
    names = sorted(r["material"] for r in relatives.relatives_of("Water (Liquid)"))
    assert names == ["Water (Ice -10C)", "Water (Steam 100C)"]
    kinds = {r["material"]: r["kind"] for r in relatives.relatives_of("Water (Liquid)")}
    assert all(k == "another phase" for k in kinds.values())
    bc = relatives.relatives_of("Boron Carbide (NIST-JANAF)")
    assert bc[0]["material"] == "Boron Carbide (NASA CEA)" and bc[0]["kind"] == "another form or source"
    assert relatives.relatives_of("Stainless Steel 304") == []


def test_equation_latex():
    row = catalog.first_row("Copper")
    tex = A.format_equation_latex(row)
    assert tex.startswith("C_p = ") and f"{row['C1']:.4g}" in tex and "t^{-2}" in tex
    assert "t = T/1000" in A.equation_caption(row)
    he = catalog.first_row("Helium")
    assert A.format_equation_latex(he).startswith("C_p = R\\left[")
    assert "\\times10^{" in A.format_equation_latex(he)
    peek = catalog.first_row("PEEK")
    assert A.format_equation_latex(peek) == f"C_p = {peek['C1']:.4g}"
    assert "Single measured point" in A.equation_caption(peek)


def test_extrema():
    ext = curves.extrema("Titanium", A.MOLAR_UNIT)
    assert ext["max"][0] > ext["min"][0]
    assert curves.extrema("Stainless Steel 304", A.MOLAR_UNIT) is None


def test_search_filters():
    assert "Titanium" in catalog.search("tita", [], "All")
    assert "Stainless Steel 304" not in catalog.search("", ["Polymer"], "All")
    for m in catalog.search("", [], "Fitted curves only"):
        assert not catalog.is_point_value(m)


def test_app_runs_and_widgets_drive_it():
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    at.sidebar.multiselect[0].set_value(["Titanium", "Stainless Steel 304", "Water (Liquid)"]).run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "Stainless Steel 304" in errors and "molar mass" in errors.lower()
    at.sidebar.radio[1].set_value(A.MASS_UNIT).run()
    assert not at.exception


def test_dropdown_add_button():
    at = AppTest.from_file("app.py", default_timeout=120).run()
    at.sidebar.selectbox[0].set_value("Polymer").run()
    at.sidebar.selectbox[1].set_value("PEEK").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert "PEEK" in at.session_state["plotted"]


def test_ten_material_overlay():
    picks = ["Copper", "Aluminum", "Titanium", "Iron", "Stainless Steel 304", "PEEK",
             "Water (Liquid)", "Graphite", "Uranium", "Helium"]
    at = AppTest.from_file("app.py", default_timeout=120).run()
    at.sidebar.multiselect[0].set_value(picks).run()
    assert not at.exception
    at.sidebar.radio[1].set_value(A.MASS_UNIT).run()
    assert not at.exception


def test_families_dropdown():
    fams = relatives.families()
    labels = [f["formula"] for f in fams]
    assert "H2O" in labels and "C6H6" in labels and "C" in labels and "B4C" in labels
    water = [f for f in fams if f["formula"] == "H2O"][0]
    assert sorted(water["members"]) == ["Water (Ice -10C)", "Water (Liquid)", "Water (Steam 100C)"]
    at = AppTest.from_file("app.py", default_timeout=120).run()
    at.sidebar.selectbox[2].set_value(water["label"]).run()
    at.sidebar.button[1].click().run()
    assert not at.exception
    for m in water["members"]:
        assert m in at.session_state["plotted"]


def test_ranking_and_extremes_on_plot_page():
    at = AppTest.from_file("app.py", default_timeout=120).run()
    at.sidebar.multiselect[0].set_value(["Copper", "Titanium", "Helium"]).run()
    assert not at.exception
    assert len(at.dataframe) >= 1
    labels = [m.label for m in at.metric]
    assert "Max Cp on plot" in labels and "Min Cp on plot" in labels
    assert len(at.latex) >= 3


def test_extremes_helper():
    entries = []
    for m in ["Copper", "Titanium"]:
        pieces, info = curves.build(m, 300, 1500, A.MOLAR_UNIT)
        entries.append((pieces, info, []))
    hi, lo = A.find_extremes(entries)
    assert hi[0] > lo[0] and hi[2] in ("Copper", "Titanium")


def test_backdrop_is_noop_without_image():
    assert not os.path.exists(os.path.join(A.HERE, A.BACKDROP_NAME))
