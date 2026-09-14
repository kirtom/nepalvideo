import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest
from nepal.spine.geocode import Place, Gazetteer, load_geonames, cluster_coords

# name, lat, lon, class, code, population
FIXTURE = [
    ("Namche Bazaar", 27.8069, 86.7133, "P", "PPL", 1647),
    ("Tengboche",     27.8361, 86.7644, "P", "PPL", 100),
    ("Kala Patthar",  27.9950, 86.8280, "T", "MT",  0),
    ("Lobuche",       27.9500, 86.8100, "P", "PPL", 65),
    ("Khumbu Glacier",27.9700, 86.8300, "H", "GLCR", 0),
    ("Unnamed Spur",  27.8070, 86.7134, "T", "SPUR", 0),   # 12 m from Namche
]


def geonames_text(rows):
    out = []
    for i, (name, lat, lon, cls, code, pop) in enumerate(rows):
        f = [""] * 19
        f[0], f[1], f[2] = str(i), name, name
        f[4], f[5], f[6], f[7] = str(lat), str(lon), cls, code
        f[14] = str(pop)
        out.append("\t".join(f))
    return "\n".join(out)


@pytest.fixture
def dump(tmp_path):
    p = tmp_path / "NP.txt"
    p.write_text(geonames_text(FIXTURE), encoding="utf-8")
    return p


def test_load_geonames(dump):
    places = load_geonames(dump)
    assert len(places) == len(FIXTURE)
    assert {p.name for p in places} >= {"Namche Bazaar", "Kala Patthar"}


def test_load_missing_file_is_not_fatal(tmp_path):
    assert load_geonames(tmp_path / "absent.txt") == []


def test_load_from_zip(tmp_path):
    import zipfile
    z = tmp_path / "NP.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("NP.txt", geonames_text(FIXTURE))
        zf.writestr("readme.txt", "ignore me")
    assert len(load_geonames(z)) == len(FIXTURE)


def test_finds_the_obvious_village(dump):
    g = Gazetteer(load_geonames(dump))
    assert g.nearest(27.8069, 86.7133).name == "Namche Bazaar"


def test_named_village_beats_a_closer_unnamed_spur(dump):
    """A spot height 12 m away is a worse answer than the village you are
    standing in and whose name appears throughout the chat."""
    g = Gazetteer(load_geonames(dump))
    assert g.nearest(27.80695, 86.71335).name == "Namche Bazaar"


def test_peak_is_returned_when_nothing_populated_is_near(dump):
    g = Gazetteer(load_geonames(dump))
    assert g.nearest(27.9951, 86.8281).name == "Kala Patthar"


def test_nothing_within_range_returns_none(dump):
    g = Gazetteer(load_geonames(dump))
    assert g.nearest(20.0, 80.0, max_m=5000) is None


def test_empty_gazetteer_is_safe():
    assert Gazetteer([]).nearest(27.8, 86.7) is None


def test_population_breaks_ties_between_settlements():
    places = [Place("Big", 27.80, 86.70, "P", "PPL", 20000),
              Place("Tiny", 27.8005, 86.70, "P", "PPL", 12)]
    # Tiny is closer, but Big is a real settlement
    assert Gazetteer(places).nearest(27.8003, 86.70).name == "Big"


# -- clustering --------------------------------------------------------

def test_clusters_nearby_coordinates():
    coords = [(27.8069, 86.7133), (27.8070, 86.7134), (27.9950, 86.8280)]
    clusters = cluster_coords(coords, radius_m=500)
    assert len(clusters) == 2
    sizes = sorted(len(idxs) for _, idxs in clusters)
    assert sizes == [1, 2]


def test_cluster_membership_indices_are_preserved():
    coords = [(27.80, 86.71), (27.99, 86.82), (27.80, 86.71)]
    clusters = cluster_coords(coords, radius_m=500)
    by_size = sorted(clusters, key=lambda c: -len(c[1]))
    assert sorted(by_size[0][1]) == [0, 2]


def test_empty_input():
    assert cluster_coords([]) == []
