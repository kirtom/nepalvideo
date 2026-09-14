import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.spine.playlist import (detect_columns, spotify_key, parse_playlist,
                                  feature_mask_for, _first_artist,
                                  _energy_from_loudness)
from nepal.spine.music import assign_acts, _resolve_mask, FEATURES

EXPORTIFY_HEADER = ("Track URI,Track Name,Artist Name(s),Album Name,Duration (ms),"
                    "Tempo,Energy,Loudness,Danceability,Acousticness,Instrumentalness,"
                    "Valence,Key,Mode")


def csv_of(rows, header=EXPORTIFY_HEADER):
    return header + "\n" + "\n".join(rows) + "\n"


def row(title, artist, ms=200000, tempo=90, energy=0.4, loud=-12, dance=0.3,
        acoustic=0.6, instr=0.9, val=0.3, key=9, mode=0):
    return (f"spotify:track:x,{title},{artist},Album,{ms},{tempo},{energy},{loud},"
            f"{dance},{acoustic},{instr},{val},{key},{mode}")


# -- column detection --------------------------------------------------

def test_detects_exportify_columns():
    cols = detect_columns(EXPORTIFY_HEADER.split(","))
    assert cols["title"] == "Track Name"
    assert cols["artist"] == "Artist Name(s)"
    assert cols["duration_ms"] == "Duration (ms)"
    assert cols["energy"] == "Energy"
    assert cols["instrumentalness"] == "Instrumentalness"


def test_detects_alternative_namings():
    cols = detect_columns(["Song", "Artist", "BPM", "Length"])
    assert cols["title"] == "Song" and cols["artist"] == "Artist"
    assert cols["tempo"] == "BPM"


def test_title_and_album_are_not_confused():
    """'Album Name' contains 'name', which must not win the title slot."""
    cols = detect_columns(["Track Name", "Album Name"])
    assert cols["title"] == "Track Name"
    assert cols["album"] == "Album Name"


# -- keys --------------------------------------------------------------

@pytest.mark.parametrize("key,mode,expected", [
    (9, 0, "Amin"), (0, 1, "Cmaj"), (7, 1, "Gmaj"), (11, 0, "Bmin"),
])
def test_spotify_key_mapping(key, mode, expected):
    assert spotify_key(key, mode) == expected


def test_spotify_key_out_of_range_or_missing():
    assert spotify_key(-1, 1) is None
    assert spotify_key("", "") is None


# -- parsing -----------------------------------------------------------

def test_parses_tracks_and_features():
    text = csv_of([row("Says", "Nils Frahm", ms=180000, tempo=62, energy=0.12)])
    tracks, rep = parse_playlist(text)
    assert rep.n_tracks == 1 and rep.has_audio_features
    t = tracks[0]
    assert t.title == "Says" and t.artist == "Nils Frahm"
    assert t.duration_s == pytest.approx(180.0)
    assert t.tempo_bpm == pytest.approx(62.0)
    assert t.energy_mean == pytest.approx(0.12)
    assert t.key_est == "Amin"


def test_dynamic_range_is_reported_as_unavailable():
    tracks, rep = parse_playlist(csv_of([row("Says", "Nils Frahm")]))
    assert "dyn_range" in rep.missing_features
    assert any("dynamic range" in n for n in rep.notes)


def test_instrumentalness_filter():
    text = csv_of([row("Vocal Track", "Someone", instr=0.02),
                   row("Ambient", "Someone Else", instr=0.95)])
    tracks, rep = parse_playlist(text, min_instrumentalness=0.5)
    assert [t.title for t in tracks] == ["Ambient"]
    assert "instrumentalness" in [e.reason.split()[0] for e in rep.exclusions if e.excluded]


def test_names_only_playlist_is_flagged():
    text = csv_of(["Says,Nils Frahm", "Ambre,Nils Frahm"], header="Track Name,Artist")
    tracks, rep = parse_playlist(text)
    assert rep.n_tracks == 2
    assert not rep.has_audio_features
    assert any("names only" in n for n in rep.notes)


def test_duplicates_are_collapsed():
    text = csv_of([row("Says", "Nils Frahm"), row("Says", "Nils Frahm")])
    tracks, _ = parse_playlist(text)
    assert len(tracks) == 1


def test_unreadable_header_reports_rather_than_crashes():
    tracks, rep = parse_playlist("colA,colB\n1,2\n")
    assert tracks == []
    assert any("no recognisable title" in n for n in rep.notes)


def test_semicolon_delimited_export():
    text = "Track Name;Artist;Tempo\nSays;Nils Frahm;62\n"
    tracks, rep = parse_playlist(text)
    assert rep.n_tracks == 1 and tracks[0].tempo_bpm == pytest.approx(62.0)


def test_reads_from_a_file(tmp_path):
    p = tmp_path / "rinse_and_repeat.csv"
    p.write_text(csv_of([row("Says", "Nils Frahm")]), encoding="utf-8")
    tracks, rep = parse_playlist(p)
    assert rep.n_tracks == 1


def test_bom_is_stripped(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("﻿" + csv_of([row("Says", "Nils Frahm")]), encoding="utf-8")
    tracks, _ = parse_playlist(p)
    assert tracks and tracks[0].title == "Says"


# -- helpers -----------------------------------------------------------

@pytest.mark.parametrize("raw,first", [
    ("Nils Frahm", "Nils Frahm"),
    ("Nils Frahm, Ólafur Arnalds", "Nils Frahm"),
    ("A & B", "A"),
    ("X feat. Y", "X"),
    (None, None),
])
def test_first_artist(raw, first):
    assert _first_artist(raw) == first


def test_energy_from_loudness_is_bounded():
    assert _energy_from_loudness(-60) == pytest.approx(0.0)
    assert _energy_from_loudness(0) == pytest.approx(1.0)
    assert _energy_from_loudness(-999) == 0.0
    assert _energy_from_loudness(None) == 0.5


# -- mask integration --------------------------------------------------

def test_playlist_mask_zeroes_dynamic_range():
    mask = feature_mask_for("playlist")
    assert mask[FEATURES.index("dyn_range")] == 0.0
    assert mask[FEATURES.index("energy_mean")] == 1.0


def test_resolved_mask_keeps_total_weight_constant():
    """Masking a dimension must not simply shrink every distance."""
    full = _resolve_mask(None)
    masked = _resolve_mask(feature_mask_for("playlist"))
    assert sum(full) == pytest.approx(sum(masked))


def test_masked_dimension_cannot_influence_the_assignment():
    """Two tracks identical except in the masked dimension must be
    interchangeable -- otherwise a placeholder value is steering the film."""
    text = csv_of([
        row("piano", "A", tempo=60, energy=0.05, acoustic=0.95, dance=0.1),
        row("strings", "B", tempo=90, energy=0.30, acoustic=0.50, dance=0.4),
        row("climb", "C", tempo=95, energy=0.45, acoustic=0.40, dance=0.5),
        row("peak", "D", tempo=140, energy=0.85, acoustic=0.05, dance=0.8),
        row("return", "E", tempo=62, energy=0.15, acoustic=0.90, dance=0.15),
    ])
    tracks, _ = parse_playlist(text)
    for t in tracks:                      # a nonsense dyn_range in every track
        t.energy_p95, t.energy_p10 = 99.0, -99.0
    a = assign_acts(tracks, feature_mask=feature_mask_for("playlist"))
    assert a.by_act[1] == "a-piano"
    assert a.by_act[4] == "d-peak"


def test_resolve_mask_rejects_a_fully_masked_vector():
    with pytest.raises(ValueError):
        _resolve_mask([0, 0, 0, 0, 0])


def test_resolve_mask_rejects_wrong_length():
    with pytest.raises(ValueError):
        _resolve_mask([1, 1])


# -- identifier columns ------------------------------------------------

from nepal.spine.playlist import looks_like_identifier, _is_identifier_column

EXPORTIFY_WITH_URIS = ("Track URI,Track Name,Artist URI(s),Artist Name(s),Album Name,"
                       "Duration (ms),Tempo,Energy,Loudness,Danceability,Acousticness,"
                       "Instrumentalness,Valence,Key,Mode")


def test_artist_name_column_beats_artist_uri_column():
    """The bug this guards: a substring match on 'artist' happily took
    'Artist URI(s)', producing track ids like
    'spotify-artist-1ghphrq36vkcy3ucvazcfo-go' and silently disabling the Act 5
    callback, which matches on artist."""
    cols = detect_columns(EXPORTIFY_WITH_URIS.split(","))
    assert cols["artist"] == "Artist Name(s)"
    assert cols["title"] == "Track Name"


def test_a_uri_only_export_yields_no_artist_rather_than_a_uri():
    hdr = "Track URI,Track Name,Artist URI(s),Album Name,Tempo,Energy,Key,Mode"
    cols = detect_columns(hdr.split(","))
    assert cols.get("artist") is None


def test_uri_values_are_discarded_and_reported():
    hdr = "Track Name,Artist,Tempo,Energy,Key,Mode"
    text = (hdr + "\nGo,spotify:artist:1GhPHrq36vKCY3UcVAzCFo,120,0.8,9,0\n")
    tracks, rep = parse_playlist(text)
    assert tracks[0].artist is None
    assert "spotify" not in tracks[0].track_id
    assert any("identifier" in n for n in rep.notes)


def test_rows_whose_title_is_an_identifier_are_skipped():
    hdr = "Track Name,Artist Name(s),Tempo"
    text = hdr + "\nspotify:track:1GhPHrq36vKCY3UcVAzCFo,Someone,120\nSays,Nils Frahm,62\n"
    tracks, _ = parse_playlist(text)
    assert [t.title for t in tracks] == ["Says"]


@pytest.mark.parametrize("value,expected", [
    ("spotify:artist:1GhPHrq36vKCY3UcVAzCFo", True),
    ("https://open.spotify.com/artist/x", True),
    ("1GhPHrq36vKCY3UcVAzCFo", True),
    ("Nils Frahm", False), ("Bi-2", False), ("", False), (None, False),
])
def test_looks_like_identifier(value, expected):
    assert looks_like_identifier(value) is expected


@pytest.mark.parametrize("header,expected", [
    ("Artist URI(s)", True), ("Track URI", True), ("Spotify ID", True),
    ("ISRC", True), ("Artist Name(s)", False), ("Track Name", False),
    ("Danceability", False),
])
def test_is_identifier_column(header, expected):
    assert _is_identifier_column(header) is expected


def test_real_exportify_header_parses_names_not_uris():
    text = (EXPORTIFY_WITH_URIS + "\n"
            "spotify:track:aaa,Says,spotify:artist:bbb,Nils Frahm,Felt,275000,62,"
            "0.09,-22,0.2,0.95,0.94,0.3,9,0\n")
    tracks, rep = parse_playlist(text)
    assert tracks[0].artist == "Nils Frahm"
    assert tracks[0].title == "Says"
    assert tracks[0].track_id == "nils-frahm-says"
    assert not any("identifier" in n for n in rep.notes)
