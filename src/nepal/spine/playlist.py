"""Music from a playlist export instead of audio files.

The delivered ``music/`` folder holds a playlist CSV and no audio. That is not
a dead end: playlist exports from Spotify (via Exportify and friends) carry
per-track audio features -- energy, tempo, loudness, danceability, key, mode,
acousticness, instrumentalness -- which is most of what S02.7's act assignment
needs. So the assignment can run before a single file is downloaded.

What a playlist cannot give is dynamic range. There is no Spotify feature for
"quiet passages against loud ones", and Act 3's target leans on it heavily
("**high**" in the spec's table). Rather than invent a proxy and pretend, the
dimension is masked out of the distance and the remaining features are
reweighted -- and the report says so, so nobody reads the assignment as more
confident than it is.

Column names vary by exporter, so detection is by fuzzy match rather than a
fixed schema.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from nepal.spine.music import PITCH_CLASSES, Track

log = logging.getLogger(__name__)

# Each logical field maps to the substrings an exporter might use for it.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "title":       ("track name", "trackname", "title", "song", "name"),
    "artist":      ("artist name", "artist(s)", "artist names", "artists", "artist", "album artist"),
    "album":       ("album name", "album"),
    "duration_ms": ("duration (ms)", "duration_ms", "durationms", "track duration (ms)"),
    "duration_s":  ("duration (s)", "duration_s", "length", "time", "duration"),
    "tempo":       ("tempo", "bpm"),
    "energy":      ("energy",),
    "loudness":    ("loudness",),
    "danceability": ("danceability",),
    "speechiness": ("speechiness",),
    "acousticness": ("acousticness",),
    "instrumentalness": ("instrumentalness",),
    "liveness":    ("liveness",),
    "valence":     ("valence",),
    "key":         ("key",),
    "mode":        ("mode",),
    "isrc":        ("isrc",),
    "uri":         ("track uri", "uri", "spotify uri", "url", "link"),
}

# Features S02.7 wants, and whether a playlist can supply them.
PLAYLIST_FEATURE_MASK = {
    "energy_mean": True,
    "dyn_range": False,      # no playlist analogue -- masked, not faked
    "centroid": True,        # approximated from acousticness
    "onset_rate": True,      # approximated from danceability and tempo
    "tempo_bpm": True,
}


CYRILLIC_RE = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")

# Best-effort list of Russophone acts that render their names in Latin script,
# where Cyrillic detection cannot help. Deliberately not exhaustive -- it exists
# to raise recall, and every exclusion is reported so the operator can correct
# both directions. Extend via music.exclude_artists in the config.
RU_LATIN_ARTISTS = frozenset({
    "molchat doma", "kino", "viktor tsoi", "tsoi", "bi-2", "bi2", "zemfira",
    "splean", "spleen", "aquarium", "akvarium", "mumiy troll", "mumiy trol",
    "nautilus pompilius", "leningrad", "ddt", "alisa", "agatha christie",
    "grazhdanskaya oborona", "civil defense", "egor letov", "yanka diaghileva",
    "pornofilmy", "lumen", "korol i shut", "king and the clown", "pilot",
    "chizh", "chaif", "mashina vremeni", "time machine", "kipelov", "aria",
    "arya", "louna", "tequilajazzz", "auktyon", "auktsyon", "zveri", "mumiytroll",
    "basta", "oxxxymiron", "oxxxxymiron", "husky", "khaski", "monetochka",
    "grechka", "lucidvox", "shortparis", "gsh", "glintshake", "ic3peak",
    "little big", "kis-kis", "tatu", "t.a.t.u.", "nogu svelo", "bravo",
    "sektor gaza", "krematorij", "krematorium", "gorshok", "noize mc",
    "kasta", "25/17", "krovostok", "scriptonite", "skriptonit", "miyagi",
    "pharaoh", "face", "kizaru", "lsp", "dolphin", "delfin", "mgzavrebi",
    "sirotkin", "buerak", "pasosh", "spasibo", "sonic death", "utro",
    "electroforez", "elektroforez", "mnogoznaal", "kate nv", "kedr livanskiy",
    "nina kraviz", "gone.fludd", "thomas mraz", "obladaet", "sqwoz bab",
})


@dataclass
class ExclusionRule:
    """One decision about one track, kept so the operator can audit it."""
    track_id: str
    title: str
    artist: str | None
    excluded: bool
    reason: str


@dataclass
class PlaylistReport:
    n_rows: int = 0
    n_tracks: int = 0
    has_audio_features: bool = False
    columns_found: dict[str, str] = field(default_factory=dict)
    missing_features: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    n_excluded: int = 0
    exclusions: list[ExclusionRule] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9() ]+", " ", str(s).strip().lower()).strip()


def detect_columns(header: Sequence[str]) -> dict[str, str]:
    """Map logical field -> actual header name, by longest-alias match."""
    normalised = {_norm(h): h for h in header if h}
    found: dict[str, str] = {}
    for field_name, aliases in COLUMN_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            for norm_h, original in normalised.items():
                if norm_h == alias:
                    found[field_name] = original
                    break
            if field_name in found:
                break
        if field_name in found:
            continue
        # fall back to a substring hit, longest alias first
        for alias in sorted(aliases, key=len, reverse=True):
            for norm_h, original in normalised.items():
                if alias in norm_h and original not in found.values():
                    found[field_name] = original
                    break
            if field_name in found:
                break
    return found


def spotify_key(key_value: Any, mode_value: Any) -> str | None:
    """Spotify pitch class (0-11) plus mode (1 major, 0 minor) -> 'Amin'.

    This is what makes the Act 5 callback bonus work from playlist data alone:
    the spec wants Act 5 to share a key or artist with Act 1, and both are
    available here.
    """
    k = _num(key_value)
    if k is None or not (0 <= int(k) <= 11):
        return None
    m = _num(mode_value)
    suffix = "maj" if m is None or int(m) == 1 else "min"
    return f"{PITCH_CLASSES[int(k)]}{suffix}"


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def _first_artist(value: str | None) -> str | None:
    """Exporters join collaborators with commas or semicolons; the first name
    is the one the Act 1/Act 5 callback should match on."""
    if not value:
        return None
    parts = re.split(r"\s*[,;]\s*|\s+&\s+|\s+feat\.?\s+", str(value), flags=re.I)
    return parts[0].strip() or None


def _read_source(source: str | Path) -> str:
    """Accept either a path or inline CSV text.

    Testing a candidate with ``Path(s).exists()`` is not safe here: a long CSV
    string raises OSError (filename too long) rather than returning False, and
    embedded NULs raise ValueError. Content is recognised first -- any CSV has
    a newline and a path essentially never does -- and the filesystem is only
    consulted for short, plausible names, inside a guard.
    """
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8-sig", errors="replace")
    text = str(source)
    if "\n" in text or "\r" in text or len(text) > 4096:
        return text
    try:
        candidate = Path(text)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8-sig", errors="replace")
    except (OSError, ValueError):
        pass
    return text


def is_russian(title: str, artist: str | None, *,
               extra_artists: Iterable[str] = (),
               keep_artists: Iterable[str] = ()) -> tuple[bool, str]:
    """Whether a track is Russian-language, and on what evidence.

    Cyrillic in the title or artist is decisive. Beyond that, a curated list
    covers acts that romanise their names, which is where this becomes
    best-effort rather than exact -- transliteration has no reliable signature,
    and guessing from name endings would sweep up Ukrainian, Polish and Balkan
    artists along with false positives. So the verdict is always returned with
    its reason, and ``keep_artists`` overrides it.
    """
    keep = {k.strip().lower() for k in keep_artists if k}
    a_low = (artist or "").strip().lower()
    if a_low and a_low in keep:
        return False, f"kept explicitly: {artist}"

    if artist and CYRILLIC_RE.search(artist):
        return True, f"Cyrillic in artist: {artist}"
    if title and CYRILLIC_RE.search(title):
        return True, f"Cyrillic in title: {title}"

    known = RU_LATIN_ARTISTS | {e.strip().lower() for e in extra_artists if e}
    if a_low and a_low in known:
        return True, f"known Russophone artist: {artist}"
    # a listed artist appearing inside a longer credit string
    for name in known:
        if len(name) >= 5 and a_low and name in a_low:
            return True, f"known Russophone artist matched in credit: {artist}"
    return False, ""


def parse_playlist(source: str | Path, *, licence: str = "personal",
                   exclude_russian: bool = False,
                   exclude_artists: Iterable[str] = (),
                   keep_artists: Iterable[str] = (),
                   min_instrumentalness: float | None = None
                   ) -> tuple[list[Track], PlaylistReport]:
    """Read a playlist CSV into Track rows plus a report on what was usable."""
    text = _read_source(source)

    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)

    header = reader.fieldnames or []
    cols = detect_columns(header)
    report = PlaylistReport(columns_found=cols)

    if "title" not in cols:
        report.notes.append(
            f"no recognisable title column in {header[:8]} -- cannot read this playlist")
        return [], report

    feature_cols = {"energy", "tempo", "loudness", "danceability", "acousticness"}
    report.has_audio_features = bool(feature_cols & set(cols))

    tracks: list[Track] = []
    seen: set[str] = set()
    for row in reader:
        report.n_rows += 1
        title = (row.get(cols["title"]) or "").strip()
        if not title:
            continue
        artist = _first_artist(row.get(cols.get("artist", ""), ""))

        dur = _num(row.get(cols.get("duration_s", ""), ""))
        dur_ms = _num(row.get(cols.get("duration_ms", ""), ""))
        if dur_ms:
            dur = dur_ms / 1000.0
        if dur is None:
            dur = 0.0

        energy = _num(row.get(cols.get("energy", ""), ""))
        tempo = _num(row.get(cols.get("tempo", ""), ""))
        loudness = _num(row.get(cols.get("loudness", ""), ""))
        dance = _num(row.get(cols.get("danceability", ""), ""))
        acoustic = _num(row.get(cols.get("acousticness", ""), ""))

        track_id = _slug(f"{artist or 'unknown'}-{title}")
        if track_id in seen:
            continue
        seen.add(track_id)

        drop, reason = (False, "")
        if exclude_russian:
            drop, reason = is_russian(title, artist, extra_artists=exclude_artists,
                                      keep_artists=keep_artists)
        instr = _num(row.get(cols.get("instrumentalness", ""), ""))
        if not drop and min_instrumentalness is not None and instr is not None \
                and instr < min_instrumentalness:
            drop = True
            reason = f"instrumentalness {instr:.2f} below {min_instrumentalness:.2f}"
        report.exclusions.append(ExclusionRule(track_id, title, artist, drop, reason))
        if drop:
            report.n_excluded += 1
            continue

        t = Track(
            track_id=track_id,
            s3_key="",                      # no file yet
            title=title,
            artist=artist,
            duration_s=round(float(dur), 3),
            tempo_bpm=float(tempo) if tempo is not None else 0.0,
            key_est=spotify_key(row.get(cols.get("key", ""), ""),
                                row.get(cols.get("mode", ""), "")),
            licence=licence,
        )
        # Map playlist features onto the five S02.7 dimensions.
        t.energy_mean = float(energy) if energy is not None else _energy_from_loudness(loudness)
        # brightness: acoustic material is darker than produced material
        t.centroid = (1.0 - acoustic) if acoustic is not None else 0.5
        # rhythmic activity: danceability is the closest available signal
        t.onset_rate = float(dance) if dance is not None else 0.5
        t.energy_p95 = t.energy_mean          # placeholders; dyn_range is masked
        t.energy_p10 = t.energy_mean
        tracks.append(t)

    report.n_tracks = len(tracks)
    if report.n_excluded:
        report.notes.append(
            f"{report.n_excluded} track(s) excluded; every decision is listed in "
            f"exclusions with its reason. Cyrillic detection is exact, the "
            f"romanised-artist list is best-effort -- review it and use "
            f"music.keep_artists / music.exclude_artists to correct either way")
    report.missing_features = [k for k, ok in PLAYLIST_FEATURE_MASK.items() if not ok]
    if report.has_audio_features:
        report.notes.append(
            "audio features read from the playlist; dynamic range is unavailable "
            "and is masked out of the act distance rather than approximated")
    else:
        report.notes.append(
            "playlist carries names only, no audio features. It fixes the running "
            "order but cannot drive act assignment -- supply audio files, or the "
            "cleared reference palette, to score the acts")
    log.info("playlist: %d tracks from %d rows, audio features=%s",
             report.n_tracks, report.n_rows, report.has_audio_features)
    return tracks, report


def _energy_from_loudness(loudness_db: float | None) -> float:
    """Spotify loudness runs about -60..0 dBFS. Used only when `energy` is
    absent, mapped onto 0..1 so it lands in the same range."""
    if loudness_db is None:
        return 0.5
    return max(0.0, min(1.0, (float(loudness_db) + 60.0) / 60.0))


def _slug(s: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")[:80]


def feature_mask_for(source: str) -> "list[float]":
    """Per-dimension weights for the act distance, given where features came
    from. ``audio`` uses everything; ``playlist`` masks dynamic range."""
    from nepal.spine.music import FEATURES
    if source == "playlist":
        return [1.0 if PLAYLIST_FEATURE_MASK.get(f, True) else 0.0 for f in FEATURES]
    return [1.0] * len(FEATURES)
