"""S01.1 -- build the asset manifest from the delivered tree.

Classification is a pure function of the relative path plus extension, so it is
testable without any media present. exiftool is invoked once over the whole
tree (recursive, JSON, numeric) rather than per file; on a few thousand files
that is the difference between seconds and many minutes of process spawning.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

log = logging.getLogger(__name__)

VIDEO360_EXT = {".insv", ".lrv"}
VIDEO_FLAT_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
PHOTO_EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp", ".insp", ".dng"}
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
DOC_EXT = {".pdf", ".gpx", ".kml", ".json", ".txt", ".doc", ".docx", ".csv"}

# Nepal runs at UTC+05:45 -- an offset that trips naive parsers that assume
# whole hours. Device timestamps that carry no zone are assumed to be local to
# wherever the device was set, and corrected by S01.5, not here.
NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))


def classify(relpath: str | PurePosixPath) -> dict[str, str]:
    """Map a path relative to nepal_data/ onto source, kind, quality curve.

    Section S01.1 rules, with the telegram rule taking precedence: anything
    under chat_export/ is telegram-sourced and gets the telegram quality curve,
    whatever its extension, because Telegram recompressed it.
    """
    p = PurePosixPath(str(relpath).replace("\\", "/"))
    parts = [s.lower() for s in p.parts]
    ext = p.suffix.lower()

    if ext in VIDEO360_EXT:
        kind = "video360"
    elif ext in VIDEO_FLAT_EXT:
        kind = "video_flat"
    elif ext in PHOTO_EXT:
        kind = "photo"
    elif ext in AUDIO_EXT:
        kind = "audio"
    else:
        kind = "document" if ext in DOC_EXT else "other"

    if "chat_export" in parts:
        source, curve = "telegram", "telegram"
        # Telegram round video messages are flat mp4 circles, never 360.
        if kind == "video360":
            kind = "video_flat"
    elif "media_from_camera" in parts:
        source, curve = "camera", "camera"
    elif "media_from_phones" in parts:
        if "keller" in parts:
            source = "phone_keller"
        elif "kulikov" in parts:
            source = "phone_kulikov"
        else:
            source = "phone_unknown"
        curve = "phone"
    elif "music" in parts:
        source, curve = "music", "camera"
    else:
        source, curve = "unknown", "camera"

    return {
        "source": source,
        "kind": kind,
        "quality_curve": curve,
        "container": ext.lstrip(".") or "none",
    }


def telegram_subkind(relpath: str | PurePosixPath) -> str | None:
    """Which chat_export/ subfolder a file came from -- round_video_messages
    are the project's best narration material and must stay identifiable."""
    parts = [s.lower() for s in PurePosixPath(str(relpath)).parts]
    for marker in ("round_video_messages", "video_files", "photos", "files",
                   "voice_messages", "stickers"):
        if marker in parts:
            return marker
    return None


# -- EXIF field access -------------------------------------------------

def exif_get(row: dict[str, Any], *tags: str) -> Any:
    """Fetch a tag regardless of which family-1 group exiftool put it in.

    With -G1 the same logical tag arrives as 'ExifIFD:DateTimeOriginal',
    'QuickTime:CreateDate', 'Track1:GPSLatitude' and so on, and which one you
    get depends on the container. Match on the tag name, prefer an exact
    ungrouped hit, and skip empty values.
    """
    for tag in tags:
        if tag in row and _usable(row[tag]):
            return row[tag]
    lowered = {k.lower(): k for k in row}
    for tag in tags:
        t = tag.lower()
        for lk, orig in lowered.items():
            if lk == t or lk.endswith(":" + t):
                if _usable(row[orig]):
                    return row[orig]
    return None


def _usable(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return s not in ("", "0", "0000:00:00 00:00:00", "-")


DATE_RE = re.compile(
    r"(?P<y>\d{4})[:\-](?P<mo>\d{2})[:\-](?P<d>\d{2})[ T]"
    r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?"
)


def parse_exif_datetime(value: Any, *, assume_tz: timezone | None = None) -> datetime | None:
    """Parse EXIF/QuickTime timestamps into an aware datetime.

    Handles '2023:10:15 14:30:22', ISO forms, fractional seconds and explicit
    offsets including Nepal's +05:45. A value with no zone is tagged with
    ``assume_tz`` if given, else UTC -- and is by definition the uncorrected
    device clock, which is what ``assets.created_at`` is meant to hold.
    """
    if value is None:
        return None
    m = DATE_RE.search(str(value))
    if not m:
        return None
    g = m.groupdict()
    try:
        micro = int(float("0." + g["frac"]) * 1_000_000) if g.get("frac") else 0
        dt = datetime(int(g["y"]), int(g["mo"]), int(g["d"]),
                      int(g["h"]), int(g["mi"]), int(g["s"]), micro)
    except ValueError:
        return None

    tz = g.get("tz")
    if tz:
        if tz == "Z":
            return dt.replace(tzinfo=timezone.utc)
        sign = 1 if tz[0] == "+" else -1
        body = tz[1:].replace(":", "")
        offset = timedelta(hours=int(body[:2]), minutes=int(body[2:4]))
        return dt.replace(tzinfo=timezone(sign * offset))
    return dt.replace(tzinfo=assume_tz or timezone.utc)


OFFSET_RE = re.compile(r"^(?P<sign>[+-])(?P<h>\d{2}):?(?P<m>\d{2})$")


def parse_offset(value: Any) -> timezone | None:
    """Parse an EXIF OffsetTime tag ('+05:45') into a tzinfo."""
    if value is None:
        return None
    m = OFFSET_RE.match(str(value).strip())
    if not m:
        return None
    delta = timedelta(hours=int(m.group("h")), minutes=int(m.group("m")))
    return timezone(delta if m.group("sign") == "+" else -delta)


# Capture-time tags in order of authority.
#
# CreationDate is Apple's com.apple.quicktime.creationdate, which exiftool
# reports as Keys:CreationDate. It is the only capture stamp in a .MOV that
# carries its own UTC offset, and -- crucially -- the only one that survives
# being exported, AirDropped or pulled out of iCloud. QuickTime:CreateDate and
# the per-track MediaCreateDate are rewritten by those operations, so a phone's
# whole library can arrive stamped with the minute it was copied. On this
# corpus that is exactly what happened: 276 clips carried
# QuickTime:CreateDate 2025-11-22 23:24-23:25 -- four-second intervals, the
# signature of a batch export -- while Keys:CreationDate held the true
# 2024-05-11 15:45:12+05:30. Reading CreateDate first put every one of them
# eighteen months after the trek, outside every act, unreachable by the film.
#
# A JPEG or HEIC has no Keys group, so DateTimeOriginal still wins there.
CAPTURE_TAGS = ("CreationDate", "DateTimeOriginal", "CreateDate",
                "MediaCreateDate", "GPSDateTime")

# The camera's heading and lens, where the device recorded them. An iPhone
# still carries GPSImgDirection against true north and a 35 mm-equivalent
# focal length, which together say where the frame was pointing and how wide
# it is -- enough to compute where a named summit falls in the picture
# (spec section 13.3). Requested from exiftool by the same derivation as
# CAPTURE_TAGS: a tag that is not asked for does not exist.
HEADING_TAGS = ("GPSImgDirection", "GPSImgDirectionRef",
                "GPSHPositioningError", "FocalLengthIn35mmFormat")


def parse_heading(row: dict[str, Any]) -> dict[str, Any]:
    """Heading in degrees, its reference ('T' true / 'M' magnetic), the
    horizontal positioning error in metres, and the 35 mm-equivalent focal
    length. None wherever the device wrote nothing."""
    ref = exif_get(row, "GPSImgDirectionRef")
    return {
        "heading_deg": _num(exif_get(row, "GPSImgDirection")),
        "heading_ref": str(ref).strip()[:1].upper() if ref is not None else None,
        "pos_error_m": _num(exif_get(row, "GPSHPositioningError")),
        "focal_35mm": _num(exif_get(row, "FocalLengthIn35mmFormat")),
    }


def asset_datetime(row: dict[str, Any], *,
                   assume_tz: timezone | None = None) -> datetime | None:
    """The device-reported capture time, with its zone resolved.

    EXIF keeps the clock reading and its UTC offset in *separate* tags:
    DateTimeOriginal has no zone, OffsetTimeOriginal carries '+05:45'. Reading
    only the first and defaulting to UTC silently shifts every Nepal photo by
    5 h 45 m -- which would land photos on the wrong day, corrupt the GPS
    interpolation in S02.2 and mis-assign acts. An inline zone on the
    timestamp itself still wins, since it is unambiguous.

    Tags are tried in ``CAPTURE_TAGS`` order; see the note there for why
    Apple's CreationDate outranks QuickTime's CreateDate.
    """
    raw = exif_get(row, *CAPTURE_TAGS)
    if raw is None:
        return None
    if DATE_RE.search(str(raw)) and DATE_RE.search(str(raw)).group("tz"):
        return parse_exif_datetime(raw)
    tz = parse_offset(exif_get(row, "OffsetTimeOriginal", "OffsetTime",
                               "OffsetTimeDigitized"))
    return parse_exif_datetime(raw, assume_tz=tz or assume_tz)


def capture_time_spread(row: dict[str, Any], *,
                        assume_tz: timezone | None = None
                        ) -> tuple[float, str, str] | None:
    """How far the container's capture-time tags disagree, and which two.

    Returns ``(seconds, earliest_tag, latest_tag)`` or None when fewer than two
    tags are present. A file whose own tags disagree by months is a file whose
    timestamp has been rewritten, and the disagreement is the evidence: it is
    worth reporting even once the right tag has been chosen, because it says
    the material needs checking rather than trusting.
    """
    found: list[tuple[str, datetime]] = []
    for tag in CAPTURE_TAGS:
        raw = exif_get(row, tag)
        if raw is None:
            continue
        dt = (parse_exif_datetime(raw) if (DATE_RE.search(str(raw))
              and DATE_RE.search(str(raw)).group("tz"))
              else parse_exif_datetime(raw, assume_tz=assume_tz))
        if dt is not None:
            found.append((tag, dt))
    if len(found) < 2:
        return None
    lo = min(found, key=lambda kv: kv[1])
    hi = max(found, key=lambda kv: kv[1])
    return (hi[1] - lo[1]).total_seconds(), lo[0], hi[0]


# What the frame's shape says about how it was shot. An Insta360 card holds
# three different things under two extensions, and the extension distinguishes
# none of them:
#
#   3840x1920, 1024x512   aspect 2.0   both fisheye circles in one frame
#   2880x2880             aspect 1.0   ONE circular lens; its partner is the
#                                      neighbouring _10_ / _00_ file
#   3840x2160,  640x360   aspect 1.78  flat, single-lens, not 360 at all
#
# On this corpus that is 84 minutes of true 360, 21 minutes of lens pairs and
# 52 minutes of flat 4K. Treating all of it as dual-fisheye reprojects flat
# footage through v360 -- the wrong output, produced the slowest possible way --
# and made the FOV solve measure a stitch seam on frames that have no seam.
DUAL_FISHEYE_ASPECT = (1.85, 2.15)
SINGLE_FISHEYE_ASPECT = (0.9, 1.15)


def frame_shape(width: Any, height: Any) -> str:
    """'dual_fisheye' | 'single_fisheye' | 'flat' | 'unknown' from the frame."""
    try:
        w, h = float(width), float(height)
    except (TypeError, ValueError):
        return "unknown"
    if w <= 0 or h <= 0:
        return "unknown"
    aspect = w / h
    if DUAL_FISHEYE_ASPECT[0] <= aspect <= DUAL_FISHEYE_ASPECT[1]:
        return "dual_fisheye"
    if SINGLE_FISHEYE_ASPECT[0] <= aspect <= SINGLE_FISHEYE_ASPECT[1]:
        return "single_fisheye"
    return "flat"


def refine_kind(kind: str, width: Any, height: Any) -> tuple[str, str | None]:
    """Correct a path-derived kind against the frame, returning (kind, shape).

    ``classify`` is a pure function of the path and stays that way -- it runs
    before anything has been probed. This is the second pass, once the frame
    size is known, and it only ever *demotes*: a .insv that turns out to be
    16:9 is flat footage in a 360 container, but an .mp4 is never promoted to
    360 on the strength of its aspect alone, since 2:1 is a legitimate
    cinematic crop.
    """
    if kind != "video360":
        return kind, None
    shape = frame_shape(width, height)
    if shape == "flat":
        return "video_flat", shape
    if shape == "unknown":
        return kind, None
    return "video360", shape


def parse_gps(row: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """Numeric lat/lon/alt. Requires exiftool's -n flag upstream."""
    lat = _num(exif_get(row, "GPSLatitude"))
    lon = _num(exif_get(row, "GPSLongitude"))
    alt = _num(exif_get(row, "GPSAltitude"))
    if lat is not None and not (-90 <= lat <= 90):
        lat = None
    if lon is not None and not (-180 <= lon <= 180):
        lon = None
    # 0,0 is the null island signature of a stripped or failed fix
    if lat == 0 and lon == 0:
        return None, None, alt
    return lat, lon, alt


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    # tolerate '27.98 N' / '86.92 E' if -n was somehow not applied
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*([NSEW])?$", s)
    if not m:
        return None
    val = float(m.group(1))
    if m.group(2) in ("S", "W"):
        val = -val
    return val


def parse_duration(value: Any) -> float | None:
    """exiftool Duration arrives as seconds, '0:10:23' or '10.5 s'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if ":" in s:
        bits = [float(b) for b in s.split(":")]
        total = 0.0
        for b in bits:
            total = total * 60 + b
        return total
    m = re.match(r"^(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


# Directories that turn up inside a delivered nepal_data/ but are not media.
# The AWS CLI installer alone is 272 MB and 5,875 .rst files, which would
# outnumber the actual footage in the manifest and pollute every count the
# operator reads at the Milestone 1 checkpoint.
DEFAULT_EXCLUDE_DIRS = frozenset({
    "aws", "work", "node_modules", "__pycache__", "venv", ".venv",
    "$RECYCLE.BIN", "System Volume Information", "lost+found",
})

# Where each source lives under data_root. A host that holds only part of the
# corpus -- the GCP box has the phones, the chat and the music but not 60 GB
# of camera originals -- must leave the rows of the sources it cannot see
# alone rather than treat every one of them as a deleted file.
SOURCE_DIRS = {
    "camera": "media_from_camera",
    "phone_keller": "media_from_phones/keller",
    "phone_kulikov": "media_from_phones/kulikov",
    "telegram": "chat_export",
    "music": "music",
}


def absent_sources(root: Path) -> set[str]:
    """Sources whose directory does not exist under ``root`` on this host."""
    return {src for src, sub in SOURCE_DIRS.items() if not (Path(root) / sub).is_dir()}


def walk_media(root: Path, *, skip_hidden: bool = True,
               exclude_dirs: frozenset[str] | set[str] | None = None,
               ignore_globs: Sequence[str] = ()) -> list[Path]:
    """Every regular media-bearing file under root, sorted for deterministic
    asset ordering. Excluded directories are pruned rather than filtered, so a
    large tree of irrelevant files costs nothing to skip. ``ignore_globs``
    match the file name or the relative path: an installer zip, the PDF
    thumbnail Telegram writes next to every PDF, a spreadsheet lock file."""
    import fnmatch
    excluded = {d.lower() for d in
                (DEFAULT_EXCLUDE_DIRS if exclude_dirs is None else exclude_dirs)}
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        if skip_hidden and any(part.startswith(".") for part in parts):
            continue
        if any(part.lower() in excluded for part in parts[:-1]):
            continue
        if any(fnmatch.fnmatch(p.name, g) or fnmatch.fnmatch(rel.as_posix(), g)
               for g in ignore_globs):
            continue
        out.append(p)
    return out
