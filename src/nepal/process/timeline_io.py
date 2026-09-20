"""S06 output -- OpenTimelineIO and FCPXML.

Written by hand rather than through the ``opentimelineio`` package. The schema
this needs is a handful of nested dicts, and the alternative is a dependency
with a compiled core for one serialisation -- the same trade the project
already made for SRTM tiles against GDAL.

FCPXML is the one the operator actually opens: Resolve and Final Cut both
import it, which is what makes the draft cut refinable by hand instead of only
re-runnable. Its frame-rate model is the trap -- every time is a rational
``N/Dz`` in the *timeline's* timebase, and a value that does not land on a
frame boundary is silently rounded by the importer, so times are quantised
here where it can be reasoned about.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

FPS = 25


def _frames(seconds: float, fps: int = FPS) -> int:
    return int(round(float(seconds) * fps))


def fcp_time(seconds: float, fps: int = FPS) -> str:
    """``N/Ds`` on a frame boundary, which is the only form FCPXML respects."""
    return f"{_frames(seconds, fps) * (1000 // fps if 1000 % fps == 0 else 1)}/" \
           f"{1000 if 1000 % fps == 0 else fps}s"


AUDIO_TRACKS = ("speech", "location", "music")


def _range(start_s: float, length_s: float, fps: int) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "TimeRange.1",
            "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": _frames(start_s, fps)},
            "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": _frames(length_s, fps)}}


def _items(rows: Sequence[Mapping[str, Any]], *, fps: int, schema: str, name_of, meta_keys) -> list[dict[str, Any]]:
    """The rows laid on one track in film time, a gap before any that does
    not start where the last one ended. An OTIO track is sequential -- an
    item sits where the one before it stopped -- so a cue that starts late
    is placed by the gap in front of it, not by a time of its own."""
    items: list[dict[str, Any]] = []
    prev_end = 0.0
    for r in rows:
        t_in, t_out = float(r["t_in"]), float(r["t_out"])
        if t_in > prev_end + 1e-6:
            items.append({"OTIO_SCHEMA": "Gap.1", "name": "gap", "source_range": _range(0.0, t_in - prev_end, fps)})
        items.append({"OTIO_SCHEMA": schema, "name": name_of(r),
                      "source_range": _range(r.get("src_in") or 0.0, t_out - t_in, fps),
                      "metadata": {"nepal": {k: r[k] for k in meta_keys if k in r}}})
        prev_end = t_out
    return items


def to_otio(rows: Sequence[Mapping[str, Any]], *, name: str = "nepal",
            fps: int = FPS, cues: Sequence[Mapping[str, Any]] = (),
            overlays: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """An OTIO timeline of one video track, gaps included where they exist,
    and -- when the cut has sound -- one audio track per cue track and an
    ``overlays`` track. An overlay is no media, so it is a gap carrying its
    payload: the editor sees where a card sits without a clip to resolve."""
    tracks = [{"OTIO_SCHEMA": "Track.1", "name": "V1", "kind": "Video",
               "children": _items(rows, fps=fps, schema="Clip.1",
                                  name_of=lambda r: str(r.get("shot_id") or r.get("kind")),
                                  meta_keys=("act", "yaw", "shot_id"))}]
    if cues:
        for track in AUDIO_TRACKS:
            mine = sorted((c for c in cues if c.get("track") == track), key=lambda c: float(c["t_in"]))
            tracks.append({"OTIO_SCHEMA": "Track.1", "name": track, "kind": "Audio",
                           "children": _items(mine, fps=fps, schema="Clip.1", name_of=lambda c: str(c["cue_id"]),
                                              meta_keys=("source", "gain_lufs", "beat_id"))})
    if overlays:
        ordered = sorted(overlays, key=lambda o: float(o["t_in"]))
        tracks.append({"OTIO_SCHEMA": "Track.1", "name": "overlays", "kind": "Video",
                       "children": _items(ordered, fps=fps, schema="Gap.1", name_of=lambda o: str(o["overlay_id"]),
                                          meta_keys=("kind", "payload", "asset_path"))})
    return {
        "OTIO_SCHEMA": "Timeline.1", "name": name,
        "global_start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": 0},
        "tracks": {"OTIO_SCHEMA": "Stack.1", "name": "tracks", "children": tracks},
    }


def to_fcpxml(rows: Sequence[Mapping[str, Any]], *, media_dir: str,
              name: str = "nepal", fps: int = FPS) -> str:
    """FCPXML 1.9 with one asset per distinct source clip."""
    fcpxml = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(resources, "format", id="r0", name=f"FFVideoFormat{fps}p",
                  frameDuration=fcp_time(1.0 / fps, fps), width="1920", height="1080")

    assets: dict[str, str] = {}
    for r in rows:
        key = str(r.get("source") or r.get("shot_id") or r.get("kind"))
        if key in assets:
            continue
        aid = f"a{len(assets) + 1}"
        assets[key] = aid
        asset = ET.SubElement(resources, "asset", id=aid, name=key,
                              start="0s", hasVideo="1", format="r0", hasAudio="1")
        ET.SubElement(asset, "media-rep", kind="original-media",
                      src=f"file://{media_dir}/{key}")

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name=name)
    project = ET.SubElement(event, "project", name=name)
    total = max((float(r["t_out"]) for r in rows), default=0.0)
    sequence = ET.SubElement(project, "sequence", format="r0",
                             duration=fcp_time(total, fps),
                             tcStart="0s", tcFormat="NDF")
    spine = ET.SubElement(sequence, "spine")
    for r in rows:
        # A v2 card slot has no shot: it is named and keyed by its kind.
        key = str(r.get("source") or r.get("shot_id") or r.get("kind"))
        t_in, t_out = float(r["t_in"]), float(r["t_out"])
        clip = ET.SubElement(spine, "asset-clip", name=str(r.get("shot_id") or r.get("kind")),
                             ref=assets[key], offset=fcp_time(t_in, fps),
                             start=fcp_time(r.get("src_in") or 0.0, fps),
                             duration=fcp_time(t_out - t_in, fps))
        if r.get("act") is not None:
            ET.SubElement(clip, "note").text = f"act {r['act']}"
    return ET.tostring(fcpxml, encoding="unicode", xml_declaration=True)


def write(rows: Sequence[Mapping[str, Any]], out_dir: Path, *,
          media_dir: str, fps: int = FPS, cues: Sequence[Mapping[str, Any]] = (),
          overlays: Sequence[Mapping[str, Any]] = ()) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    otio_path = out_dir / "timeline.otio"
    fcp_path = out_dir / "timeline.fcpxml"
    otio_path.write_text(json.dumps(to_otio(rows, fps=fps, cues=cues, overlays=overlays), indent=1))
    fcp_path.write_text(to_fcpxml(rows, media_dir=media_dir, fps=fps))
    return {"otio": otio_path, "fcpxml": fcp_path}
