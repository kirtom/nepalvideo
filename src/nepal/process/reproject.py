"""S03.1 -- dual-fisheye reprojection and the four rectilinear yaw views.

One ffmpeg invocation per recording produces every output the rest of S03 and
S04 need: a 540p equirectangular proxy for shot detection and metrics, four
rectilinear views for face detection and framing choice, and a 16 kHz mono
audio track. One decode, five encodes, no intermediate files -- re-reading a
5.7 K H.265 stream per output is the single largest avoidable cost in the stage.

Two corrections to the command as given in the specification, both of which
make it fail rather than degrade:

  * ``yaw=270`` is rejected: ffmpeg's v360 accepts yaw in [-180, 180]. The
    fourth view must be ``yaw=-90``, which is the same direction.
  * ``split=6`` produces an output labelled ``[aud]`` that is never mapped --
    audio comes from ``-map 0:a``, not from a video split -- and ffmpeg refuses
    a filter with an unconnected output. It must be ``split=5``.

Verified by running the corrected graph: five outputs from one pass.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

DEFAULT_YAWS = (0, 90, 180, 270)


def normalise_yaw(degrees: float) -> float:
    """Map any yaw onto the [-180, 180] range ffmpeg's v360 accepts.

    The pipeline talks about yaw in compass terms (0/90/180/270), which is what
    the shot records and what Gate 2 shows the operator. Only the filter string
    needs the signed form, so the conversion lives here rather than leaking
    into the data model.
    """
    y = float(degrees) % 360.0
    return y - 360.0 if y > 180.0 else y


@dataclass
class ReprojectPlan:
    """Everything needed to run one recording's pass, and nothing else."""
    filter_complex: str
    outputs: list[tuple[str, Path, str]]      # (label, path, bitrate)
    audio_path: Path
    is_360: bool
    source: Path
    yaws: tuple[int, ...] = DEFAULT_YAWS

    @property
    def view_paths(self) -> dict[int, Path]:
        return {yaw: path for (_, path, _), yaw in zip(self.outputs[1:], self.yaws)}

    @property
    def proxy_path(self) -> Path:
        return self.outputs[0][1]


def build_proxy_only_graph(fov_deg: float, *,
                           proxy_size: tuple[int, int] = (1024, 512),
                           work_scale: float = 2.0) -> tuple[str, list[str]]:
    """Phase A: the equirectangular proxy alone, with no split.

    The four yaw views were measured at 79% of this stage's runtime, and nothing
    consumes them as *video* until S07 conforms the ~180 selected shots. Face
    detection and the VLM framing decision both want sampled frames, which
    ``yaw_still_command`` extracts straight from the source -- faster, and
    cleaner pixels than sampling a 1 Mbps re-encode.

    Deliberately emits no ``split``: a split whose outputs are not all consumed
    makes ffmpeg refuse the whole graph.

    The frame is scaled down before v360 rather than after. Reprojection costs
    per pixel and is the whole cost of this pass, so running it on a 3840x1920
    source to produce 1024x512 does roughly four times the necessary work.

    ``min(iw, ...)`` rather than a fixed size, because half this corpus's 360
    material is already a 1024x512 .lrv proxy. Enlarging that to 2048x1024 to
    reproject it and then shrinking it back is four times the work of leaving it
    alone -- a downscale that upscales is worse than no downscale at all.
    """
    pw, ph = proxy_size
    pre_w, pre_h = int(pw * work_scale), int(ph * work_scale)
    return (f"[0:v]scale=w='min(iw,{pre_w})':h='min(ih,{pre_h})',"
            f"v360=input=dfisheye:output=e:ih_fov={fov_deg:g}:iv_fov={fov_deg:g},"
            f"scale={pw}:{ph}[eqout]"), ["eqout"]


def yaw_still_command(source: Path, t_s: float, yaw: float, dest: Path, *,
                      fov_deg: float, view_size: tuple[int, int] = (960, 540),
                      view_h_fov: float = 100.0, view_v_fov: float = 70.0,
                      quality: int = 2) -> list[str]:
    """Phase B: one rectilinear still at one yaw, read from the original.

    Sampling the source rather than a proxy avoids a generation of H.264 loss
    before face embedding and CLIP, which is the difference between a usable
    face cluster and a marginal one on distant subjects.
    """
    vw, vh = view_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{t_s:.3f}", "-i", str(source), "-frames:v", "1",
            "-vf", (f"v360=input=dfisheye:output=rectilinear:"
                    f"ih_fov={fov_deg:g}:iv_fov={fov_deg:g}:yaw={normalise_yaw(yaw):g}:"
                    f"h_fov={view_h_fov:g}:v_fov={view_v_fov:g},scale={vw}:{vh}"),
            "-q:v", str(quality), "-y", str(dest)]


def flat_still_command(source: Path, t_s: float, dest: Path, *,
                       view_size: tuple[int, int] = (960, 540),
                       quality: int = 2) -> list[str]:
    """Phase B for flat footage: no reprojection, just a scaled still."""
    vw, vh = view_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{t_s:.3f}", "-i", str(source), "-frames:v", "1",
            "-vf", f"scale={vw}:{vh}", "-q:v", str(quality), "-y", str(dest)]


def build_360_graph(fov_deg: float, *, proxy_size: tuple[int, int] = (1024, 512),
                    view_size: tuple[int, int] = (960, 540),
                    yaws: Sequence[int] = DEFAULT_YAWS,
                    view_h_fov: float = 100.0,
                    view_v_fov: float = 70.0) -> tuple[str, list[str]]:
    """The filter_complex for a dual-fisheye source, plus its output labels.

    Every branch chains straight from the split to its target: going via an
    intermediate equirectangular file would double the decode and resample the
    image twice.
    """
    n = 1 + len(yaws)
    view_labels = [f"v{i}" for i in range(len(yaws))]
    parts = [f"[0:v]split={n}[eq]" + "".join(f"[{v}]" for v in view_labels)]

    pw, ph = proxy_size
    parts.append(
        f"[eq]v360=input=dfisheye:output=e:ih_fov={fov_deg:g}:iv_fov={fov_deg:g},"
        f"scale={pw}:{ph}[eqout]"
    )

    vw, vh = view_size
    out_labels = ["eqout"]
    for label, yaw in zip(view_labels, yaws):
        signed = normalise_yaw(yaw)
        out = f"y{yaw}"
        parts.append(
            f"[{label}]v360=input=dfisheye:output=rectilinear:"
            f"ih_fov={fov_deg:g}:iv_fov={fov_deg:g}:yaw={signed:g}:"
            f"h_fov={view_h_fov:g}:v_fov={view_v_fov:g},scale={vw}:{vh}[{out}]"
        )
        out_labels.append(out)

    return ";".join(parts), out_labels


def build_lens_pair_graph(fov_deg: float, *,
                          proxy_size: tuple[int, int] = (1024, 512),
                          work_scale: float = 2.0) -> tuple[str, list[str]]:
    """Two circular-lens inputs stacked into the frame v360 expects.

    An Insta360 in dual-stream mode writes one circle per file -- _00_ and _10_
    of the same moment, 2880x2880 each. v360's dfisheye input wants both circles
    side by side in one frame, which is exactly what hstack builds. Feeding it
    one file alone reprojects half a world into a whole one.

    Both inputs must be the same height for hstack, which they are: the camera
    writes the two lenses identically.

    Each lens is scaled down *before* the stack, so v360 reprojects a 2048x1024
    frame rather than the 5760x2880 the card wrote. v360 is the whole cost of
    this pass and its cost is per pixel, so this is roughly eight times less
    work for output that is scaled to 1024x512 either way -- there is no sense
    doing expensive geometry on pixels that are about to be thrown away.
    """
    pw, ph = proxy_size
    lens = max(64, int(pw * work_scale / 2))          # each circle is square
    box = f"scale=w='min(iw,{lens})':h='min(ih,{lens})'"   # shrink only
    return (f"[0:v]{box}[l];[1:v]{box}[r];"
            f"[l][r]hstack=inputs=2[df];"
            f"[df]v360=input=dfisheye:output=e:"
            f"ih_fov={fov_deg:g}:iv_fov={fov_deg:g},scale={pw}:{ph}[eqout]"), ["eqout"]


def build_flat_graph_clamped(*, proxy_size: tuple[int, int] = (960, 540)
                             ) -> tuple[str, list[str]]:
    """Flat video fitted inside the proxy box -- never up, never distorted.

    An .lrv is already a 640x360 proxy. Enlarging it to 960x540 and re-encoding
    costs the full pass to produce something worse than the input.

    ``force_original_aspect_ratio=decrease`` is the whole point of this
    filter and was missing from the first version, which set width and height
    independently. That is correct only for 16:9 input, and 367 of this
    corpus's 458 flat recordings are not: a phone shooting portrait writes
    720x1280, and an iPhone writes 1080x1920 behind a rotation flag that
    ffmpeg applies on decode. Both were being squashed into a 16:9 box --
    stretched 2.4x and 3.2x horizontally. Nothing downstream noticed, because
    every metric still computes happily on a distorted frame; it took
    InsightFace failing to find a face that filled the frame to surface it.
    S07 renders the draft from these proxies, so the film itself would have
    had stretched people in it.
    """
    pw, ph = proxy_size
    return (f"[0:v]scale=w='min(iw,{pw})':h='min(ih,{ph})'"
            f":force_original_aspect_ratio=decrease:force_divisible_by=2[eqout]"), ["eqout"]


def build_flat_graph(*, proxy_size: tuple[int, int] = (960, 540)) -> tuple[str, list[str]]:
    """Flat or wide-mode footage skips v360 entirely -- a single proxy.

    Fits inside the box rather than filling it, for the reason spelled out in
    :func:`build_flat_graph_clamped`: this corpus is mostly not 16:9.
    """
    w, h = proxy_size
    return (f"[0:v]scale=w={w}:h={h}:force_original_aspect_ratio=decrease"
            f":force_divisible_by=2[eqout]"), ["eqout"]


def plan_for_mode(sources: Sequence[Path], recording_id: str, work: Path, *,
                  mode: str, fov_deg: float = 193.0,
                  proxy_size: tuple[int, int] = (1024, 512),
                  view_size: tuple[int, int] = (960, 540),
                  proxy_bitrate: str = "2M",
                  passthrough: bool = False) -> "ReprojectPlan":
    """One recording's pass, chosen by what the frame actually is.

    ``flat`` skips v360 completely. That is not only correct -- reprojecting
    16:9 footage as a fisheye warps it -- but by far the cheapest branch, and on
    this corpus it is 52 of the 130 minutes of camera video.
    """
    proxies = work / "proxies"
    audio = work / "audio"
    for d in (proxies, audio):
        d.mkdir(parents=True, exist_ok=True)
    out = proxies / f"{recording_id}_eq.mp4"

    if mode == "lens_pair":
        fc, _ = build_lens_pair_graph(fov_deg, proxy_size=proxy_size)
    elif mode == "dual_fisheye":
        fc, _ = build_proxy_only_graph(fov_deg, proxy_size=proxy_size)
    elif passthrough:
        # Already no larger than the proxy: remux it rather than re-encode.
        # 55 of this corpus's minutes are .lrv files the camera wrote as
        # proxies; re-encoding them produces something worse, slowly.
        fc = ""
    else:
        fc, _ = build_flat_graph_clamped(proxy_size=view_size)

    return ReprojectPlan(filter_complex=fc,
                         outputs=[("eqout", out, proxy_bitrate)],
                         audio_path=audio / f"{recording_id}.wav",
                         is_360=mode != "flat",
                         source=sources[0], yaws=())


def plan(source: Path, recording_id: str, work: Path, *, is_360: bool,
         fov_deg: float = 193.0, proxy_size: tuple[int, int] = (1024, 512),
         view_size: tuple[int, int] = (960, 540),
         yaws: Sequence[int] = DEFAULT_YAWS,
         view_h_fov: float = 100.0, view_v_fov: float = 70.0,
         proxy_bitrate: str = "2M", view_bitrate: str = "1M",
         yaw_videos: bool = False) -> ReprojectPlan:
    """Plan one recording's pass.

    ``yaw_videos=False`` (the default) is phase A: proxy plus audio only, which
    is 4.3x faster over a whole corpus. Set it True to reproduce the
    specification's single-pass design, which writes all four yaw views as
    video for every recording.
    """
    proxies = work / "proxies"
    views = work / "views"
    audio = work / "audio"
    for d in (proxies, views, audio):
        d.mkdir(parents=True, exist_ok=True)

    if is_360 and not yaw_videos:
        fc, labels = build_proxy_only_graph(fov_deg, proxy_size=proxy_size)
        outputs = [("eqout", proxies / f"{recording_id}_eq.mp4", proxy_bitrate)]
        return ReprojectPlan(filter_complex=fc, outputs=outputs,
                             audio_path=audio / f"{recording_id}.wav",
                             is_360=True, source=source, yaws=())

    if is_360:
        fc, labels = build_360_graph(fov_deg, proxy_size=proxy_size, view_size=view_size,
                                     yaws=yaws, view_h_fov=view_h_fov, view_v_fov=view_v_fov)
        outputs = [("eqout", proxies / f"{recording_id}_eq.mp4", proxy_bitrate)]
        for yaw in yaws:
            outputs.append((f"y{yaw}", views / f"{recording_id}_y{yaw}.mp4", view_bitrate))
    else:
        fc, labels = build_flat_graph(proxy_size=view_size)
        outputs = [("eqout", proxies / f"{recording_id}_eq.mp4", proxy_bitrate)]

    return ReprojectPlan(filter_complex=fc, outputs=outputs,
                         audio_path=audio / f"{recording_id}.wav",
                         is_360=is_360, source=source,
                         yaws=tuple(yaws) if is_360 else ())


def build_command(p: ReprojectPlan, *, hwaccel: str | None = None,
                  encoder: str = "libx264", has_audio: bool = True,
                  sample_rate: int = 16000, extra_input: Sequence[str] = (),
                  inputs: Sequence[Path] | None = None,
                  fps: float | None = None,
                  preset: str | None = None) -> list[str]:
    """The full argv. Kept separate from execution so it can be asserted on.

    ``inputs`` carries more than one source for a lens pair, where the graph
    stacks [0:v] and [1:v]. ``fps`` caps the proxy's frame rate: shot detection
    and metric sampling do not need thirty frames a second, and halving them
    halves the v360 work, which is the whole cost of this pass.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    srcs = list(inputs) if inputs else [p.source]
    for src in srcs:
        cmd += list(extra_input)
        cmd += ["-i", str(src)]
    if p.filter_complex:
        cmd += ["-filter_complex", p.filter_complex]

    for label, path, bitrate in p.outputs:
        if not p.filter_complex:                 # remux: no filter, no re-encode
            cmd += ["-map", "0:v:0", "-c:v", "copy", str(path)]
            continue
        cmd += ["-map", f"[{label}]", "-c:v", encoder, "-b:v", bitrate]
        if preset:
            cmd += ["-preset", preset]
        if fps:
            cmd += ["-r", f"{fps:g}"]
        cmd += [str(path)]

    if has_audio:
        cmd += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
                "-c:a", "pcm_s16le", str(p.audio_path)]
    return cmd


_HWACCEL_CACHE: dict[str, str | None] = {}
_ENCODER_CACHE: dict[str, str] = {}


def _ffmpeg_lists(flag: str) -> str:
    from nepal.util import proc
    if not proc.have("ffmpeg"):
        return ""
    try:
        return proc.run(["ffmpeg", "-hide_banner", flag], check=False).stdout or ""
    except (OSError, proc.ToolMissing):
        return ""


def _works(args: Sequence[str]) -> bool:
    """Run a one-frame probe and report whether it actually succeeded."""
    from nepal.util import proc
    try:
        r = proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *args],
                     check=False, timeout=60)
        return r.returncode == 0
    except (OSError, proc.ToolMissing, Exception):
        return False


def detect_hwaccel(preferred: Sequence[str] = ("cuda", "videotoolbox", "qsv", "vaapi")
                   ) -> str | None:
    """First hardware decoder that actually works, or None.

    Listing is not enough. ffmpeg reports every accelerator and encoder compiled
    into the binary whether or not the hardware is present, so a CPU-only box
    happily advertises ``cuda`` and ``h264_nvenc`` and then fails at runtime
    with "Error initializing a simple filtergraph" -- which would take down
    every S03 job, including a container that lands on a non-GPU instance. So
    each candidate is probed by decoding one synthetic frame through it.

    ``v360`` itself is CPU-only regardless; acceleration helps the H.265 decode
    and the H.264 encode around the reprojection, not the reprojection.
    """
    key = ",".join(preferred)
    if key in _HWACCEL_CACHE:
        return _HWACCEL_CACHE[key]

    listed = {line.strip() for line in _ffmpeg_lists("-hwaccels").splitlines()[1:]
              if line.strip()}
    chosen = None
    for name in preferred:
        if name not in listed:
            continue
        if _works(["-hwaccel", name, "-f", "lavfi",
                   "-i", "testsrc2=size=64x64:rate=1:duration=1",
                   "-frames:v", "1", "-f", "null", "-"]):
            chosen = name
            break
        log.debug("hwaccel %s is compiled in but not functional here", name)
    _HWACCEL_CACHE[key] = chosen
    log.info("hardware decode: %s", chosen or "none (CPU)")
    return chosen


def detect_encoder(preferred: Sequence[str] = ("h264_nvenc", "h264_videotoolbox",
                                               "h264_qsv", "libx264")) -> str:
    """Fastest H.264 encoder that actually works, falling back to libx264."""
    key = ",".join(preferred)
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]

    listed = _ffmpeg_lists("-encoders")
    chosen = "libx264"
    for name in preferred:
        if name not in listed:
            continue
        if name == "libx264":
            chosen = name
            break
        if _works(["-f", "lavfi", "-i", "testsrc2=size=128x128:rate=1:duration=1",
                   "-c:v", name, "-frames:v", "1", "-f", "null", "-"]):
            chosen = name
            break
        log.debug("encoder %s is compiled in but not functional here", name)
    _ENCODER_CACHE[key] = chosen
    log.info("video encoder: %s", chosen)
    return chosen


def reset_detection_cache() -> None:
    _HWACCEL_CACHE.clear()
    _ENCODER_CACHE.clear()


def pick_sources(assets: Sequence[dict], data_root: Path
                 ) -> tuple[list[Path], str] | None:
    """What to decode for one recording, and how it must be reprojected.

    Returns ``(paths, mode)`` where mode is:

      ``dual_fisheye``  one stream with both circles in the frame -- v360 as-is
      ``lens_pair``     two files, one circle each, to be stacked side by side
                        before v360 sees them
      ``flat``          ordinary 16:9 video, which must skip v360 entirely

    An Insta360 card holds all three under .insv and .lrv, so the choice is made
    on the frame shape recorded by S01.1, never the extension. Getting it wrong
    is not a slow path but a wrong one: v360 on flat footage produces a warped
    frame, and on a single lens it produces half a world.

    Within a mode the cheapest adequate source wins. A 1024x512 .lrv is the same
    dual-fisheye layout as its 3840x1920 .insv original and is being scaled to
    540p regardless, so decoding the original buys nothing.
    """
    def resolve(a: dict) -> Path:
        key = str(a["s3_key"])
        rel = key[4:] if key.startswith("raw/") else key
        return data_root / rel

    def shape_of(a: dict) -> str:
        """Fisheye semantics apply only to 360 containers.

        Guessing from dimensions alone calls every square video a lens: a
        Telegram round video message is square by definition, and this demanded
        a partner lens for each of them before skipping the recording entirely.
        A phone clip is flat whatever its aspect.
        """
        recorded = (a.get("frame_shape") or "").strip()
        if recorded:
            return recorded
        if (a.get("kind") or "") != "video360":
            return "flat"
        from nepal.probe.manifest import frame_shape
        shape = frame_shape(a.get("width"), a.get("height"))
        return "flat" if shape == "unknown" else shape

    live = [a for a in assets if resolve(a).exists()]
    by_shape: dict[str, list[dict]] = {}
    for a in live:
        by_shape.setdefault(shape_of(a), []).append(a)

    def ordered(group: Sequence[dict]) -> list[dict]:
        return sorted(group, key=lambda x: (x.get("chapter_index") or 0,
                                            str(x.get("s3_key"))))

    def cheapest(group: Sequence[dict]) -> list[dict]:
        """Proxy container first; within it, chapter order."""
        for container in ("lrv", "insv", "mp4", "mov"):
            same = [a for a in group
                    if (a.get("container") or "").lower() == container]
            if same:
                return ordered(same)
        return ordered(group)

    if by_shape.get("dual_fisheye"):
        return [resolve(a) for a in cheapest(by_shape["dual_fisheye"])], "dual_fisheye"

    if by_shape.get("single_fisheye"):
        # Exactly two lenses of the same moment. More than two means chapters as
        # well, which this corpus does not have and which would need pairing per
        # chapter before stacking -- refuse rather than stack the wrong two.
        pair = ordered(by_shape["single_fisheye"])
        if len(pair) == 2:
            return [resolve(a) for a in pair], "lens_pair"
        log.warning("recording has %d single-lens files, expected 2 -- skipping",
                    len(pair))
        return None

    group = by_shape.get("flat") or by_shape.get("unknown")
    if group:
        return [resolve(a) for a in cheapest(group)], "flat"
    return None


def concat_list(paths: Sequence[Path], dest: Path) -> Path:
    """An ffconcat list for the concat demuxer.

    Paths are quoted with the demuxer's own escaping -- a single quote inside a
    filename is written as '\'' -- because a path with an apostrophe in it would
    otherwise silently truncate the list.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ffconcat version 1.0"]
    for p in paths:
        escaped = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    dest.write_text("\n".join(lines) + "\n")
    return dest


