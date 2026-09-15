"""S04 -- the semantic layer.

S04.1 embeds one representative frame per surviving shot. S04.2 and S04.3 are
Bedrock and are not built yet; on the account this was developed against,
Bedrock was gated behind a support case, and S04.1 needs none of it -- which
is why the stage is split here rather than waiting.

The embedding artefact is deliberately a plain ``.npy`` plus a JSON index
rather than rows in the database: the spec asks for it to outlive the film as
a searchable index of a personal video library.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from nepal import db, freshness
from nepal.config import Config
from nepal.process import embed as embed_mod, metrics as metrics_mod
from nepal.util.progress import Progress

log = logging.getLogger(__name__)
STAGE = "S04"


def surviving_shots(conn) -> list[dict[str, Any]]:
    """Every shot the gate did not reject, with what it takes to find a frame.

    The predicate is the complement of the gate's verdict rather than a list of
    the statuses that happen to exist today. S05 promotes its picks to
    'shortlisted', so a status whitelist of 'candidate' silently stops matching
    the shots the film is made of the moment a cut has been built once.
    """
    return [dict(r) for r in conn.execute(
        "SELECT s.shot_id, s.recording_id, s.asset_id, s.media_kind, "
        "s.start_s, s.end_s, a.s3_key, r.is_360 "
        "FROM shots s "
        "LEFT JOIN assets a ON a.asset_id = s.asset_id "
        "LEFT JOIN recordings r ON r.recording_id = s.recording_id "
        "WHERE s.status <> 'rejected' ORDER BY s.shot_id")]


def embed_shots(cfg: Config, conn, *, force: bool = False) -> dict[str, Any]:
    """S04.1 -- CLIP embedding of the sharpest frame of every surviving shot.

    Photographs are embedded from their own file; video shots from the proxy.
    Only shots that survived the gate: the point of the gate is that nothing
    past it should pay for what it rejected.

    "Survived" is ``status <> 'rejected'``, not ``status = 'candidate'``. S05
    promotes its picks to 'shortlisted', so once a cut exists the 400 shots the
    film is actually made of no longer match 'candidate' -- and those are
    precisely the rows MMR needs an embedding for. Asking for the complement of
    the gate's verdict says what is meant and cannot drift as statuses are
    added.
    """
    out_dir = cfg.workdir("semantic")
    emb_path = out_dir / "clip.npy"
    idx_path = out_dir / "clip_index.json"
    done: set[str] = set()
    if emb_path.exists() and idx_path.exists() and not force:
        done = set(json.loads(idx_path.read_text())["shot_ids"])

    rows = surviving_shots(conn)
    todo = [r for r in rows if r["shot_id"] not in done]
    if not todo:
        return {"n_shots": len(rows), "note": "every surviving shot already embedded"}

    model_name = str(cfg.get("semantic.clip_model", embed_mod.CLIP_MODEL))
    pretrained = str(cfg.get("semantic.clip_pretrained", embed_mod.CLIP_PRETRAINED))
    gpu = bool(cfg.get("semantic.clip_gpu", False))
    batch_n = int(cfg.get("semantic.clip_batch", 16))
    n_samples = int(cfg.get("process.metric_samples_per_shot", 5))
    log.info("S04.1 embedding %d shot(s) with %s/%s on %s",
             len(todo), model_name, pretrained, "GPU" if gpu else "CPU")
    try:
        model, preprocess, device = embed_mod.load_model(
            model_name, pretrained, gpu=gpu,
            threads=int(cfg.get("semantic.clip_threads", 0)))
    except ImportError:
        log.warning("S04.1 open_clip not installed -- skipping "
                    "(pip install '.[semantic]')")
        return {"skipped": "open_clip not installed"}

    proxies = cfg.work_root / "proxies"
    every = int(cfg.get("semantic.clip_checkpoint_every", 64))
    prior = np.load(emb_path) if (done and emb_path.exists()) else None
    prior_ids = list(json.loads(idx_path.read_text())["shot_ids"]) if done else []
    vecs: list[np.ndarray] = []
    ids: list[str] = []
    pending_imgs: list[np.ndarray] = []
    pending_ids: list[str] = []
    failed: dict[str, int] = {}
    saved = [0]
    bar = Progress("S04.1 embedding shots", len(todo))

    def merged() -> tuple[np.ndarray, list[str]]:
        if prior is not None:
            return (np.vstack([prior] + vecs) if vecs else prior), prior_ids + ids
        return (np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)), list(ids)

    def checkpoint() -> np.ndarray:
        """Write what has been embedded so far, atomically.

        ViT-L-14 measures 13 s/shot on CPU, so a full corpus is ~6 hours. The
        stage used to accumulate everything in memory and write once at the
        end, which means a kill at hour five -- or the OOM killer on a box
        already deep in swap -- threw away five hours and left no way to
        resume. ``done`` is read back from the index on the next run, so a
        checkpoint is also the resume point.

        Written to a temporary name and renamed, because the failure this
        guards against is the process dying, and dying midway through
        ``np.save`` would leave a truncated array that reads as corrupt.
        """
        matrix, all_ids = merged()
        tmp_e = emb_path.with_name(emb_path.name + ".tmp")
        tmp_i = idx_path.with_name(idx_path.name + ".tmp")
        with open(tmp_e, "wb") as fh:          # not np.save(path): it appends .npy
            np.save(fh, matrix)
        tmp_i.write_text(json.dumps({"shot_ids": all_ids, "model": model_name,
                                     "pretrained": pretrained}))
        tmp_e.replace(emb_path)
        tmp_i.replace(idx_path)
        return matrix

    def flush() -> None:
        if not pending_imgs:
            return
        vecs.append(embed_mod.encode(model, preprocess, device, pending_imgs))
        ids.extend(pending_ids)
        pending_imgs.clear()
        pending_ids.clear()
        if len(ids) - saved[0] >= every:
            checkpoint()
            saved[0] = len(ids)
            log.info("S04.1 checkpoint: %d of %d embedded", len(ids), len(todo))

    import cv2
    for r in todo:
        bar.step(note=r["shot_id"][-24:])
        try:
            if r["media_kind"] == "photo":
                path = cfg.data_root / str(r["s3_key"]).replace("raw/", "")
                frame = cv2.imread(str(path))
                frames = [frame] if frame is not None else []
            else:
                px = proxies / f"{r['recording_id']}_eq.mp4"
                if not px.exists():
                    failed["no proxy"] = failed.get("no proxy", 0) + 1
                    continue
                times = metrics_mod.sample_times(r["start_s"], r["end_s"], n_samples)
                groups = metrics_mod.read_samples(px, times, frames_per_sample=1)
                frames = [g[0] for g in groups if g]
            pick = embed_mod.sharpest(frames)
            if pick < 0:
                failed["no readable frame"] = failed.get("no readable frame", 0) + 1
                continue
            pending_imgs.append(frames[pick])
            pending_ids.append(r["shot_id"])
            if len(pending_imgs) >= batch_n:
                flush()
        except Exception as exc:                       # one shot must not end the stage
            log.debug("S04.1 %s: %s", r["shot_id"], exc)
            failed[type(exc).__name__] = failed.get(type(exc).__name__, 0) + 1
    flush()
    bar.close(f"{len(ids)} embedded")

    matrix = checkpoint()
    log.info("S04.1 %d embedding(s) of dim %d in %s", matrix.shape[0],
             matrix.shape[1] if matrix.ndim > 1 else 0, emb_path)
    if failed:
        log.warning("S04.1 %d shot(s) produced nothing: %s", sum(failed.values()), failed)
    return {"n_shots": len(rows), "n_embedded": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]) if matrix.ndim > 1 else 0, "failed": failed}


def run(cfg: Config, *, force: bool = False,
        redo: set[str] | None = None) -> dict[str, Any]:
    conn = db.init(cfg.db_path)
    report: dict[str, Any] = {"stage": STAGE, "started_utc": db.utcnow()}
    report["skipped_stale"] = freshness.warn_if_stale(
        log, conn, STAGE, force=force, rerun_hint="nepal s04 --force")
    done = db.done_units(conn, STAGE)
    steps = [("embeddings", lambda: embed_shots(
        cfg, conn, force=force or "embeddings" in (redo or ())))]
    always = set(redo or ())
    for name, fn in steps:
        if not force and name in done and name not in always:
            report[name] = {"skipped": "already done"}
            continue
        report[name] = fn()
        db.mark_unit(conn, STAGE, name, detail=json.dumps(report[name], default=str)[:2000])
    report["finished_utc"] = db.utcnow()
    cfg.work("reports", "s04_semantic.json").write_text(
        json.dumps(report, indent=2, default=str))
    conn.close()
    return report
