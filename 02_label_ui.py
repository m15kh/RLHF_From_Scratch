#!/usr/bin/env python3
"""
Step 2 of the RLHF pipeline: web UI for labeling pairs of episode segments.

    ./02_label_ui.py -d db-v0                    # http://localhost:8080
    ./02_label_ui.py -d db-v0 --host 0.0.0.0     # reachable from other machines (no auth!)

Keys: 1 - first is better, 2 - second is better, 0 - both are equally good/bad,
      arrow up/down - move through the list.
"""
import argparse
import functools
import io
import pathlib
import random
import re
import typing as tt

from fastapi import HTTPException, Response
from nicegui import app, ui
from PIL import Image

from lib import rlhf

PAIRS_PER_LIST = 20
# Recording is 60 FPS, but browsers clamp GIF frame delays below 20ms to 100ms,
# so we play segments a bit slower than real time.
GIF_FRAME_MS = 40
GIF_SCALE = 2

LABEL_TEXT = {0: "both good", 1: "#1 better", 2: "#2 better"}
SEGMENT_RE = re.compile(r"^\d+/\d+" + re.escape(rlhf.SEGMENT_SUFFIX) + "$")

DB_PATH: pathlib.Path = pathlib.Path(".")


# ---------------------------------------------------------------------------------------
# Segment -> animated GIF, served by a plain HTTP endpoint
# ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def render_gif(rel_path: str) -> bytes:
    obs, _ = rlhf.load_segment(DB_PATH / rel_path)
    h, w = obs.shape[1:3]
    frames = [Image.fromarray(o).resize((w * GIF_SCALE, h * GIF_SCALE), Image.Resampling.NEAREST) for o in obs]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=GIF_FRAME_MS, loop=0)
    return buf.getvalue()


@app.get("/segment/{env_dir}/{name}")
def segment_gif(env_dir: str, name: str):
    rel_path = f"{env_dir}/{name}"
    if not SEGMENT_RE.match(rel_path) or not (DB_PATH / rel_path).is_file():
        raise HTTPException(status_code=404)
    return Response(content=render_gif(rel_path), media_type="image/gif",
                    headers={"Cache-Control": "max-age=3600"})


def gif_url(rel_path: str) -> str:
    return f"/segment/{rel_path}"


# ---------------------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------------------

def make_row(idx: int, sample1: str, sample2: str, label: int | None) -> dict:
    return {
        "id": idx,
        "sample1": sample1,
        "sample2": sample2,
        "label": label,
        "label_text": "" if label is None else LABEL_TEXT[label],
    }


def sample_pairs(count: int) -> list[dict]:
    """Random pairs of segments that weren't labeled yet."""
    segments = rlhf.list_segments(DB_PATH)
    if len(segments) < 2:
        return []
    known = set()
    for lbl in rlhf.load_labels(DB_PATH):
        known.add((lbl.sample1, lbl.sample2))
        known.add((lbl.sample2, lbl.sample1))
    rows = []
    for _ in range(count * 50):
        if len(rows) >= count:
            break
        s1, s2 = random.sample(segments, 2)
        if (s1, s2) in known:
            continue
        known.add((s1, s2))
        known.add((s2, s1))
        rows.append(make_row(len(rows), s1, s2, None))
    return rows


def existing_rows() -> list[dict]:
    return [make_row(i, lbl.sample1, lbl.sample2, lbl.label) for i, lbl in enumerate(rlhf.load_labels(DB_PATH))]


# ---------------------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------------------

def page_layout():
    with ui.header().classes("items-center"):
        ui.button(icon="menu", on_click=lambda: drawer.toggle()).props("flat color=white")
        ui.label("RLHF label UI").classes("text-xl")
    with ui.left_drawer(value=True, bordered=True).props("width=180").classes("bg-blue-50") as drawer:
        ui.link("Overview", "/")
        ui.link("Label new data", "/label")
        ui.link("Existing labels", "/labels")


def labeling_view(rows: list[dict], resample: tt.Callable[[], list[dict]] | None):
    """
    Shared by 'Label new data' (resample is given, jump to next unlabeled pair after labeling)
    and 'Existing labels' (no resample, jump to the next row after labeling).
    """
    columns = [
        {"name": "sample1", "label": "Sample 1", "field": "sample1", "align": "left"},
        {"name": "sample2", "label": "Sample 2", "field": "sample2", "align": "left"},
        {"name": "label", "label": "Label", "field": "label_text", "align": "left"},
    ]

    with ui.row().classes("w-full no-wrap items-start gap-6"):
        with ui.column().classes("w-[400px] shrink-0"):
            if resample is not None:
                ui.button("Resample list", on_click=lambda: do_resample()).classes("w-full")
            table = ui.table(rows=rows, columns=columns, row_key="id", selection="single") \
                .classes("w-full").props("dense bordered virtual-scroll").style("max-height: 78vh")
            status = ui.label().classes("text-sm text-gray-500")

        with ui.column().classes("gap-4"):
            with ui.row().classes("no-wrap gap-4"):
                with ui.column().classes("items-center gap-1"):
                    ui.label("Sample 1").classes("text-lg")
                    img1 = ui.image().classes("w-[272px] h-[357px] bg-gray-200").props("no-transition no-spinner")
                    name1 = ui.label().classes("text-xs text-gray-500")
                with ui.column().classes("items-center gap-1"):
                    ui.label("Sample 2").classes("text-lg")
                    img2 = ui.image().classes("w-[272px] h-[357px] bg-gray-200").props("no-transition no-spinner")
                    name2 = ui.label().classes("text-xs text-gray-500")
            with ui.row().classes("gap-4"):
                ui.button("#1 is better (1)", on_click=lambda: assign(1))
                ui.button("Both are good (0)", on_click=lambda: assign(0))
                ui.button("#2 is better (2)", on_click=lambda: assign(2))
            ui.label("Use buttons above or keys '1', '2' or '0' to set label and go to the next sample. "
                     "Arrow keys move through the list.").classes("text-sm")

    def find_row(row_id: int) -> dict | None:
        return next((r for r in table.rows if r["id"] == row_id), None)

    def update_status():
        done = sum(1 for r in table.rows if r["label"] is not None)
        status.set_text(f"{done} of {len(table.rows)} pairs labeled")

    def show(row: dict | None):
        if row is None:
            img1.set_source("")
            img2.set_source("")
            name1.set_text("")
            name2.set_text("")
            return
        img1.set_source(gif_url(row["sample1"]))
        img2.set_source(gif_url(row["sample2"]))
        name1.set_text(row["sample1"])
        name2.set_text(row["sample2"])

    def select(row: dict | None):
        table.selected = [row] if row is not None else []
        table.update()
        if row is not None:
            # keep the selected row visible in long lists
            table.run_method("scrollTo", table.rows.index(row), "center")
        show(row)

    def current() -> dict | None:
        return find_row(table.selected[0]["id"]) if table.selected else None

    def move(delta: int):
        rows_ = table.rows
        if not rows_:
            return
        cur = current()
        pos = rows_.index(cur) if cur is not None else -delta
        select(rows_[max(0, min(len(rows_) - 1, pos + delta))])

    def advance(after: dict):
        rows_ = table.rows
        pos = rows_.index(after)
        if resample is None:
            # existing labels: just go to the next row
            if pos + 1 < len(rows_):
                select(rows_[pos + 1])
            return
        # new labels: next unlabeled row (wrapping around)
        for r in rows_[pos + 1:] + rows_[:pos]:
            if r["label"] is None:
                select(r)
                return
        ui.notify("All pairs in the list are labeled, press 'Resample list' to get new ones")

    def assign(label: int):
        row = current()
        if row is None:
            ui.notify("Select a pair first")
            return
        rlhf.set_label(DB_PATH, row["sample1"], row["sample2"], label)
        row["label"] = label
        row["label_text"] = LABEL_TEXT[label]
        table.update()
        update_status()
        advance(row)

    def do_resample():
        new_rows = resample()
        table.update_rows(new_rows)
        table.update()
        update_status()
        select(table.rows[0] if table.rows else None)
        if not new_rows:
            ui.notify("Not enough unlabeled segment pairs in the DB")

    def on_key(e):
        if not e.action.keydown:
            return
        if e.action.repeat and not e.key.is_cursorkey:
            return
        if e.key == "1":
            assign(1)
        elif e.key == "2":
            assign(2)
        elif e.key == "0":
            assign(0)
        elif e.key == "ArrowDown":
            move(1)
        elif e.key == "ArrowUp":
            move(-1)

    table.on_select(lambda e: show(find_row(e.selection[0]["id"]) if e.selection else None))
    table.on("rowClick", lambda e: select(find_row(e.args)), js_handler="(evt, row) => emit(row.id)")
    # ignore=[] - keep the hotkeys working right after a button was clicked (button keeps focus)
    ui.keyboard(on_key=on_key, ignore=[])

    update_status()
    if table.rows:
        select(table.rows[0])
    elif resample is not None:
        ui.notify("Not enough unlabeled segment pairs in the DB")


@ui.page("/")
def overview_page():
    page_layout()
    labels = rlhf.load_labels(DB_PATH)
    counts = {k: sum(1 for lbl in labels if lbl.label == k) for k in LABEL_TEXT}
    with ui.column().classes("gap-2"):
        ui.label(f"DB path: {DB_PATH}")
        ui.label(f"Trajectories: {len(rlhf.list_segments(DB_PATH))}")
        ui.label(f"Human labels: {len(labels)}")
        ui.label(f"  #1 better: {counts[1]}, #2 better: {counts[2]}, both good: {counts[0]}").classes("text-sm text-gray-500")


@ui.page("/label")
def label_page():
    page_layout()
    labeling_view(sample_pairs(PAIRS_PER_LIST), resample=lambda: sample_pairs(PAIRS_PER_LIST))


@ui.page("/labels")
def labels_page():
    page_layout()
    rows = existing_rows()
    if not rows:
        ui.label("No labels yet")
        return
    labeling_view(rows, resample=None)


def main():
    global DB_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db", required=True, help="Path to the DB with episode segments")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to listen on, default=127.0.0.1")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on, default=8080")
    args = parser.parse_args()
    DB_PATH = pathlib.Path(args.db)
    if not DB_PATH.is_dir():
        parser.error(f"DB directory {DB_PATH} doesn't exist")
    ui.run(host=args.host, port=args.port, title="RLHF label UI", reload=False, show=False)


if __name__ == "__main__":
    main()
