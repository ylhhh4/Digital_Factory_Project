#!/usr/bin/env python3
"""
vebjs_yolo.py - convert VEB.js Data Generation exports to/from Ultralytics YOLO.

VEB.js export (Data Generation panel -> Export JSON) is a JSON list, one entry per image:

    {
      "width": 1378, "height": 814,                      # image size in pixels
      "image": "data:image/jpeg;base64,...",             # the image itself, embedded
      "annotations": [
        {
          "id": "P07",                                   # asset id in the scene
          "class": "P07",                                # class name
          "bbox": [xmin, ymin, xmax, ymax, cx, cy, w, h],# all normalised 0-1, origin top-left
          "segmentation": [[x1, y1, x2, y2, ...], ...],  # one or more open polygons, normalised
          "area": 241295,                                # segmented area in pixels
          "is_crowd": 0
        }
      ]
    }

Commands
    veb2yolo   one or more VEB.js JSON files -> YOLO dataset folder (detect, segment or both)
    yolo2veb   YOLO dataset folder -> one VEB.js JSON file
    show       draw labels of a VEB.js JSON or a YOLO folder on sample images (visual check)

Examples
    python vebjs_yolo.py veb2yolo runs/*.json -o dataset --task both      # labels/ + labels_seg/
    python vebjs_yolo.py veb2yolo M3_seed17.json -o dataset --rename P07=tslot_plate
    python vebjs_yolo.py yolo2veb dataset -o back.json
    python vebjs_yolo.py show dataset -n 12 -o overlays

Requires: Python 3.8+, Pillow (pip install pillow). PyYAML optional.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import re
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: pip install pillow")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- utils
def clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def fmt(v: float) -> str:
    return f"{clamp01(v):.6f}"


def polygon_area_px(poly: list[float], w: int, h: int) -> float:
    xs, ys = [x * w for x in poly[0::2]], [y * h for y in poly[1::2]]
    n = len(xs)
    return abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(n))) / 2.0


def bbox8_from_points(xs: list[float], ys: list[float]) -> list[float]:
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1, y1, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def decode_data_uri(uri: str) -> tuple[bytes, str]:
    m = re.match(r"data:image/(\w+);base64,(.*)", uri, re.S)
    if not m:
        raise ValueError("image field is not a base64 data URI")
    ext = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp"}.get(m.group(1).lower(), "." + m.group(1).lower())
    return base64.b64decode(m.group(2)), ext


def encode_data_uri(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def merge_multi_segment(polys: list[list[float]]) -> list[float]:
    """Join several polygons into one outline by zero-width 'bridges' between nearest points
    (same idea as Ultralytics' COCO converter), because a YOLO-seg line holds one polygon."""
    segs = [[(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)] for p in polys if len(p) >= 6]
    if not segs:
        return []
    if len(segs) == 1:
        return [c for pt in segs[0] for c in pt]

    def nearest(a, b):
        best = (math.inf, 0, 0)
        for i, pa in enumerate(a):
            for j, pb in enumerate(b):
                d = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                if d < best[0]:
                    best = (d, i, j)
        return best[1], best[2]

    links = [(None, None)] * len(segs)
    for k in range(1, len(segs)):
        links[k] = nearest(segs[k - 1], segs[k])  # (index in k-1, index in k)
    # rotate each polygon so it starts at the point linked to its predecessor
    out: list[tuple[float, float]] = []
    # forward pass
    for k, s in enumerate(segs):
        if k == 0:
            start = 0
        else:
            start = links[k][1]
        rot = s[start:] + s[:start] + [s[start]]
        if k < len(segs) - 1:
            # walk until the point that links to the next polygon, leave the rest for the way back
            nxt = links[k + 1][0]
            pos = (nxt - start) % len(s)
            out.extend(rot[: pos + 1])
        else:
            out.extend(rot)
    # backward pass: finish each earlier polygon
    for k in range(len(segs) - 2, -1, -1):
        s = segs[k]
        start = 0 if k == 0 else links[k][1]
        rot = s[start:] + s[:start] + [s[start]]
        pos = (links[k + 1][0] - start) % len(s)
        out.extend(rot[pos:])
    return [c for pt in out for c in pt]


def load_names(dataset: Path) -> list[str]:
    y = dataset / "data.yaml"
    if not y.exists():
        return []
    text = y.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        names = yaml.safe_load(text).get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return [str(n) for n in names]
    except Exception:
        names: dict[int, str] = {}
        in_names = False
        for line in text.splitlines():
            if line.strip().startswith("names:"):
                rest = line.split(":", 1)[1].strip()
                if rest.startswith("["):
                    return [s.strip().strip("'\"") for s in rest.strip("[]").split(",") if s.strip()]
                in_names = True
                continue
            if in_names:
                m = re.match(r"\s+(\d+)\s*:\s*(.+)", line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip().strip("'\"")
                elif line.strip() and not line.startswith(" "):
                    break
        return [names[k] for k in sorted(names)]


def write_data_yaml(dataset: Path, names: list[str]) -> None:
    lines = [
        "# Generated by vebjs_yolo.py",
        f"path: {dataset.resolve().as_posix()}",
        "train: images  # whole dataset; split later",
        "val: images",
        f"nc: {len(names)}",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (dataset / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- veb -> yolo
def veb2yolo(args) -> None:
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if args.task == "both" and args.multipart == "split":
        sys.exit("--task both needs --multipart merge or largest (one polygon line per box line)")

    def label_sets(det, seg):
        if args.task == "detect":
            return [("labels", det)]
        if args.task == "segment":
            return [("labels", seg)]
        return [("labels", det), ("labels_seg", seg)]

    for folder, _ in label_sets([], []):
        (out / folder).mkdir(parents=True, exist_ok=True)
    rename = dict(r.split("=", 1) for r in (args.rename or []))
    names = load_names(out)  # keep ids stable when appending runs
    if args.classes:
        for n in Path(args.classes).read_text(encoding="utf-8").split():
            if n not in names:
                names.append(n)

    manifest = out / "manifest.csv"
    new_manifest = not manifest.exists()
    mf = manifest.open("a", encoding="utf-8")
    if new_manifest:
        mf.write("image,source_json,source_index,width,height,n_instances,classes,ids\n")

    n_img = n_inst = n_skipped = n_multi = 0
    for src in args.inputs:
        src = Path(src)
        data = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        prefix = args.prefix or re.sub(r"[^A-Za-z0-9_.-]+", "_", src.stem)
        for idx, entry in enumerate(data):
            anns = entry.get("annotations", [])
            if not anns and not args.keep_empty:
                n_skipped += 1
                continue
            raw, ext = decode_data_uri(entry["image"])
            stem = f"{prefix}_{idx:04d}"
            (out / "images" / f"{stem}{ext}").write_bytes(raw)
            w, h = entry.get("width"), entry.get("height")
            if not w or not h:
                w, h = Image.open(io.BytesIO(raw)).size
            det_lines, seg_lines, classes, ids = [], [], [], []
            for a in anns:
                cname = rename.get(a["class"], a["class"])
                if cname not in names:
                    if args.classes:
                        sys.exit(f"class '{cname}' ({src.name} #{idx}) not in {args.classes}")
                    names.append(cname)
                cid = names.index(cname)
                classes.append(cname)
                ids.append(str(a.get("id", "")))
                b = a["bbox"]
                cx, cy, bw, bh = (b[4], b[5], b[6], b[7]) if len(b) >= 8 else (
                    (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])
                det_lines.append(f"{cid} {fmt(cx)} {fmt(cy)} {fmt(bw)} {fmt(bh)}")
                polys = [p for p in a.get("segmentation", []) if len(p) >= 6]
                if not polys:
                    polys = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]]]
                if len(polys) > 1:
                    n_multi += 1
                    if args.multipart == "largest":
                        polys = [max(polys, key=lambda p: polygon_area_px(p, w, h))]
                    elif args.multipart == "merge":
                        polys = [merge_multi_segment(polys)]
                for p in polys:  # 'split' writes one line per part
                    seg_lines.append(f"{cid} " + " ".join(fmt(v) for v in p))
                n_inst += 1
            for folder, lines in label_sets(det_lines, seg_lines):
                (out / folder / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            mf.write(f"{stem}{ext},{src.name},{idx},{w},{h},{len(anns)},{'|'.join(classes)},{'|'.join(ids)}\n")
            n_img += 1
    mf.close()
    write_data_yaml(out, names)
    print(f"{n_img} images, {n_inst} instances -> {out}  (task={args.task}, classes={names})")
    if n_multi and args.task != "detect":
        print(f"{n_multi} instances had multi-part segmentation (handled with --multipart {args.multipart})")
    if n_skipped:
        print(f"{n_skipped} images without annotations skipped (use --keep-empty to keep them)")


# --------------------------------------------------------------------------- yolo -> veb
def yolo2veb(args) -> None:
    ds = Path(args.dataset)
    img_dir = ds / "images"
    lbl_dir = ds / (args.labels_dir or ("labels_seg" if (ds / "labels_seg").is_dir() else "labels"))
    print(f"reading labels from {lbl_dir.name}/")
    names = load_names(ds)
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    out = []
    for p in images:
        rel = p.relative_to(img_dir).with_suffix(".txt")
        lbl = lbl_dir / rel
        w, h = Image.open(p).size
        anns = []
        if lbl.exists():
            for k, line in enumerate(lbl.read_text(encoding="utf-8").split("\n")):
                vals = line.split()
                if not vals:
                    continue
                cid, nums = int(vals[0]), [float(v) for v in vals[1:]]
                cname = names[cid] if cid < len(names) else str(cid)
                if len(nums) == 4:  # detection box
                    cx, cy, bw, bh = nums
                    x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                    poly = [x0, y0, x1, y0, x1, y1, x0, y1]
                elif len(nums) >= 6 and len(nums) % 2 == 0:  # segmentation polygon
                    poly = nums
                else:
                    sys.exit(f"{lbl}: line {k + 1} is neither a box (4 values) nor a polygon")
                bbox = bbox8_from_points(poly[0::2], poly[1::2])
                anns.append({
                    "id": f"{cname}_{len(anns)}" if args.unique_ids else cname,
                    "class": cname,
                    "bbox": bbox,
                    "segmentation": [poly],
                    "area": int(round(polygon_area_px(poly, w, h))),
                    "is_crowd": 0,
                })
        out.append({"annotations": anns, "width": w, "height": h, "image": encode_data_uri(p)})
    Path(args.output).write_text(json.dumps(out), encoding="utf-8")
    print(f"{len(out)} images, {sum(len(e['annotations']) for e in out)} instances -> {args.output}")


# --------------------------------------------------------------------------- show
def _draw(im: Image.Image, anns: list[tuple[str, list[float], list[list[float]]]]) -> Image.Image:
    im = im.convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, W // 400)
    for cname, box, polys in anns:
        for p in polys:
            pts = [(p[i] * W, p[i + 1] * H) for i in range(0, len(p) - 1, 2)]
            if len(pts) >= 2:
                d.line(pts + [pts[0]], fill=(0, 220, 0), width=lw)
        if box:
            d.rectangle([box[0] * W, box[1] * H, box[2] * W, box[3] * H], outline=(255, 40, 40), width=lw)
            d.text((box[0] * W + 4, max(0, box[1] * H - 14)), cname, fill=(255, 40, 40))
    return im


def show(args) -> None:
    src, out = Path(args.source), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    if src.is_file():
        data = json.loads(src.read_text(encoding="utf-8"))
        for i, e in enumerate(data):
            items.append((f"{src.stem}_{i:04d}", lambda e=e: Image.open(io.BytesIO(decode_data_uri(e["image"])[0])),
                          [(a["class"], a["bbox"][:4], a.get("segmentation", [])) for a in e["annotations"]]))
    else:
        names = load_names(src)

        def read_lines(folder, p):
            f = src / folder / p.relative_to(src / "images").with_suffix(".txt")
            if not f.exists():
                return []
            return [[float(x) for x in l.split()] for l in f.read_text().split("\n") if l.split()]

        has_seg = (src / "labels_seg").is_dir()
        for p in sorted(x for x in (src / "images").rglob("*") if x.suffix.lower() in IMG_EXT):
            anns = []
            boxes = read_lines("labels", p)
            polys = read_lines("labels_seg", p) if has_seg else []
            for k, v in enumerate(boxes):
                cid, n = int(v[0]), v[1:]
                cname = names[cid] if cid < len(names) else str(cid)
                if len(n) == 4:
                    box = [n[0] - n[2] / 2, n[1] - n[3] / 2, n[0] + n[2] / 2, n[1] + n[3] / 2]
                    poly = [polys[k][1:]] if k < len(polys) else []
                else:
                    box, poly = bbox8_from_points(n[0::2], n[1::2])[:4], [n]
                anns.append((cname, box, poly))
            items.append((p.stem, lambda p=p: Image.open(p), anns))
    random.seed(args.seed)
    pick = items if len(items) <= args.n else random.sample(items, args.n)
    for stem, loader, anns in pick:
        _draw(loader(), anns).save(out / f"{stem}_overlay.jpg", quality=90)
    print(f"{len(pick)} overlays -> {out}  (red = box, green = segmentation)")


# --------------------------------------------------------------------------- cli
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("veb2yolo", help="VEB.js JSON export(s) -> YOLO dataset folder")
    a.add_argument("inputs", nargs="+", help="VEB.js JSON files (one per capture run)")
    a.add_argument("-o", "--output", required=True, help="dataset folder (created; runs are appended)")
    a.add_argument("--task", choices=["detect", "segment", "both"], default="detect",
                   help="detect: labels/ = class cx cy w h | segment: labels/ = class x1 y1 x2 y2 ... | "
                        "both: labels/ (boxes) + labels_seg/ (polygons) (default detect)")
    a.add_argument("--multipart", choices=["merge", "largest", "split"], default="merge",
                   help="segment task, instance made of several polygons: merge into one outline (default), "
                        "keep the largest part, or split into one line per part")
    a.add_argument("--classes", help="text file with class names in id order (fixes class ids)")
    a.add_argument("--rename", nargs="*", metavar="OLD=NEW", help="rename VEB.js classes, e.g. P07=tslot_plate")
    a.add_argument("--prefix", help="file-name prefix for this run (default: JSON file name)")
    a.add_argument("--keep-empty", action="store_true", help="keep images without annotations (empty label file)")
    a.set_defaults(func=veb2yolo)

    b = sub.add_parser("yolo2veb", help="YOLO dataset folder -> VEB.js JSON")
    b.add_argument("dataset", help="folder with images/, labels/, data.yaml")
    b.add_argument("-o", "--output", required=True, help="output .json")
    b.add_argument("--labels-dir", help="label folder to read (default: labels_seg if present, else labels)")
    b.add_argument("--unique-ids", action="store_true", help="id = <class>_<n> instead of the class name")
    b.set_defaults(func=yolo2veb)

    c = sub.add_parser("show", help="draw labels on sample images for a visual check")
    c.add_argument("source", help="VEB.js JSON file or YOLO dataset folder")
    c.add_argument("-o", "--output", default="overlays")
    c.add_argument("-n", type=int, default=20, help="number of images (default 20)")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=show)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
