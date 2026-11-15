"""Draws docs/architecture.svg, the README's pipeline diagram, in two rows (train & register -> ship & serve).

    python docs/make_architecture.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1100, 520
BW, BH, GAP, X0 = 150, 66, 30, 25
INK, MUTED, LINE = "#1f2937", "#5b6472", "#6b7280"
ROW1_FILL, ROW2_FILL = "#eef4fc", "#eefaf4"
ROW1_EDGE, ROW2_EDGE = "#2a78d6", "#1baf7a"

out = []
add = out.append


def x(i):
    return X0 + i * (BW + GAP)


def box(i, y, title, sub, edge):
    add(f'<rect x="{x(i)}" y="{y}" width="{BW}" height="{BH}" rx="10" fill="#ffffff" stroke="{edge}" stroke-width="1.6"/>')
    cx = x(i) + BW / 2
    add(f'<text x="{cx}" y="{y + 28}" text-anchor="middle" font-size="14" font-weight="600" fill="{INK}">{escape(title)}</text>')
    for k, line in enumerate(sub):
        add(f'<text x="{cx}" y="{y + 46 + 14 * k}" text-anchor="middle" font-size="11.5" fill="{MUTED}">{escape(line)}</text>')


def harrow(i, y):  # arrow from box i to box i+1 at height y
    add(f'<line x1="{x(i) + BW}" y1="{y}" x2="{x(i + 1) - 3}" y2="{y}" stroke="{LINE}" stroke-width="1.6" marker-end="url(#a)"/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    'font-family="Segoe UI, Helvetica, Arial, sans-serif">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" rx="14" fill="#ffffff"/>')

# ---- row 1: train & register (GPU machine)
add(f'<rect x="12" y="12" width="{W - 24}" height="226" rx="12" fill="{ROW1_FILL}" stroke="{ROW1_EDGE}" stroke-opacity="0.35"/>')
add(f'<text x="28" y="38" font-size="15" font-weight="700" fill="{ROW1_EDGE}">1 · Train, evaluate and register   (GPU machine)</text>')
y1 = 58
row1 = [("MSD dataset", ["484 labelled brains", "4 MRI scans each"]),
        ("make_split.py", ["340 train / 72 val", "/ 72 test, seed 42"]),
        ("train.py", ["3 networks,", "identical recipe"]),
        ("evaluate.py", ["72 test brains,", "Dice, HD95, Wilcoxon"]),
        ("export_onnx.py", ["PyTorch → ONNX,", "checked + timed"]),
        ("register.py", ["model registry,", "champion chosen"])]
for i, (t, s) in enumerate(row1):
    box(i, y1, t, s, ROW1_EDGE)
    if i < 5:
        harrow(i, y1 + BH / 2)

# MLflow store under the middle of row 1, fed by train.py and register.py
mx, my, mw, mh = x(2) + BW / 2 + 40, 160, 270, 52
add(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="26" fill="#ffffff" stroke="{ROW1_EDGE}" stroke-width="1.6"/>')
add(f'<text x="{mx + mw / 2}" y="{my + 22}" text-anchor="middle" font-size="14" font-weight="600" fill="{INK}">MLflow</text>')
add(f'<text x="{mx + mw / 2}" y="{my + 39}" text-anchor="middle" font-size="11.5" fill="{MUTED}">runs, metrics, curves · model registry</text>')
tx = x(2) + BW / 2
add(f'<path d="M{tx},{y1 + BH} V{my + mh / 2} H{mx - 3}" fill="none" stroke="{LINE}" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#a)"/>')
rx_ = x(5) + BW / 2
add(f'<path d="M{rx_ - 30},{y1 + BH} V{my + mh / 2} H{mx + mw + 3}" fill="none" stroke="{LINE}" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#a)"/>')

# ---- row 2 background (drawn before the connector so its arrowhead stays on top)
add(f'<rect x="12" y="278" width="{W - 24}" height="{H - 290}" rx="12" fill="{ROW2_FILL}" stroke="{ROW2_EDGE}" stroke-opacity="0.35"/>')

# ---- connector: end of row 1 (register.py) -> start of row 2 (Hugging Face Hub)
sx = x(5) + BW / 2 + 30
gy = 258
hx = x(0) + BW / 2
add(f'<path d="M{sx},{y1 + BH} V{gy} H{hx} V{300 - 3}" fill="none" stroke="{INK}" stroke-width="2" marker-end="url(#a)"/>')
add(f'<rect x="{W / 2 - 150}" y="{gy - 11}" width="300" height="22" rx="11" fill="#ffffff" stroke="{LINE}" stroke-width="1"/>')
add(f'<text x="{W / 2}" y="{gy + 4}" text-anchor="middle" font-size="12" fill="{INK}">scripts/hub.py upload: models, samples, gallery</text>')

# ---- row 2: ship & serve (CPU)
y2 = 300
row2 = [("Hugging Face Hub", ["ONNX models, sample", "brains, 3D gallery"]),
        ("GitHub Actions", ["tests, viewer build,", "Docker e2e test"]),
        ("Container image", ["GHCR, 2 CPUs / 16 GB", "deploy target"]),
        ("FastAPI service", ["upload checks, queue,", "ONNX Runtime (CPU)"]),
        ("brainviz", ["masks → 3D meshes", "+ scan volumes"]),
        ("3D viewer", ["three.js: toon, hatching,", "expert comparison"])]
for i, (t, s) in enumerate(row2):
    box(i, y2, t, s, ROW2_EDGE)
    if i < 5:
        harrow(i, y2 + BH / 2)
add(f'<text x="28" y="{H - 22}" font-size="15" font-weight="700" fill="{ROW2_EDGE}">2 · Ship and serve   (CPU only)</text>')
add(f'<text x="{x(1) + BW / 2}" y="{y2 - 8}" text-anchor="middle" font-size="11" fill="{MUTED}">on every git push</text>')

# user: uploads into the API, gets the 3D result from the viewer
ux, uy = x(3) + BW / 2 + 90, 455
add(f'<ellipse cx="{ux}" cy="{uy}" rx="46" ry="20" fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
add(f'<text x="{ux}" y="{uy + 5}" text-anchor="middle" font-size="14" font-weight="600" fill="{INK}">User</text>')
ax = x(3) + BW / 2
add(f'<path d="M{ux - 46},{uy} H{ax} V{y2 + BH + 3}" fill="none" stroke="{LINE}" stroke-width="1.6" marker-end="url(#a)"/>')
add(f'<text x="{ax - 8}" y="{y2 + BH + 40}" text-anchor="end" font-size="11.5" fill="{MUTED}">uploads 4 MRI scans</text>')
vx = x(5) + BW / 2
add(f'<path d="M{vx},{y2 + BH} V{uy} H{ux + 46 + 3}" fill="none" stroke="{LINE}" stroke-width="1.6" marker-end="url(#a)"/>')
add(f'<text x="{vx - 8}" y="{y2 + BH + 40}" text-anchor="end" font-size="11.5" fill="{MUTED}">tumour shown in 3D</text>')

add('</svg>')
Path(__file__).with_name("architecture.svg").write_text("\n".join(out))
print("wrote docs/architecture.svg")
