"""Conceptual schematic: the cost-cell complex and the three quantities.
Draws plan-optimality cells (a clipped Voronoi diagram, in the spirit of plan diagrams),
the true point, a small 'small-error' ball (kappa), a large 'large-error' spread (ACS / MSO).
Output: concept_cells.pdf  (used as a figure in the paper).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from scipy.spatial import Voronoi

# deterministic seed points -> plan-optimality cells
seeds = np.array([[3.0, 3.1], [6.4, 4.6], [7.0, 1.4], [4.6, 0.7],
                  [1.0, 1.0], [1.2, 5.0], [5.2, 5.4]])
BOX = (0, 8, 0, 6)
# cost-ratios per cell (the cell of the true point is optimal, r=1)
ratios = [1.0, 1.3, 2.4, 1.8, 6.1, 3.0, 2.0]
cstar = np.array([3.0, 3.1])  # true point, inside the optimal cell

vor = Voronoi(seeds)


def finite_polygons(vor, radius=20):
    new_regions, new_vertices = [], vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(p1, []).append((p2, v1, v2))
        ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        verts = vor.regions[region]
        if all(v >= 0 for v in verts):
            new_regions.append(verts); continue
        ridge = ridges[p1]
        new_region = [v for v in verts if v >= 0]
        for p2, v1, v2 in ridge:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]; t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices)); new_vertices.append(far.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        ang = np.arctan2(vs[:, 1] - vs[:, 1].mean(), vs[:, 0] - vs[:, 0].mean())
        new_region = [new_region[i] for i in np.argsort(ang)]
        new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)


def clip(poly, box):
    x0, x1, y0, y1 = box
    out = poly
    for (nx, ny, d) in [(1, 0, x0), (-1, 0, -x1), (0, 1, y0), (0, -1, -y1)]:
        res = []
        for i in range(len(out)):
            a, b = out[i], out[(i + 1) % len(out)]
            da, db = nx * a[0] + ny * a[1] - d, nx * b[0] + ny * b[1] - d
            if da >= 0:
                res.append(a)
            if (da >= 0) != (db >= 0):
                t = da / (da - db)
                res.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
        out = res
        if not out:
            break
    return np.asarray(out)


regions, vertices = finite_polygons(vor)
fig, ax = plt.subplots(figsize=(7.2, 5.2))
cmap = plt.cm.Blues
for i, reg in enumerate(regions):
    poly = clip(vertices[reg], BOX)
    if len(poly) < 3:
        continue
    shade = 0.05 + 0.11 * (ratios[i] - 1) / max(1e-9, max(ratios) - 1)
    ax.add_patch(Polygon(poly, closed=True, facecolor=cmap(shade), edgecolor="0.5", lw=1.0))
    c = poly.mean(axis=0)
    label = r"$P^\ast,\ r=1$" if ratios[i] == 1.0 else (
        rf"$r={ratios[i]}$" + ("\n(MSO)" if ratios[i] == max(ratios) else ""))
    ax.text(c[0], c[1], label, ha="center", va="center", fontsize=9)

ax.add_patch(Circle(cstar, 0.62, fill=False, ls="--", ec="#2a7", lw=1.6))
ax.add_patch(Circle(cstar, 1.9, fill=False, ls="--", ec="#c33", lw=1.4))
ax.plot(*cstar, "ko", ms=5)
ax.annotate(r"$c^\ast$ (true)", cstar, textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.annotate("small error:\nstays in $P^\\ast$  ($\\kappa$)", (cstar[0] + 0.45, cstar[1] - 0.45),
            color="#176", fontsize=8.5)
ax.annotate("large error: lands in any cell\n($\\mathrm{ACS}_\\infty$ = average,  MSO = worst)",
            (cstar[0] - 0.2, cstar[1] + 2.05), color="#a22", fontsize=8.5, ha="center")
ax.set_xlim(BOX[0], BOX[1]); ax.set_ylim(BOX[2], BOX[3])
ax.set_xlabel(r"$\log$ cardinality (sub-plan $A$)")
ax.set_ylabel(r"$\log$ cardinality (sub-plan $B$)")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Plan-optimality cells: one query, the cost-ratio spectrum $\\{r_k\\}$")
fig.tight_layout()
fig.savefig("concept_cells.pdf"); fig.savefig("concept_cells.png", dpi=150)
print("wrote concept_cells.pdf + .png")
