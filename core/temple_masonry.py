"""Stone courses, bearing-aligned entablatures and open timber-roof envelopes.

The module authors real solids. Its dimensions are design proportions, not a
calculation of permissible stone spans or an archaeological reconstruction.
"""
from __future__ import annotations

import math
import numpy as np

from .architectural_geometry import extrude_profile, mesh_parameters
from .architecture.masonry import MAX_COURSE, MAX_STRETCHER, MIN_STONE, fill_box, iter_ashlar

STONE = (222, 213, 191, 255)
ROOF = (160, 112, 87, 255)
# Stylobate / stereobate ashlars: large paving, not cubit cubes and not one plate.
_PAVE_TARGET = 1.5
_EPS = 1e-9


def solid(scene, mesh, element, order, **tags):
    position, dimensions, parameters = mesh_parameters(mesh)
    return scene.add_block(position=np.asarray(position), dimensions=np.asarray(dimensions),
        block_type="MESH", color=ROOF if element in {"roof_panel", "roof_ridge"} else STONE,
        parameters={**parameters, "element": element, "order": order, **tags})


def block(scene, position, dimensions, element, order, color=STONE, **tags):
    return scene.add_block(position=np.asarray(position), dimensions=np.asarray(dimensions),
        block_type="CUBE", color=color, parameters={"element": element, "order": order, **tags})


def _paving_count(span):
    if span <= MAX_STRETCHER + 1e-9:
        return 1
    count = max(2, int(round(span / _PAVE_TARGET)))
    while span / count > MAX_STRETCHER + 1e-9:
        count += 1
    return count


def _paving_spans(start, end, offset):
    """Cover [start, end] with ~1.2–1.6 m stones; odd rows add a stone so joints stagger."""
    span = end - start
    count = _paving_count(span)
    if offset and count >= 2 and span / (count + 1) >= 1.2 - 1e-9:
        count += 1
    if count <= 1:
        return [(start, end)]
    module = span / count
    return [(start + module * index, start + module * (index + 1)) for index in range(count)]


def crepidoma(scene, width, depth, height, order):
    """Three stepped courses of large running-bond paving stones; the top is the stylobate."""
    tread = min(width, depth) * .035
    course_h = height / 3.0
    for course in range(3):
        extra = (2 - course) * tread
        x0, x1 = -extra, width + extra
        z0, z1 = -extra, depth + extra
        y = height * (course + 0.5) / 3.0
        dx, dz = x1 - x0, z1 - z0
        along_x = dx >= dz
        surface = "stylobate" if course == 2 else "crepidoma"
        minor = _paving_spans(z0, z1, False) if along_x else _paving_spans(x0, x1, False)
        for row, (a, b) in enumerate(minor):
            stagger = (course + row) % 2 == 1
            major = _paving_spans(x0, x1, stagger) if along_x else _paving_spans(z0, z1, stagger)
            for c, d in major:
                if along_x:
                    xa, xb, za, zb = c, d, a, b
                else:
                    xa, xb, za, zb = a, b, c, d
                block(scene, ((xa + xb) / 2.0, y, (za + zb) / 2.0),
                      (xb - xa, course_h, zb - za), "podium", order,
                      course=course, unit="paving", bond="running",
                      construction="crepidoma paving",
                      surface=surface)


def _emit_ashlar(scene, stones, element, order, color=STONE, **tags):
    for stone in stones:
        payload = {"construction": "ashlar running bond", **tags}
        payload.update(unit=stone.unit, course=stone.course, bond="running")
        block(scene, stone.position, stone.dimensions, element, order, color=color, **payload)


def _course_box(scene, center, dimensions, element, order, color=STONE, **tags):
    width, height, depth = (float(value) for value in dimensions)
    if min(width, height, depth) <= _EPS:
        return
    _emit_ashlar(scene, fill_box(center, dimensions), element, order, color=color, **tags)


def _rake_count(run, rise, band):
    """Boxes along a raking band; fill_box then splits each AABB to human scale."""
    run, rise, band = abs(float(run)), abs(float(rise)), max(float(band), _EPS)
    if run <= _EPS or rise <= _EPS:
        return 1
    count = max(1, int(math.ceil(math.hypot(run, rise) / MAX_STRETCHER - _EPS)))
    while rise / count + band > MAX_COURSE + _EPS or run / count > MAX_STRETCHER + _EPS:
        count += 1
        if count > 400:
            break
    return count


def stair_run(scene, center_x, center_z, width_x, width_z, ascend_axis, direction_sign,
              top_edge_coord, top_y, step_count, order, color=STONE, **tags):
    """Course each stair box as running-bond ashlars instead of one oversized slab."""
    if step_count <= 0 or top_y <= _EPS:
        return
    step_rise = float(top_y) / float(step_count)
    step_depth = float(width_z if ascend_axis == "Z" else width_x)
    for index in range(int(step_count)):
        current_height = step_rise * (index + 1)
        y_center = current_height / 2.0
        outward = (int(step_count) - index) - 0.5
        if ascend_axis == "Z":
            position = (float(center_x), y_center,
                        float(top_edge_coord) + float(direction_sign) * (step_depth * outward))
            dimensions = (float(width_x), current_height, step_depth)
        else:
            position = (float(top_edge_coord) + float(direction_sign) * (step_depth * outward),
                        y_center, float(center_z))
            dimensions = (step_depth, current_height, float(width_z))
        extra = {key: value for key, value in tags.items() if key not in {"element", "order", "color"}}
        _course_box(scene, position, dimensions, "stair", order, color=color,
                    step_index=int(index), **extra)


def entablature(scene, order, column_top_y, podium_top_y, x_min, x_max, z_min, z_max,
                member_width, front_column_centers, apply_symmetry,
                depth_multiplier=1., side_overhang_multiplier=1.):
    """The beam centerlines sit over the capitals, including both side runs."""
    left, right = x_min + member_width / 2, x_max - member_width / 2
    back, front = z_min + member_width / 2, z_max - member_width / 2
    scale = float(np.clip(depth_multiplier, .65, 1.8))
    side_scale = float(np.clip(side_overhang_multiplier, .65, 1.8))
    total = (column_top_y - podium_top_y) * order.entablature_ratio
    current = column_top_y
    name = order.name
    cornice_d = member_width * 1.20 * scale
    cornice_side = member_width * 1.20 * side_scale
    for role, fraction, projection in (("architrave", .40, .78), ("frieze", .36, .88), ("cornice", .24, 1.20)):
        height = total * fraction
        thickness, side = member_width * projection * scale, member_width * projection * side_scale
        levels = 3 if role == "architrave" and name in {"Ionic", "Corinthian", "Composite"} else 1
        for fascia in range(levels):
            fh = height / levels
            # Fasciae advance subtly as they rise; all retain a common bearing.
            fac = 1 - .10 * (levels - 1 - fascia) / levels
            zthick, xthick = thickness * fac, side * fac
            y = current + (fascia + .5) * fh
            for face, z in (("front", front), ("back", back)):
                if face == "back" and not apply_symmetry:
                    continue
                _course_box(scene, ((left + right) / 2, y, z), (right-left + xthick, fh, zthick),
                            role, name, face=face, fascia=fascia)
            if apply_symmetry:
                length = front - back - zthick
                if length > 0:
                    for face, x in (("left", left), ("right", right)):
                        _course_box(scene, (x, y, (front+back)/2), (xthick, fh, length),
                                    role, name, face=face, fascia=fascia)
        if role == "frieze":
            if name == "Doric":
                glyph_width = min(member_width * .38, height * .65)
                glyph_depth = member_width * .075
                # Corner glyphs finish the frieze; internal glyphs align over
                # columns and bay midpoints. No duplicate near-corner plaques.
                xs = [left-side/2+glyph_width/2, *front_column_centers[1:-1], right+side/2-glyph_width/2]
                xs += [(a+b)/2 for a,b in zip(front_column_centers[:-1],front_column_centers[1:])]
                for face, z, sign in (("front", front+thickness/2, 1), ("back", back-thickness/2, -1)):
                    if face == "back" and not apply_symmetry:
                        continue
                    for index, x in enumerate(sorted(set(xs))):
                        triglyph(scene, (x, current+height/2, z+sign*glyph_depth/2),
                                 glyph_width, height*.94, glyph_depth, name, face, False, index)
                if apply_symmetry:
                    shafts = [b for b in scene.blocks if b.parameters.get("element") == "shaft"]
                    zs = sorted({round(float(b.position[2]),6) for b in shafts
                                 if abs(float(b.position[0])-left) < 1e-4})
                    if len(zs) >= 2:
                        positions=[back-thickness/2+glyph_width/2, *zs[1:-1], front+thickness/2-glyph_width/2]
                        positions += [(a+b)/2 for a,b in zip(zs[:-1],zs[1:])]
                        for face,x,sign in (("left",left-side/2,-1),("right",right+side/2,1)):
                            for index,z in enumerate(sorted(set(positions))):
                                triglyph(scene,(x+sign*glyph_depth/2,current+height/2,z),glyph_width,height*.94,glyph_depth,name,face,True,index)
            elif name != "Tuscan":
                # A continuous, quiet moulding preserves the Ionic frieze.
                for face,z in (("front",front+thickness/2),("back",back-thickness/2)):
                    if face == "back" and not apply_symmetry:
                        continue
                    _course_box(scene,((left+right)/2,current+height*.93,z),
                          (right-left+side,height*.09,member_width*.07),"frieze_relief",name,face=face)
        current += height
    return dict(x_min=left-cornice_side/2,x_max=right+cornice_side/2,
                z_min=back-cornice_d/2,z_max=front+cornice_d/2,
                entablature_top_y=current,cornice_depth=cornice_d)


def triglyph(scene, center, width, height, depth, order, face, sideways, index):
    # Two complete vertical grooves in a connected slab. Three raised glyphs
    # remain; unlike applied blocks, the grooves belong to the solid profile.
    w, d = width/2, depth/2
    profile=[(-w,-d),(w,-d),(w,d),(.23*width,d),(.23*width,-.3*d),
             (.10*width,-.3*d),(.10*width,d),(-.10*width,d),
             (-.10*width,-.3*d),(-.23*width,-.3*d),(-.23*width,d),(-w,d)]
    mesh=extrude_profile(profile,height)
    # Profile coordinates X/Z, extrusion Y. Swap with sign to retain winding.
    mesh.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float))
    if face in {"front","right"}:
        mesh.apply_transform(np.diag([-1,1,-1,1]))
    if sideways:
        mesh.apply_transform(np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]],float))
    mesh.apply_translation(center)
    solid(scene,mesh,"triglyph",order,face=face,glyph=index,grooves=2)


def open_arch(scene, center_x, bottom, center_z, width, height, depth, order, element="arch_portal"):
    """Jambs, bearing and two half-heads as cubit ashlars. The opening stays a void."""
    half = float(width) / 2.0
    t = min(width * 0.16, max(depth * 0.35, 0.28))
    inner = max(half - t, width * 0.12)
    bearing = min(max(t, height * 0.08), height * 0.22)
    spring = min(max(height * 0.22, 0.16), height - bearing - 0.2)
    inner_rise = max(height - spring - bearing, 0.2)
    host = f"{element}.0"
    if spring > MIN_STONE * 0.5:
        _course_box(scene, (center_x - (half + inner) / 2.0, bottom + spring / 2.0, center_z),
                    (half - inner, spring, depth), element, order,
                    construction="ashlar running bond", host_id=host)
        _course_box(scene, (center_x + (half + inner) / 2.0, bottom + spring / 2.0, center_z),
                    (half - inner, spring, depth), element, order,
                    construction="ashlar running bond", host_id=host)
    if bearing > MIN_STONE * 0.5:
        _course_box(scene, (center_x, bottom + height - bearing / 2.0, center_z),
                    (width, bearing, depth), element, order,
                    construction="ashlar running bond", host_id=host)
    n = 10
    left = [(-half, spring), (-half, height - bearing), (0.0, height - bearing)]
    left.extend((inner * math.cos(th), spring + inner_rise * math.sin(th))
                for th in np.linspace(math.pi / 2.0, math.pi, n + 1)[1:])
    right = [(half, spring), (half, height - bearing), (0.0, height - bearing)]
    right.extend((inner * math.cos(th), spring + inner_rise * math.sin(th))
                 for th in np.linspace(math.pi / 2.0, 0.0, n + 1)[1:])
    _tile_profile(scene, left, depth, (center_x, bottom, center_z), element, order,
                  construction="ashlar running bond", host_id=host)
    _tile_profile(scene, right, depth, (center_x, bottom, center_z), element, order,
                  construction="ashlar running bond", host_id=host)


def dome_cap(scene, center_x, bottom, center_z, width, height, depth, order, element="dome_cap"):
    """Hollow oculus dome as cubit shell cells, not one DOME primitive."""
    from .architecture.classical_forms import _dome_shell_cells

    radius = min(float(width), float(depth)) / 2.0
    rise = float(height)
    if radius <= _EPS or rise <= _EPS:
        return
    thickness = min(radius, rise) * 0.14
    cells = _dome_shell_cells(radius, rise, thickness, float(bottom), 24, 10, oculus=0.12)
    sx = (float(width) / 2.0) / radius
    sz = (float(depth) / 2.0) / radius
    host = f"{element}.0"
    for course, cell in enumerate(cells):
        mesh = cell.copy()
        if abs(sx - 1.0) > 1e-9 or abs(sz - 1.0) > 1e-9:
            mesh.apply_scale((sx, 1.0, sz))
        mesh.apply_translation((float(center_x), 0.0, float(center_z)))
        extent = mesh.bounds[1] - mesh.bounds[0]
        if (float(extent[1]) > MAX_COURSE + 1e-6
                or max(float(extent[0]), float(extent[2])) > MAX_STRETCHER + 1e-6):
            continue
        solid(scene, mesh, element, order, construction="ashlar running bond",
              unit="profile_course", course=course, host_id=host)


def _tile_profile(scene, profile, depth, translation, element, order, construction, **tags):
    """Split a long extrusion into cubit bands so a roof or tympanum is tiles, not a slab."""
    from .architecture.cathedral import iter_fitted_profile_pieces

    run = float(depth)
    count = max(1, int(math.ceil(abs(run) / MAX_STRETCHER - 1e-9)))
    chunk = run / count
    host = tags.pop("host_id", None) or f"{element}.{tags.get('side', tags.get('face', 0))}"
    last = None
    for index in range(count):
        zc = translation[2] - run / 2.0 + (index + 0.5) * chunk
        try:
            pieces = list(iter_fitted_profile_pieces(
                profile, chunk, 0, translation[0], translation[1], zc))
        except ValueError:
            pieces = []
        for piece, course in pieces:
            mesh = extrude_profile(piece, chunk)
            mesh.apply_translation((translation[0], translation[1], zc))
            last = solid(scene, mesh, element, order, construction=construction,
                         course=course, tile=index, host_id=host, **tags)
    return last


def pediment(scene, center_x, base_y, center_z, width, depth, peak_height, order_name, face, **_):
    _tile_profile(
        scene, [(-width / 2, 0), (width / 2, 0), (0, peak_height)], depth,
        (center_x, base_y, center_z), "pediment", order_name,
        construction="ashlar running bond", unit="profile_course",
        face=face, host_id=f"pediment.{face}")
    band=min(depth*.26,peak_height*.10)
    run=width/2.0
    thick=depth*1.13
    count=_rake_count(run, peak_height, band)
    for side in (-1,1):
        x_eave=side*run
        for index in range(count):
            t0, t1 = index / count, (index + 1) / count
            x0, x1 = x_eave * (1.0 - t0), x_eave * (1.0 - t1)
            xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
            y0, y1 = peak_height * t0, peak_height * t1 + band
            _course_box(scene, (center_x + (xa + xb) / 2.0, base_y + (y0 + y1) / 2.0, center_z),
                        (xb - xa, y1 - y0, thick), "raking_cornice", order_name,
                        face=face, side=side, rake_index=index)


def roof(scene, center_x, base_y, z_min, z_max, width, rise, order, thickness):
    """Two thin roof planes enclosing an open attic, with a narrow ridge cap."""
    depth=z_max-z_min
    t=min(thickness,rise*.2)
    for side in (-1,1):
        profile=[(side*width/2,0),(0,rise),(0,rise+t),(side*width/2,t)]
        _tile_profile(
            scene, profile, depth, (center_x, base_y, (z_min + z_max) / 2),
            "roof_panel", order, construction="timber and tile roof",
            side=side, host_id=f"roof_panel.{side}")
    # Narrow ridge-to-eave cover-tile joints explain the roof's construction
    # without exposing triangulation or filling the silhouette with a grid.
    seam_count=max(12,min(20,int(round(depth/max(width*.08,.1)))))
    seam_width=t*.20
    for index in range(1,seam_count):
        z=z_min+depth*index/seam_count
        for side in (-1,1):
            profile=[(side*width/2,t),(0,rise+t),(0,rise+t+seam_width),
                     (side*width/2,t+seam_width)]
            _tile_profile(
                scene, profile, seam_width, (center_x, base_y, z),
                "roof_seam", order, construction="tile joint",
                side=side, seam=index, host_id=f"roof_seam.{side}.{index}")
    ridge_w, ridge_h = t * 2.6, t * .8
    _course_box(scene, (center_x, base_y + rise + t + ridge_h / 2.0, (z_min + z_max) / 2.0),
                (ridge_w, ridge_h, depth), "ridge", order, construction="ridge tile")
    return base_y+rise+t*1.8


def cella(scene, left, right, back, front, bottom, top, member_width, order):
    """An enclosed chamber of ashlar courses with a real doorway in the front wall."""
    inset=member_width*1.12
    x0,x1,z0,z1=left+inset,right-inset,back+inset,front-inset
    if x1-x0 < member_width*.9 or z1-z0 < member_width*.9:
        return False
    wall=member_width*.24
    height=top-bottom
    doorway=min((x1-x0)*.42,member_width*1.35)
    opening_top=bottom+height*.70
    if x1-x0-2*wall-doorway<=0:
        return False

    def emit(stones, element="cella_wall", **tags):
        for stone in stones:
            name = "cella_lintel" if stone.unit == "lintel" else element
            payload = dict(tags)
            payload.update(construction="ashlar running bond", unit=stone.unit,
                           course=stone.course, bond="running")
            block(scene, stone.position, stone.dimensions, name, order, **payload)

    emit(iter_ashlar(length=z1-z0-wall, height=height, thickness=wall, axis="z",
                     base_y=bottom, cross=x0, along_offset=(z0+z1)/2))
    emit(iter_ashlar(length=z1-z0-wall, height=height, thickness=wall, axis="z",
                     base_y=bottom, cross=x1, along_offset=(z0+z1)/2))
    emit(iter_ashlar(length=x1-x0, height=height, thickness=wall, axis="x",
                     base_y=bottom, cross=z0, along_offset=(x0+x1)/2))
    emit(iter_ashlar(length=x1-x0, height=height, thickness=wall, axis="x",
                     base_y=bottom, cross=z1, along_offset=(x0+x1)/2,
                     openings=[{"id": "cella.door", "kind": "door", "center": 0.0,
                                "width": doorway, "height": opening_top-bottom, "sill": 0.0}]),
         opening_width=doorway, opening_height=opening_top-bottom)
    return True
