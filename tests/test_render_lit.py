from core.architecture import blockcraft
from core.architecture.render_lit import light_levels, render_view, room_camera


def test_room_view_is_lit_and_framed():
    b = blockcraft.house("blocky", 1, width=9, depth=8, storeys=2)
    eye, look = room_camera(b, 1)
    # The eye stands inside the house, at a player's eye height.
    (lx, _, lz), (hx, _, hz) = b.bounds()
    assert lx < eye[0] < hx and lz < eye[2] < hz and abs(eye[1] - 2.62) < 1e-6
    im = render_view(b, eye, look, width=160, height=90, sun=0.2, glow=1.0)
    assert im.size == (160, 90)
    px = list(im.getdata())
    lum = [0.3 * r + 0.59 * g + 0.11 * bb for r, g, bb in px]
    # Not blank, and lit unevenly: lantern light falls off.
    assert max(lum) - min(lum) > 60


def test_lantern_light_falls_off():
    from core.architecture.blockmode import BlockBuild, state
    b = BlockBuild("test")
    for x in range(10):
        for z in range(10):
            b.set(x, 0, z, state("stone"))
            b.set(x, 4, z, state("stone"))
    b.set(5, 1, 5, state("lantern", hanging="false"))
    bl, sl = light_levels(b, (12, 6, 12), (0, 0, 0), sky=False)
    assert bl[5, 1, 5] == 15 and bl[6, 1, 5] == 14 and bl[9, 1, 5] == 11
    assert bl[5, 0, 5] == 0          # stone is opaque
