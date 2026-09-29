# ruff: noqa: F403, F405
# ─────────────────────────────────────────────────────────────
# MAP reconstruction animation (Manim CE): truth | advancing guess | truth - guess, each column the IC projected
# along every line of sight with the lensing kernel (seen through the faces of the box, as in notebook 14,
# section 11) above the two kappa maps.
#
# Panel PNGs come from animate_map_prep.py (project venv):
#   uv run --no-sync python animate_map_prep.py --out-dir ../data/PILOT_MESH512_DES --title "DES Y3"
#
# Render (from this directory) at 1080p60 to MP4, then a palette-optimised GIF from the MP4:
#   MAP_ANIM_FRAMES=../data/PILOT_MESH512_DES/anim_frames manim -qh animate_map_scene.py MapAnimation
#   ffmpeg -i media/videos/animate_map_scene/1080p60/MapAnimation.mp4 \
#     -vf "fps=20,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a" \
#     animate_map_scene.gif
#
# Manim Community v0.20.x
# ─────────────────────────────────────────────────────────────

import json
import os
from pathlib import Path

from manim import *  # Manim's idiom

config.background_color = "#FFFFFF"  # white scene; the panel PNGs carry a transparent bg

FRAME_DIR = Path(os.environ.get("MAP_ANIM_FRAMES", "../data/PILOT_MESH512_DES/anim_frames"))

LX, RX, DX = -4.75, 0.0, 4.75  # column centres: truth, guess, truth - guess
CUBE_Y, CUBE_H, KAPPA_H = 1.45, 2.35, 1.30
KAPPA_Y = (-1.25, -2.65)


class MapAnimation(Scene):
    def construct(self):
        meta = json.loads((FRAME_DIR / "frame_index.json").read_text())
        n, steps, loss = meta["n_frames"], meta["steps"], meta["loss"]
        r_kappa, r_proj = meta["r_kappa"], meta["r_proj"]

        def img(name, height):
            m = ImageMobject(str(FRAME_DIR / name))
            m.scale_to_fit_height(height)
            return m

        def kappa_column(name_of, x):
            """Two kappa maps with their bin labels, one per source bin."""
            # shifted right so the bin label to the left stays inside the column
            k = [img(name_of(p), KAPPA_H).move_to([x + 0.55, y, 0]) for p, y in enumerate(KAPPA_Y)]
            z = [
                MathTex(rf"\kappa\ \mathrm{{bin}}\ {p + 1}", color=GREY_E).scale(0.8).next_to(m, LEFT, buff=0.15)
                for p, m in enumerate(k)
            ]
            return Group(*k), VGroup(*z)

        def column(cube_name, kappa_of, x, caption):
            cube = img(cube_name, CUBE_H).move_to([x, CUBE_Y, 0])
            cap = Text(caption, font_size=19, color=BLACK).next_to(cube, DOWN, buff=0.02)
            k, z = kappa_column(kappa_of, x)
            arrow = Arrow(
                cap.get_bottom() + DOWN * 0.03,
                [x, k[0].get_top()[1] + 0.05, 0],
                buff=0.02,
                stroke_width=4,
                tip_length=0.16,
                color=GREY_D,
            )
            return cube, cap, arrow, k, z

        def readout_text(r):
            head = "first guess" if r == 0 else f"step {steps[r]}"
            parts = [
                head,
                f"loss {loss[r]:.5e}",
                "r(κ) " + " / ".join(f"{v:.2f}" for v in r_kappa[r]),
                f"r(projected IC) {r_proj[r]:.2f}",
            ]
            return Text("   ·   ".join(parts), font_size=22, color=GREY_D).move_to([0, -3.7, 0])

        title = Text(
            "Initial conditions reconstruction" + (f" — {meta['title']}" if meta["title"] else ""),
            font_size=30,
            color=BLACK,
        ).to_edge(UP, buff=0.18)
        heads = VGroup(
            *[
                Text(t, font_size=32, color=BLACK).move_to([x, 3.0, 0])
                for t, x in (("Truth", LX), ("Guess", RX), ("Truth − Guess", DX))
            ]
        )
        dividers = VGroup(
            *[
                DashedLine([x, 2.7, 0], [x, -3.4, 0], color=GREY_B, stroke_opacity=0.6, dash_length=0.12)
                for x in ((LX + RX) / 2, (RX + DX) / 2)
            ]
        )
        caption = "IC projected along each line of sight"
        t_cube, t_cap, t_arrow, t_k, t_z = column("truth_cube.png", lambda p: f"truth_kappa_{p}.png", LX, caption)
        g_cube, g_cap, g_arrow, g_k, g_z = column("cube_0.png", lambda p: f"kappa_0_{p}.png", RX, caption)
        d_cube, d_cap, d_arrow, d_k, d_z = column("diff_cube_0.png", lambda p: f"diff_kappa_0_{p}.png", DX, caption)
        readout = readout_text(0)

        # ── intro: truth column, then the first guess and the difference ──
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.6)
        self.play(FadeIn(heads, shift=DOWN * 0.2), Create(dividers), run_time=0.8)
        for cube, cap, arrow, k, z in (
            (t_cube, t_cap, t_arrow, t_k, t_z),
            (g_cube, g_cap, g_arrow, g_k, g_z),
            (d_cube, d_cap, d_arrow, d_k, d_z),
        ):
            self.play(FadeIn(cube), FadeIn(cap), run_time=0.5)
            self.play(GrowArrow(arrow), LaggedStart(*[FadeIn(m) for m in (*k, *z)], lag_ratio=0.2), run_time=0.7)
        self.play(FadeIn(readout), run_time=0.4)
        self.wait(0.8)

        # ── advance the guess across the saved MAP frames ──
        for r in range(1, n):
            anims = [
                Transform(readout, readout_text(r)),
                Transform(g_cube, img(f"cube_{r}.png", CUBE_H).move_to(g_cube.get_center())),
                Transform(d_cube, img(f"diff_cube_{r}.png", CUBE_H).move_to(d_cube.get_center())),
            ]
            for p in range(2):
                anims.append(Transform(g_k[p], img(f"kappa_{r}_{p}.png", KAPPA_H).move_to(g_k[p].get_center())))
                anims.append(Transform(d_k[p], img(f"diff_kappa_{r}_{p}.png", KAPPA_H).move_to(d_k[p].get_center())))
            self.play(*anims, run_time=0.7)
            self.wait(0.25)

        self.play(Circumscribe(Group(g_cube, *g_k), color=ORANGE, run_time=1.2))
        self.wait(2.0)
