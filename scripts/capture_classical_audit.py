"""Capture the actual renderer for Doric/Ionic comparison and inspection."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from core.architecture import BuildRequest, build
from core.generators.classical import ORDER_SPECS, _build_column
from core.scene import Scene
from gui.viewport import ViewportWidget
from gui.camera import frame_cathedral


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"builds/masonry-upgrade/after")
    parser.add_argument("--performance",action="store_true")
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);app.setQuitOnLastWindowClosed(False)
    view=ViewportWidget();view.setAttribute(Qt.WA_ShowWithoutActivating);view.resize(950,480);view.show();app.processEvents()
    board=Image.new("RGB",(1920,1080),"black");draw=ImageDraw.Draw(board)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",24)
    receipt={"source":"actual Blocks Qt/VTK renderer","orders":{}}
    for i,order in enumerate(("doric","ionic")):
        result=build(BuildRequest(product="classical-building",variant=order,seed=73,detail="presentation"))
        assert result.quality.status=="pass"
        receipt["orders"][order]={"quality":"pass","fingerprint":result.document.fingerprint(),"members":len(result.scene.blocks)}
        for j,scene in enumerate((result.scene,None)):
            if scene is None:
                scene=Scene(use_collision=False);scene.support_check_mode="OFF";spec=ORDER_SPECS[order.title()]
                _build_column(scene,spec,0,0,0,.5,spec.shaft_height_ratio,spec.base_height_ratio,spec.base_width_ratio,spec.capital_height_ratio,spec.capital_width_ratio)
            view.display_meshes(scene.blocks);app.processEvents()
            points=np.vstack([b.mesh.vertices for b in scene.blocks])
            frame_cathedral(view.plotter,points,(-1.8,.8,2.2),margin=1.10)
            if j == 0:
                ids = {b.block_id for b in scene.blocks}
                assert set(view._batch_cell_ids) == ids
                assert len(view._batch_cell_ids) == sum(len(b.mesh.faces) for b in scene.blocks)
                selected = []
                view.block_picked.connect(selected.append)
                renderer = view.plotter.renderer
                for block in scene.blocks:
                    point = block.mesh.bounds.mean(axis=0)
                    renderer.SetWorldPoint(*point, 1)
                    renderer.WorldToDisplay()
                    x, y, _ = renderer.GetDisplayPoint()
                    picker = view.plotter.iren.picker
                    picker.Pick(x, y, 0, renderer)
                    if picker.GetActor() == view._batch_actor:
                        view._handle_pick(picker.GetActor())
                        break
                view.block_picked.disconnect(selected.append)
                assert selected and selected[-1] in ids
                view._clear_highlight()
                receipt["orders"][order]["picking"] = "exact batch cell IDs and live selection pass"
            shot=Image.fromarray(view.plotter.screenshot(return_img=True))
            shot.save(out/(order+("-temple" if j==0 else "-column")+".png"))
            board.paste(shot,(i*960,j*540+40))
            draw.text((i*960+20,j*540+8),order.upper()+(" / TEMPLE" if j==0 else " / COLUMN"),fill="#00ff41",font=font)
        if args.performance:
            view.resize(1682,935);view.display_meshes(result.scene.blocks);app.processEvents()
            samples=[]
            for frame in range(120):
                start=time.perf_counter();view.plotter.camera.Azimuth(.15);view.plotter.render()
                elapsed=time.perf_counter()-start
                if frame>=15:samples.append(elapsed)
            receipt["orders"][order]["median_fps"]=1/float(np.median(samples))
            receipt["orders"][order]["p95_ms"]=float(np.percentile(samples,95))*1000
            view.resize(950,480);app.processEvents()
    board.save(out/"doric-ionic-audit.png")
    (out/"audit-capture.json").write_text(json.dumps(receipt,indent=2)+"\n")
    view.close();print(json.dumps(receipt))


if __name__=="__main__":main()
