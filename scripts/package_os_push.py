"""Create portable source/plugin archives and a reviewed OS integration bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def archive(target, members):
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for source, name in sorted(members, key=lambda item: item[1]):
            bundle.write(source, name, compress_type=zipfile.ZIP_STORED if source.suffix in {".mp4", ".zip", ".png"} else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive integrity check failed: " + str(target))
    return {"path": target.name, "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "builds/cathedral-upgrade/release")
    parser.add_argument("--media", type=Path, default=ROOT / "builds/cathedral-upgrade/showcase")
    parser.add_argument("--assets", type=Path, default=ROOT / "builds/cathedral-upgrade/cathedral-studies")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    allowed = {".py", ".md", ".json", ".toml", ".txt", ".svg", ".qss"}
    files = []
    for directory in ("core", "gui", "docs", "examples", "plugins", "integrations", "scripts", "tests"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in allowed and "__pycache__" not in path.parts:
                files.append((path, str(path.relative_to(ROOT))))
    for name in ("README.md", "LICENSE", "LICENSE.md", "pyproject.toml", "requirements.txt", "blockn.py", "cli.py", "config.py", "main.py"):
        if (ROOT / name).is_file():
            files.append((ROOT / name, name))
    records = [archive(out / "blocks-source.zip", files)]
    plugin = ROOT / "plugins/blocks-kitbash"
    records.append(archive(out / "blocks-kitbash-plugin.zip", [(p, "blocks-kitbash/" + str(p.relative_to(plugin))) for p in plugin.rglob("*") if p.is_file() and "__pycache__" not in p.parts]))
    if not args.source_only:
        media = args.media.resolve()
        showcase = json.loads((media / "showcase-manifest.json").read_text())
        if hashlib.sha256((media / showcase["video"]).read_bytes()).hexdigest() != showcase["sha256"]:
            raise RuntimeError("Showcase checksum mismatch")
        members = [(out / item["path"], item["path"]) for item in records]
        if (out / "validation.json").is_file():
            members.append((out / "validation.json", "validation.json"))
        evidence = out / "validation"
        if evidence.is_dir():
            members.extend((p, "validation/" + str(p.relative_to(evidence)))
                           for p in evidence.rglob("*") if p.is_file())
        members.extend((p, "integration/" + str(p.relative_to(ROOT / "integrations/os"))) for p in (ROOT / "integrations/os").rglob("*") if p.is_file())
        previews = showcase.get("preview_images", ["01-cathedral-hero.png", "workbench.png", "workbench-cutaway.png"])
        media_files = [showcase["video"], "showcase-manifest.json", "x-post.txt", *previews]
        score = showcase.get("audio_master", {}).get("file")
        if score:
            media_files.append(score)
        for name in ("quiet-courtyard-score.json", "video-validation.json"):
            if (media / name).is_file():
                media_files.append(name)
        for name in media_files:
            members.append((media / name, "showcase/" + name))
        assets = args.assets.resolve()
        batch = json.loads((assets / "batch.manifest.json").read_text())
        if batch["status"] != "pass":
            raise RuntimeError("Only a successful asset kit can be packaged")
        members.extend((p, assets.name + "/" + str(p.relative_to(assets))) for p in assets.rglob("*") if p.is_file())
        records.append(archive(out / "blocks-os-push.zip", members))
    receipt = {"status": "pass", "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "contains_working_tree_changes": True, "archives": records}
    (out / "release-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
