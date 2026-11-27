"""Import a glTF file and write an FBX. Blender runs this in background mode.

    blender -b --python blender_fbx_script.py -- source.glb output.fbx
"""

import sys

import bpy


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("usage: blender -b --python blender_fbx_script.py -- source.glb output.fbx")
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    if len(arguments) != 2:
        raise SystemExit("usage: blender -b --python blender_fbx_script.py -- source.glb output.fbx")
    source, output = arguments
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source)
    bpy.ops.export_scene.fbx(filepath=output, use_selection=False, path_mode="COPY")


if __name__ == "__main__":
    main()
