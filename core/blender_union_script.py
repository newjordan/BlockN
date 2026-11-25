# Blender script to union multiple meshes
import bpy
import sys
import os
import argparse
import addon_utils # Import addon_utils to manage addons
from typing import List # Add this import

# Update the function signature below
def _apply_boolean_union(target, other) -> bool:
    """Union ``other`` into ``target`` and delete ``other``."""
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    modifier = target.modifiers.new(name="Union", type="BOOLEAN")
    modifier.operation = "UNION"
    modifier.object = other
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    except Exception:
        try:
            with bpy.context.temp_override(object=target, active_object=target):
                bpy.ops.object.modifier_apply(modifier=modifier.name)
        except Exception as exc:
            print(f"  Error applying boolean union of {other.name} into {target.name}: {exc}")
            return False
    bpy.data.objects.remove(other, do_unlink=True)
    return True


def union_meshes(input_files: List[str], output_file: str):
    """Import meshes, boolean-union them, and export the solid."""
    print(f"Union Script: Processing {len(input_files)} inputs for output {output_file}")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    imported_objs = []
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"  Warning: Input file not found, skipping: {input_file}")
            continue
        try:
            bpy.ops.import_scene.glb(filepath=input_file)
            current_selection = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
            imported_objs.extend(current_selection)
            print(f"  Imported {len(current_selection)} mesh(es) from {input_file}")
            bpy.ops.object.select_all(action="DESELECT")
        except Exception as exc:
            print(f"  Error importing file {input_file}: {exc}")

    if not imported_objs:
        print("  Error: No valid mesh objects were imported. Cannot perform union.")
        return False

    target = imported_objs[0]
    if len(imported_objs) == 1:
        print("  Only one mesh imported, exporting it directly.")
    else:
        print(f"  Unioning {len(imported_objs)} meshes into {target.name}.")
        for other in imported_objs[1:]:
            if not _apply_boolean_union(target, other):
                return False

    try:
        if target.name not in bpy.data.objects:
            print(f"  Error: Union result '{target.name}' is not in the scene.")
            return False
        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        bpy.context.view_layer.objects.active = target
        bpy.ops.export_scene.glb(filepath=output_file, use_selection=True, export_format="GLB")
        print(f"  Exported union to {output_file}")
        return True
    except Exception as exc:
        print(f"  Error exporting union to {output_file}: {exc}")
        return False


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Blender Mesh Union Script")
    parser.add_argument('--input', action='append', required=True, help='Input GLB file path (can be specified multiple times)')
    parser.add_argument('--output', required=True, help='Output GLB file path')

    # Parse arguments after '--'
    try:
        idx = sys.argv.index("--") + 1
        argv = sys.argv[idx:]
    except ValueError:
        argv = sys.argv[1:] # Skip the script name itself

    args = parser.parse_args(argv)

    print("Blender Union Script Started")
    print(f"Args: Input={args.input}, Output={args.output}")

    if not args.input:
        print("Error: No input files provided.")
        sys.exit(1)

    success = union_meshes(args.input, args.output)

    print(f"Blender Union Script Finished (Success: {success})")
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
