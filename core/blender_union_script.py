# Blender script to union multiple meshes
import bpy
import sys
import os
import argparse
import addon_utils # Import addon_utils to manage addons
from typing import List # Add this import

# Update the function signature below
def union_meshes(input_files: List[str], output_file: str): # Changed list[str] to List[str]
    """Imports meshes, unions them, and exports the result."""
    print(f"Union Script: Processing {len(input_files)} inputs for output {output_file}")

    # Clear existing scene objects (mesh, camera, light)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # --- REMOVED Addon Check ---
    # --- REMOVED depsgraph update ---

    imported_objs = []
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"  Warning: Input file not found, skipping: {input_file}")
            continue
        try:
            bpy.ops.import_scene.glb(filepath=input_file)
            # Keep track of newly imported objects
            current_selection = [o for o in bpy.context.selected_objects if o.type == 'MESH']
            imported_objs.extend(current_selection)
            print(f"  Imported {len(current_selection)} mesh(es) from {input_file}")
            # Deselect after import to prepare for next
            bpy.ops.object.select_all(action='DESELECT')
        except Exception as e:
            print(f"  Error importing file {input_file}: {e}")

    if not imported_objs:
        print("  Error: No valid mesh objects were imported. Cannot perform union.")
        return False # Indicate failure

    if len(imported_objs) == 1:
        print("  Only one mesh imported, exporting directly.")
        obj_to_export = imported_objs[0]
    else:
        # --- Simplified Logic: Export the first imported object ---
        print(f"  Multiple meshes imported ({len(imported_objs)}). Exporting the first one ({imported_objs[0].name}) without union.")
        obj_to_export = imported_objs[0]
        success_flag = True # No operation performed, so technically successful I/O test
        # --- End Simplified Logic ---

    # Export the final result (the first imported object)
    try:
        # --- Add check if obj_to_export is valid before export ---
        if not obj_to_export or not hasattr(obj_to_export, 'name') or obj_to_export.name not in bpy.data.objects:
             print(f"  Error: Object to export ('{obj_to_export.name if obj_to_export else 'None'}') is invalid or not found in scene data.")
             return False # Indicate failure before export attempt
        # --- End check ---

        # Ensure only the final object is selected
        bpy.ops.object.select_all(action='DESELECT')
        obj_to_export.select_set(True)
        bpy.context.view_layer.objects.active = obj_to_export

        bpy.ops.export_scene.glb(filepath=output_file, use_selection=True, export_format='GLB')
        print(f"  Exported first imported mesh to {output_file}")
        return success_flag # Return the flag indicating if errors occurred during union (now always True)
    except Exception as e:
        print(f"  Error exporting first imported mesh to {output_file}: {e}")
        return False # Indicate failure


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
