import os
import sys
import re

try:
    import sapien.core as sapien
except ImportError:
    import sapien


def check_urdf_meshes(urdf_path):
    """
    Parse URDF and verify all referenced mesh files exist.
    Returns (ok, missing_files) tuple.
    """
    obj_dir = os.path.dirname(urdf_path)
    missing = []
    
    try:
        with open(urdf_path, 'r') as f:
            content = f.read()
        
        # Find all mesh filename references in URDF
        # Pattern matches: filename="something.obj" or filename="path/to/mesh.obj"
        mesh_pattern = r'filename=["\']([^"\']+\.(?:obj|stl|dae))["\']'
        meshes = re.findall(mesh_pattern, content, re.IGNORECASE)
        
        for mesh_ref in meshes:
            # Handle relative paths
            if not os.path.isabs(mesh_ref):
                mesh_path = os.path.join(obj_dir, mesh_ref)
            else:
                mesh_path = mesh_ref
            
            if not os.path.exists(mesh_path):
                missing.append(mesh_ref)
    
    except Exception as e:
        # If we can't parse, let SAPIEN handle it
        return True, []
    
    return len(missing) == 0, missing


def load_partnet_object(scene, partnet_root, obj_id):
    # 1. Resolve Absolute Paths (Fixes 'cannot make canonical path' errors)
    partnet_root = os.path.abspath(partnet_root)
    obj_dir = os.path.join(partnet_root, obj_id)
    urdf_path = os.path.join(obj_dir, "mobility.urdf")

    if not os.path.exists(urdf_path):
        raise RuntimeError(f"URDF not found: {urdf_path}")

    # 2. Pre-check mesh files before loading
    meshes_ok, missing_meshes = check_urdf_meshes(urdf_path)
    if not meshes_ok:
        raise RuntimeError(f"Missing mesh files: {missing_meshes[:3]}{'...' if len(missing_meshes) > 3 else ''}")

    # 3. Select Loader
    if hasattr(scene, 'create_urdf_loader'):
        loader = scene.create_urdf_loader()
    else:
        loader = sapien.URDFLoader(scene)

    loader.fix_root_link = True
    
    # 4. Safe Load
    try:
        robot = loader.load(urdf_path)
    except Exception as e:
        # This catches the "filesystem error" from C++ backend
        raise RuntimeError(f"SAPIEN failed to load URDF (missing meshes?): {e}")

    if robot is None:
        raise RuntimeError("URDF load returned None (corrupt dataset)")

    return robot