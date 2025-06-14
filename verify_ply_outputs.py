#!/usr/bin/env python3
"""
PLY Output Verification Script
Tests the creation of different PLY formats and validates their structure
"""

import os
import numpy as np
import trimesh
from pathlib import Path
from plyfile import PlyData, PlyElement
import tempfile

def create_test_mesh():
    """Create a simple test mesh"""
    print("Creating test mesh...")
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f"✓ Test mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh

def create_gaussian_splatting_ply(mesh, num_points=1000):
    """Create Gaussian Splatting PLY format"""
    print(f"Creating Gaussian Splatting PLY ({num_points} points)...")
    
    # Sample points from mesh surface
    points, face_indices = mesh.sample(num_points, return_index=True)
    face_normals = mesh.face_normals[face_indices]
    
    # RGB to SH conversion (simplified)
    def RGB2SH(rgb):
        return rgb / (2 * 3.14159 ** 0.5)
    
    # Create attributes
    colors = np.ones((num_points, 3)) * 0.7  # Gray
    sh_dc = RGB2SH(colors)
    sh_rest = np.zeros((num_points, 45))  # 15 coefficients * 3 channels
    opacities = np.ones((num_points, 1)) * 2.0
    scales = np.ones((num_points, 3)) * (-3.0)
    rotations = np.zeros((num_points, 4))
    rotations[:, 0] = 1.0  # Identity quaternion
    
    # Create vertex data
    vertex_data = []
    for i in range(num_points):
        vertex = [
            points[i, 0], points[i, 1], points[i, 2],  # x, y, z
            face_normals[i, 0], face_normals[i, 1], face_normals[i, 2],  # nx, ny, nz
            sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2],  # f_dc_0, f_dc_1, f_dc_2
        ]
        vertex.extend(sh_rest[i])  # f_rest_0 through f_rest_44
        vertex.append(opacities[i, 0])  # opacity
        vertex.extend([scales[i, 0], scales[i, 1], scales[i, 2]])  # scales
        vertex.extend([rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]])  # rotations
        vertex_data.append(tuple(vertex))
    
    # Define properties
    properties = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    for i in range(45):
        properties.append((f'f_rest_{i}', 'f4'))
    properties.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])
    
    # Create PLY
    vertex_element = PlyElement.describe(
        np.array(vertex_data, dtype=properties),
        'vertex'
    )
    ply_data = PlyData([vertex_element])
    
    # Save to file
    ply_data.write("test_gaussian_splatting.ply")
    print(f"✓ Gaussian Splatting PLY created: {len(properties)} properties")
    return "test_gaussian_splatting.ply"

def create_viewable_mesh_ply(mesh):
    """Create viewable mesh PLY format"""
    print("Creating viewable mesh PLY...")
    
    # Export mesh as PLY
    mesh.export("test_viewable_mesh.ply")
    print(f"✓ Viewable mesh PLY created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return "test_viewable_mesh.ply"

def create_simple_points_ply(mesh, num_points=1000):
    """Create simple points PLY format"""
    print(f"Creating simple points PLY ({num_points} points)...")
    
    # Sample points
    points, _ = mesh.sample(num_points, return_index=True)
    
    # Create vertex data
    vertex_data = [(point[0], point[1], point[2]) for point in points]
    
    # Create PLY
    vertex_element = PlyElement.describe(
        np.array(vertex_data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
        'vertex'
    )
    ply_data = PlyData([vertex_element])
    
    # Save to file
    ply_data.write("test_simple_points.ply")
    print(f"✓ Simple points PLY created: {num_points} points")
    return "test_simple_points.ply"

def validate_ply_file(filename):
    """Validate PLY file structure"""
    print(f"\nValidating {filename}...")
    
    try:
        ply = PlyData.read(filename)
        
        if 'vertex' not in [elem.name for elem in ply.elements]:
            print(f"❌ No vertex element found in {filename}")
            return False
        
        vertex_element = ply['vertex']
        properties = [prop.name for prop in vertex_element.properties]
        
        print(f"✓ {filename}: {len(vertex_element)} vertices, {len(properties)} properties")
        print(f"  Properties: {properties[:5]}{'...' if len(properties) > 5 else ''}")
        
        # Check for faces
        if 'face' in [elem.name for elem in ply.elements]:
            face_element = ply['face']
            print(f"  Faces: {len(face_element)}")
        else:
            print(f"  Faces: None (point cloud)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating {filename}: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 PLY Output Verification")
    print("=" * 40)
    
    # Create test mesh
    mesh = create_test_mesh()
    
    # Create different PLY formats
    print("\n📁 Creating PLY files...")
    gs_ply = create_gaussian_splatting_ply(mesh, 1000)
    viewable_ply = create_viewable_mesh_ply(mesh)
    simple_ply = create_simple_points_ply(mesh, 1000)
    
    # Validate all files
    print("\n✅ Validating PLY files...")
    all_valid = True
    for ply_file in [gs_ply, viewable_ply, simple_ply]:
        if not validate_ply_file(ply_file):
            all_valid = False
    
    # Summary
    print("\n" + "=" * 40)
    if all_valid:
        print("🎉 All PLY files created and validated successfully!")
        print("\nFiles created:")
        print(f"  • {gs_ply} - Gaussian Splatting format (for validation)")
        print(f"  • {viewable_ply} - Mesh format (viewable in 3D software)")
        print(f"  • {simple_ply} - Point cloud format (simple viewing)")
    else:
        print("❌ Some PLY files failed validation")
    
    return all_valid

if __name__ == "__main__":
    main() 