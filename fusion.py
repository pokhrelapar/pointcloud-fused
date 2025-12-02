import open3d as o3d
import numpy as np
import copy
import glob
import os

SCANS_FOLDER = "."
FLIP_FACES = True
REFINEMENT_ITERATIONS = 10


def preprocess(mesh, color):
    """
    Center, normalize, sample for point cloud, and outlier removal.
    """
    center = mesh.get_center()
    mesh.translate(-center)
    mesh.compute_vertex_normals()

    # Sample
    pcd = mesh.sample_points_poisson_disk(100_000)
    pcd.paint_uniform_color(color)

    # Flip
    if FLIP_FACES:
        R_flip = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        pcd.rotate(R_flip, center=(0, 0, 0))

    # Clean noise
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    return pcd


def get_rough_transform(face_name, cube_size):
    """Moves face from center to relative position"""

    dist = (cube_size / 2.0) * 0.85
    T = np.eye(4)

    if "top" in face_name:
        R = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
        T[:3, :3] = R
        T[:3, 3] = [0, dist, -dist]

    elif "right" in face_name:
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
        T[:3, :3] = R
        T[:3, 3] = [dist, 0, -dist]

    elif "left" in face_name:
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, -np.pi / 2, 0))
        T[:3, :3] = R
        T[:3, 3] = [-dist, 0, -dist]

    elif "bottom" in face_name:
        R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        T[:3, :3] = R
        T[:3, 3] = [0, -dist, -dist]

    elif "back" in face_name:
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0))
        T[:3, :3] = R
        T[:3, 3] = [0, 0, -dist * 2]

    return T


def align_centers_on_axis(source, target, axis_index):
    """align on a specific axis"""
    source_center = source.get_center()
    target_center = target.get_center()
    diff = target_center[axis_index] - source_center[axis_index]
    translation = [0, 0, 0]
    translation[axis_index] = diff
    source.translate(translation)
    return source


def main():
    files = glob.glob(os.path.join(SCANS_FOLDER, "*.stl"))
    if not files:
        print("No files found!")
        return

    # Load & Preprocess
    print("Loading and cleaning clouds...")
    clouds = {}
    for f in files:
        name = f.lower()
        if "front" in name:
            clouds["front"] = preprocess(o3d.io.read_triangle_mesh(f), [1, 0, 0])
        elif "back" in name:
            clouds["back"] = preprocess(o3d.io.read_triangle_mesh(f), [0, 1, 0])
        elif "top" in name:
            clouds["top"] = preprocess(o3d.io.read_triangle_mesh(f), [0, 0, 1])
        elif "bottom" in name:
            clouds["bottom"] = preprocess(o3d.io.read_triangle_mesh(f), [1, 1, 0])
        elif "left" in name:
            clouds["left"] = preprocess(o3d.io.read_triangle_mesh(f), [0, 1, 1])
        elif "right" in name:
            clouds["right"] = preprocess(o3d.io.read_triangle_mesh(f), [1, 0, 1])

    if "front" not in clouds:
        print("Error: 'front.stl' missing.")
        return

    bbox = clouds["front"].get_axis_aligned_bounding_box()
    cube_size = max(bbox.get_extent())
    print(f"Cube Size: {cube_size:.2f}")

    #  Initial placement
    aligned_clouds = {}
    aligned_clouds["front"] = clouds["front"]  # Front stays at 0,0,0

    for side in ["back", "top", "bottom", "left", "right"]:
        if side not in clouds:
            continue

        pcd = copy.deepcopy(clouds[side])
        pcd.transform(get_rough_transform(side, cube_size))

        # Lock Axis to Front
        if side in ["top", "bottom"]:
            pcd = align_centers_on_axis(pcd, aligned_clouds["front"], 0)
        elif side in ["left", "right"]:
            pcd = align_centers_on_axis(pcd, aligned_clouds["front"], 1)

        aligned_clouds[side] = pcd

    print(f"\n--- Starting Refinement ({REFINEMENT_ITERATIONS} passes) ---")

    sides_to_optimize = list(aligned_clouds.keys())
    # keep "front" fixed as the anchor
    sides_to_optimize.remove("front")

    for i in range(REFINEMENT_ITERATIONS):
        print(f"Pass {i+1}/{REFINEMENT_ITERATIONS}...")
        total_fitness = 0

        for side in sides_to_optimize:
            current_cloud = aligned_clouds[side]

            others = o3d.geometry.PointCloud()
            for other_name, other_cloud in aligned_clouds.items():
                if other_name != side:
                    others += other_cloud

            # Downsample target
            others = others.voxel_down_sample(cube_size * 0.02)

            # ICP: Align [Side] -> [Everyone Else]
            reg = o3d.pipelines.registration.registration_icp(
                current_cloud,
                others,
                cube_size * 0.03,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
            )

            # Apply transformation
            current_cloud.transform(reg.transformation)
            aligned_clouds[side] = current_cloud
            total_fitness += reg.fitness

        print(f"   Avg Fitness: {total_fitness / len(sides_to_optimize):.4f}")

    # 4. Final Merge & Visualize
    full_cloud = o3d.geometry.PointCloud()
    for name, pcd in aligned_clouds.items():
        full_cloud += pcd

    print("Done. Visualizing...")
    o3d.visualization.draw_geometries([full_cloud], window_name="Refined Cube")
    o3d.io.write_point_cloud("final_refined_1k.ply", full_cloud)


if __name__ == "__main__":
    main()
