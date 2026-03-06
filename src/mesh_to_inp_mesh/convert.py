from pathlib import Path
import meshio
import numpy as np

def convert(in_path, out_path: Path) -> None:
    """ 
    Convert a meshio mesh file (used for .mesh) to .ply
    Intended for a tetrahedral mesh with cell data representing region index
    
    Inputs
    ------
    in_path:
        Input mesh path
    out_path:
        Output .ply path
    """

    mesh = meshio.read(in_path)
    
    key = next(iter(mesh.cell_data), None)
    if key is not None:

        region_lut: dict[int, np.array] = {}
        out_tetras = np.empty((0,4), dtype = np.int64)
        out_points = np.empty((0,3), dtype = np.float64)
        for region_id in set(mesh.cell_data_dict[key]['tetra']):
            region_tetras = mesh.cells_dict['tetra'][mesh.cell_data_dict[key]['tetra'] == region_id,:]
            region_points = np.unique(region_tetras.ravel())

            region_lut[region_id] = np.full(mesh.points.shape[0], 0, dtype=smallest_uint_dtype(region_points.size + out_points.shape[0] - 1))
            region_lut[region_id][region_points] = np.arange(region_points.size) + out_points.shape[0]

            out_points = np.vstack([out_points, mesh.points[region_points, :]])
            out_tetras = np.vstack([out_tetras, region_lut[region_id][region_tetras].astype(np.int64)])
    else:
        out_points, out_tris = surface_from_mesh(mesh.points, mesh.cells_dict['tetra'])

    if out_tris.max() > np.iinfo(np.int32).max:
        raise OverflowError("PLY cannot store indices > int32")

    out_tris = out_tris.astype(np.int32)

    surface_mesh = meshio.Mesh(
        points = out_points,
        cells=[("triangle", out_tris)]
        )

    meshio.write(out_path, surface_mesh, file_format="ply")

def surface_from_mesh(points, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tris = tets[:, [[0,1,2],[0,3,1],[1,3,2],[0,2,3]]].reshape(-1,3)

    _, inverse, counts = np.unique(np.sort(tris), axis = 0, return_inverse = True, return_counts = True)
    is_boundary_face = counts[inverse] == 1
    boundary_tris = tris[is_boundary_face, :]

    surface_point_id = np.unique(boundary_tris.flatten())

    lut = np.full(surface_point_id.max()+1, -1, dtype=np.int64)
    lut[surface_point_id] = np.arange(surface_point_id.size)

    return points[surface_point_id, :], lut[boundary_tris]

def smallest_uint_dtype(max_value: int):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("Value too large")