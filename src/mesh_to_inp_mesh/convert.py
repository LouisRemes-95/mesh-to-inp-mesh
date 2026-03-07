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
    if key is None:
        meshio.write(out_path, mesh, file_format="abaqus")
        return

    region_lut: dict[int, np.array] = {}
    points_chunks = []
    tetras_chunks = []
    offset = 0
    for region_id in set(mesh.cell_data_dict[key]['tetra']):
        region_tetras = mesh.cells_dict['tetra'][mesh.cell_data_dict[key]['tetra'] == region_id,:]
        region_points = np.unique(region_tetras.ravel())

        region_lut[region_id] = np.full(mesh.points.shape[0], 0, dtype=_smallest_uint_dtype(region_points.size + offset - 1))
        region_lut[region_id][region_points] = np.arange(region_points.size) + offset

        points_chunks.append(mesh.points[region_points, :])
        tetras_chunks.append(region_lut[region_id][region_tetras].astype(np.int64))
        offset += region_points.size

    out_points = np.vstack(points_chunks)
    out_tetras = np.vstack(tetras_chunks)

    tris = mesh.cells_dict['tetra'][:, [[0,2,1],[0,1,3],[1,2,3],[0,3,2]]].reshape(-1,3)
    regions = np.repeat(mesh.cell_data_dict[key]['tetra'], 4)[:, None]
    sorted_tris_region = np.hstack([np.sort(tris, axis = 1), regions])

    _, inverse, counts = np.unique(sorted_tris_region, axis=0, return_inverse=True, return_counts=True)
    is_boundary = counts[inverse] == 1
    tris = tris[is_boundary, :]
    sorted_tris_region = sorted_tris_region[is_boundary, :]

    _, index, inverse = np.unique(sorted_tris_region[:, :3], axis=0, return_index=True, return_inverse=True)
    tris_regions = np.hstack([tris, sorted_tris_region[:, 3:], sorted_tris_region[index[inverse], 3:]])
    keep = np.ones(len(tris_regions), dtype=bool)
    keep[index] = False
    tris_regions = tris_regions[keep]

    cohesive_mesh = meshio.Mesh(
        points = out_points,
        cells=[("tetra", out_tetras)]
        )

    meshio.write(out_path, cohesive_mesh, file_format="abaqus")

def _smallest_uint_dtype(max_value: int):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("Value too large")