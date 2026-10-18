"""The gmsh backend's max_surface_deviation keeps facets within a distance of
the CAD surface, as the cad-to-dagmc-mesher tolerance does."""

import cadquery as cq
import gmsh
import numpy as np
import pytest
from test_python_api import get_volumes_and_materials_from_h5m

from cad_to_dagmc import CadToDagmc
from cad_to_dagmc.core import _chord_length, init_gmsh, set_surface_deviation_sizes

RADIUS = 10.0


def _box_and_sphere():
    """A box with a sphere sitting apart from it, in one assembly."""
    assembly = cq.Assembly()
    assembly.add(cq.Workplane().box(20, 20, 20), name="box")
    assembly.add(cq.Workplane().moveTo(40, 0).sphere(RADIUS), name="sphere")
    model = CadToDagmc()
    model.add_cadquery_object(assembly, material_tags=["steel", "water"])
    return model


def _sphere_deviation_and_triangle_counts(msh_file, centre, radius):
    """The largest distance of the sphere's facets from the true sphere, and
    the number of triangles on curved and on flat faces."""
    init_gmsh()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(msh_file))
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.zeros((int(tags.max()) + 1, 3))
    xyz[tags.astype(int)] = coords.reshape(-1, 3)
    deviation, curved, flat = 0.0, 0, 0
    for _, surface in gmsh.model.getEntities(2):
        _, _, node_tags = gmsh.model.mesh.getElements(2, surface)
        triangles = xyz[node_tags[0].astype(int).reshape(-1, 3)]
        distances = np.linalg.norm(triangles - centre, axis=2)
        if np.allclose(distances, radius, atol=1e-6 * radius):
            curved += len(triangles)
            # the centre and the edge midpoints of each facet
            points = np.concatenate(
                [
                    triangles.mean(axis=1),
                    (triangles[:, 0] + triangles[:, 1]) / 2,
                    (triangles[:, 1] + triangles[:, 2]) / 2,
                    (triangles[:, 2] + triangles[:, 0]) / 2,
                ]
            )
            deviation = max(
                deviation,
                np.abs(np.linalg.norm(points - centre, axis=1) - radius).max(),
            )
        else:
            flat += len(triangles)
    gmsh.finalize()
    return deviation, curved, flat


@pytest.mark.parametrize("max_surface_deviation", [0.1, 0.02])
def test_facets_stay_within_max_surface_deviation(tmp_path, max_surface_deviation):
    msh_file = tmp_path / "mesh.msh"
    _box_and_sphere().export_gmsh_mesh_file(
        filename=str(msh_file), max_surface_deviation=max_surface_deviation
    )
    deviation, curved, _ = _sphere_deviation_and_triangle_counts(
        msh_file, np.array([40.0, 0, 0]), RADIUS
    )
    assert curved > 0
    assert deviation <= max_surface_deviation


def test_max_surface_deviation_is_in_scaled_units(tmp_path):
    msh_file = tmp_path / "mesh.msh"
    _box_and_sphere().export_gmsh_mesh_file(
        filename=str(msh_file), scale_factor=10, max_surface_deviation=0.5
    )
    deviation, _, _ = _sphere_deviation_and_triangle_counts(
        msh_file, np.array([400.0, 0, 0]), 10 * RADIUS
    )
    assert deviation <= 0.5


def test_smaller_deviation_refines_only_curved_surfaces(tmp_path):
    counts = []
    for max_surface_deviation in [0.1, 0.01]:
        msh_file = tmp_path / f"mesh_{max_surface_deviation}.msh"
        _box_and_sphere().export_gmsh_mesh_file(
            filename=str(msh_file), max_surface_deviation=max_surface_deviation
        )
        _, curved, flat = _sphere_deviation_and_triangle_counts(
            msh_file, np.array([40.0, 0, 0]), RADIUS
        )
        counts.append((curved, flat))
    (coarse_curved, coarse_flat), (fine_curved, fine_flat) = counts
    assert fine_curved > 5 * coarse_curved
    # flat faces are left to the other size settings, so they are not refined
    assert fine_flat <= 1.1 * coarse_flat


@pytest.mark.parametrize("radius, deviation", [(1, 0.001), (1, 0.5), (50, 0.01)])
def test_chord_length_sags_by_the_deviation(radius, deviation):
    chord = _chord_length(1 / radius, deviation)
    sag = radius - np.sqrt(radius**2 - chord**2 / 4)
    assert sag == pytest.approx(deviation)


def test_chord_length_is_about_sqrt_8dr_for_small_deviations():
    assert _chord_length(1, 1e-6) == pytest.approx(np.sqrt(8e-6), rel=1e-6)


@pytest.mark.parametrize("deviation", [1, 3])
def test_chord_length_is_the_diameter_when_deviation_reaches_the_radius(deviation):
    assert _chord_length(1, deviation) == 2


def test_radius_below_max_surface_deviation_is_still_resolved(tmp_path):
    """A pin thinner than the deviation keeps its shape, rather than being
    meshed as coarsely as Gmsh allows."""
    assembly = cq.Assembly()
    assembly.add(cq.Workplane().box(20, 20, 20), name="box")
    assembly.add(cq.Workplane().moveTo(40, 0).sphere(0.5), name="pin")
    model = CadToDagmc()
    model.add_cadquery_object(assembly, material_tags=["steel", "water"])
    msh_file = tmp_path / "mesh.msh"
    model.export_gmsh_mesh_file(filename=str(msh_file), max_surface_deviation=1.0)
    deviation, curved, _ = _sphere_deviation_and_triangle_counts(
        msh_file, np.array([40.0, 0, 0]), 0.5
    )
    # sized at half the diameter: about 60 triangles and a tenth of the radius,
    # where the small-deviation formula gave about 30 and a quarter
    assert curved > 50
    assert deviation < 0.1


def test_dagmc_h5m_file_with_max_surface_deviation(tmp_path):
    h5m_file = tmp_path / "dagmc.h5m"
    _box_and_sphere().export_dagmc_h5m_file(
        filename=str(h5m_file), max_surface_deviation=0.05
    )
    assert h5m_file.is_file()
    assert set(get_volumes_and_materials_from_h5m(str(h5m_file)).values()) == {
        "mat:steel",
        "mat:water",
    }


def test_max_surface_deviation_selects_gmsh(capsys, tmp_path):
    _box_and_sphere().export_dagmc_h5m_file(
        filename=str(tmp_path / "dagmc.h5m"), max_surface_deviation=0.1
    )
    assert "Using meshing backend: gmsh" in capsys.readouterr().out


def test_max_surface_deviation_with_mesher_arguments_is_ambiguous(tmp_path):
    with pytest.raises(ValueError, match="Ambiguous backend"):
        _box_and_sphere().export_dagmc_h5m_file(
            filename=str(tmp_path / "dagmc.h5m"),
            max_surface_deviation=0.1,
            tolerance=0.1,
        )


@pytest.mark.parametrize(
    "value, error",
    [(0, ValueError), (-0.1, ValueError), (True, TypeError), ("0.1", TypeError)],
)
def test_invalid_max_surface_deviation(value, error):
    init_gmsh()
    gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.synchronize()
    try:
        with pytest.raises(error):
            set_surface_deviation_sizes(gmsh, value)
    finally:
        gmsh.finalize()
