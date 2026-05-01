"""
Unit tests for geo_utils.py

Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geo_utils import (
    BOUNDS,
    calculate_slope_aspect,
    calculate_curvature,
    get_tile_list,
    get_worldcover_tiles,
    get_canopy_height_tiles,
)


# ---------------------------------------------------------------------------
# calculate_slope_aspect
# ---------------------------------------------------------------------------


class TestSlopeAspect:
    def test_flat_surface_zero_slope(self):
        """A flat surface should have zero slope everywhere (except edges)."""
        flat = np.ones((10, 10)) * 100.0
        slope, aspect = calculate_slope_aspect(flat, res=30.0)
        # Interior pixels should be ~0
        assert slope[2:-2, 2:-2].max() < 0.01

    def test_north_facing_slope(self):
        """A surface tilting north (decreasing y = increasing lat) → aspect ~0°."""
        elev = np.zeros((10, 10))
        for i in range(10):
            elev[i, :] = i * 30.0  # increases downward (south), so faces north
        slope, aspect = calculate_slope_aspect(elev, res=30.0)
        # Slope should be ~45° (rise = run at 30m spacing)
        assert abs(slope[5, 5] - 45.0) < 1.0
        # Aspect should be ~0° (north) or ~360°
        center_aspect = aspect[5, 5]
        assert center_aspect < 10 or center_aspect > 350

    def test_east_facing_slope(self):
        """A surface tilting east → aspect ~90°."""
        elev = np.zeros((10, 10))
        for j in range(10):
            elev[:, j] = -j * 30.0  # decreases eastward
        slope, aspect = calculate_slope_aspect(elev, res=30.0)
        assert abs(aspect[5, 5] - 90.0) < 10.0

    def test_output_shapes(self):
        """Output shapes should match input."""
        elev = np.random.rand(20, 30) * 1000
        slope, aspect = calculate_slope_aspect(elev)
        assert slope.shape == (20, 30)
        assert aspect.shape == (20, 30)

    def test_aspect_range(self):
        """Aspect should be in [0, 360)."""
        elev = np.random.rand(50, 50) * 2000
        _, aspect = calculate_slope_aspect(elev)
        assert aspect.min() >= 0
        assert aspect.max() < 360


# ---------------------------------------------------------------------------
# calculate_curvature
# ---------------------------------------------------------------------------


class TestCurvature:
    def test_flat_surface_zero_curvature(self):
        """Flat surface → zero curvature."""
        flat = np.ones((20, 20)) * 500.0
        prof, plan = calculate_curvature(flat, res=30.0)
        assert np.abs(prof[3:-3, 3:-3]).max() < 1e-6
        assert np.abs(plan[3:-3, 3:-3]).max() < 1e-6

    def test_planar_slope_zero_curvature(self):
        """A planar tilted surface has zero curvature."""
        elev = np.zeros((20, 20))
        for i in range(20):
            for j in range(20):
                elev[i, j] = 2 * i + 3 * j  # linear plane
        prof, plan = calculate_curvature(elev, res=30.0)
        # Interior should be ~0 (edges have boundary effects)
        assert np.abs(prof[3:-3, 3:-3]).max() < 1e-6
        assert np.abs(plan[3:-3, 3:-3]).max() < 1e-6

    def test_concave_surface_negative_profile(self):
        """A bowl (concave up) should have negative profile curvature."""
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        elev = X**2 + Y**2  # paraboloid (bowl)
        prof, plan = calculate_curvature(elev, res=1.0)
        # Center of bowl should have negative profile curvature
        # (concave = decelerating flow)
        assert prof[25, 25] < 0

    def test_output_shapes(self):
        elev = np.random.rand(30, 40) * 1000
        prof, plan = calculate_curvature(elev)
        assert prof.shape == (30, 40)
        assert plan.shape == (30, 40)


# ---------------------------------------------------------------------------
# Tile list functions
# ---------------------------------------------------------------------------


class TestTileLists:
    def test_get_tile_list_default_bounds(self):
        """Default bounds (Switzerland) should produce ~24 DEM tiles."""
        tiles = get_tile_list()
        assert len(tiles) > 10
        assert len(tiles) < 50
        assert all("copernicus-dem-30m" in t for t in tiles)
        assert all(t.endswith(".tif") for t in tiles)

    def test_get_tile_list_custom_bounds(self):
        """Small bounds should produce fewer tiles."""
        small = {"lat_min": 46.0, "lat_max": 47.0, "lon_min": 7.0, "lon_max": 8.0}
        tiles = get_tile_list(small)
        assert len(tiles) == 1  # one 1°×1° tile

    def test_worldcover_tiles_default(self):
        """Switzerland needs 6 ESA Worldcover tiles (3°×3° grid)."""
        tiles = get_worldcover_tiles()
        assert len(tiles) == 6
        assert all("esa-worldcover" in t for t in tiles)
        assert all("v200" in t for t in tiles)

    def test_worldcover_tiles_naming(self):
        """Tile names should follow the expected pattern."""
        tiles = get_worldcover_tiles()
        for t in tiles:
            assert "ESA_WorldCover_10m_2021_v200_" in t
            assert "_Map.tif" in t

    def test_canopy_height_tiles_default(self):
        """Switzerland needs 6 canopy height tiles (same 3°×3° grid)."""
        tiles = get_canopy_height_tiles()
        assert len(tiles) == 6
        assert all("ETH_GlobalCanopyHeight" in t for t in tiles)


# ---------------------------------------------------------------------------
# BOUNDS constant
# ---------------------------------------------------------------------------


class TestBounds:
    def test_bounds_covers_switzerland(self):
        """BOUNDS should cover Switzerland (roughly 45.8-47.8°N, 5.9-10.5°E)."""
        assert BOUNDS["lat_min"] < 45.8
        assert BOUNDS["lat_max"] > 47.8
        assert BOUNDS["lon_min"] < 5.9
        assert BOUNDS["lon_max"] > 10.5

    def test_bounds_not_too_large(self):
        """BOUNDS shouldn't extend far beyond Switzerland."""
        assert BOUNDS["lat_min"] > 44.0
        assert BOUNDS["lat_max"] < 49.0
        assert BOUNDS["lon_min"] > 4.0
        assert BOUNDS["lon_max"] < 12.0
