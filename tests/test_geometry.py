"""Tests for geometry module: log maps, graph construction, curvature."""
import pytest
import torch
import numpy as np

from rieVAE.model.decoder import NodeDecoder
from rieVAE.geometry.log_map import (
    riemannian_log_map_single,
    riemannian_log_maps_batched,
    riemannian_distances,
)
from rieVAE.geometry.graph import (
    euclidean_knn_graph,
    riemannian_knn_graph,
    graph_changed,
)
from rieVAE.geometry.curvature import (
    find_triangles,
    ambient_closure_vectors,
    curvature_proxy,
)


@pytest.fixture
def linear_decoder() -> NodeDecoder:
    """Linear decoder f(z) = Az: J_f = A everywhere, H_f = 0."""
    dec = NodeDecoder(dim_latent=4, dim_out=8, hidden_dims=(), dropout=0.0)
    dec.eval()
    return dec


@pytest.fixture
def nonlinear_decoder() -> NodeDecoder:
    """Small nonlinear decoder for general tests."""
    dec = NodeDecoder(dim_latent=4, dim_out=8, hidden_dims=(16,), dropout=0.0)
    dec.eval()
    return dec


class TestLogMaps:
    def test_log_map_shape_single(self, nonlinear_decoder):
        z = torch.randn(4)
        dz = torch.randn(4)
        log_map = riemannian_log_map_single(nonlinear_decoder, z, dz)
        assert log_map.shape == (8,)

    def test_log_map_shape_batched(self, nonlinear_decoder):
        E, d, G = 10, 4, 8
        z_src = torch.randn(E, d)
        dz = torch.randn(E, d)
        log_maps = riemannian_log_maps_batched(nonlinear_decoder, z_src, dz)
        assert log_maps.shape == (E, G)

    def test_log_map_no_gradient_on_decoder(self, nonlinear_decoder):
        """Stop-gradient: decoder params must not receive gradient from log maps."""
        z_src = torch.randn(5, 4)
        dz = torch.randn(5, 4)
        for p in nonlinear_decoder.parameters():
            p.grad = None

        log_maps = riemannian_log_maps_batched(nonlinear_decoder, z_src, dz)
        loss = log_maps.sum()
        loss.backward()

        for p in nonlinear_decoder.parameters():
            assert p.grad is None, "Decoder params should have no gradient from log maps."

    def test_linear_decoder_log_map_exact(self, linear_decoder):
        """For a linear decoder f(z) = Az, the log map equals A @ dz exactly."""
        A = linear_decoder.net[-1].weight.data
        z_src = torch.zeros(1, 4)
        dz = torch.randn(1, 4)
        log_maps = riemannian_log_maps_batched(linear_decoder, z_src, dz)
        expected = dz @ A.T
        assert torch.allclose(log_maps, expected, atol=1e-4), \
            "Log map should equal J_f @ dz for linear decoder."

    def test_log_map_linearity_in_delta_z(self, nonlinear_decoder):
        """Log map is linear in delta_z (JVP is linear)."""
        z = torch.randn(1, 4)
        dz1 = torch.randn(1, 4)
        dz2 = torch.randn(1, 4)
        alpha = 2.5

        lm1 = riemannian_log_maps_batched(nonlinear_decoder, z, dz1)
        lm2 = riemannian_log_maps_batched(nonlinear_decoder, z, dz2)
        lm_sum = riemannian_log_maps_batched(nonlinear_decoder, z, alpha * dz1 + dz2)

        assert torch.allclose(lm_sum, alpha * lm1 + lm2, atol=1e-4), \
            "JVP must be linear in delta_z."

    def test_riemannian_distances_positive(self, nonlinear_decoder):
        z_src = torch.randn(8, 4)
        dz = torch.randn(8, 4)
        log_maps = riemannian_log_maps_batched(nonlinear_decoder, z_src, dz)
        dists = riemannian_distances(log_maps)
        assert (dists >= 0).all(), "Distances must be non-negative."
        assert dists.shape == (8,)


class TestGraph:
    def test_euclidean_knn_shape(self):
        x = torch.randn(20, 10)
        edge_index, edge_weight = euclidean_knn_graph(x, k=3)
        assert edge_index.shape[0] == 2
        assert edge_weight.shape[0] == edge_index.shape[1]

    def test_euclidean_knn_no_self_loops(self):
        x = torch.randn(20, 10)
        edge_index, _ = euclidean_knn_graph(x, k=3)
        src, dst = edge_index[0], edge_index[1]
        assert (src != dst).all(), "Self-loops must not appear in the graph."

    def test_euclidean_knn_symmetric(self):
        x = torch.randn(10, 5)
        edge_index, _ = euclidean_knn_graph(x, k=2)
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        edge_set = set(zip(src, dst))
        for s, d in zip(src, dst):
            assert (d, s) in edge_set, "Graph must be symmetric (both directions present)."

    def test_graph_changed_detects_difference(self):
        ei1 = torch.tensor([[0, 1], [1, 0]])
        ei2 = torch.tensor([[0, 2], [2, 0]])
        assert graph_changed(ei1, ei2)

    def test_graph_changed_same(self):
        ei = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
        assert not graph_changed(ei, ei)


class TestCurvature:
    def test_find_triangles_returns_tensor(self):
        ei = torch.tensor([[0, 1, 0, 1, 2, 2], [1, 0, 2, 2, 1, 0]])
        triangles = find_triangles(ei)
        assert triangles.ndim == 2
        assert triangles.shape[1] == 3

    def test_ambient_closure_linear_decoder(self, linear_decoder):
        """For a linear decoder, ambient closure is NOT zero in general
        (it depends on the Jacobian at different base points being the same --
        for a truly constant Jacobian = A, J_f(z_i) = J_f(z_j) = A, so:
        c_ijk = A @ (dz_ij + dz_jk + dz_ki) = A @ 0 = 0 exactly)."""
        N, d = 10, 4
        z_mu = torch.randn(N, d)
        ei = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 0, 1, 2, 0, 2]], dtype=torch.long)
        triangles = find_triangles(ei)

        if triangles.shape[0] == 0:
            pytest.skip("No triangles found in test graph.")

        c = ambient_closure_vectors(linear_decoder, z_mu, triangles)
        assert torch.allclose(c, torch.zeros_like(c), atol=1e-4), \
            "For a constant-Jacobian (linear) decoder, closure must be zero."

    def test_curvature_proxy_shape(self, nonlinear_decoder):
        N, d = 15, 4
        z_mu = torch.randn(N, d)
        ei, _ = euclidean_knn_graph(z_mu, k=4)
        triangles = find_triangles(ei)

        if triangles.shape[0] == 0:
            pytest.skip("No triangles found.")

        kappa = curvature_proxy(nonlinear_decoder, z_mu, triangles)
        assert kappa.shape == (triangles.shape[0],)
        assert (kappa >= 0).all(), "Curvature proxy must be non-negative."
