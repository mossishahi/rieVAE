"""Tests for model components and end-to-end forward pass."""
import pytest
import torch

from rieVAE import SCRVAE
from rieVAE.train.loss import SCRVAELoss, node_reconstruction_loss, node_kl_loss
from rieVAE.geometry.graph import euclidean_knn_graph


@pytest.fixture
def small_model() -> SCRVAE:
    return SCRVAE(
        dim_features=20,
        dim_latent=8,
        dim_edge=4,
        encoder_hidden=(32,),
        decoder_hidden=(32,),
        edge_hidden=(16,),
        dropout=0.0,
    )


@pytest.fixture
def small_graph() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(30, 20)
    return euclidean_knn_graph(x, k=4)


class TestSCRVAE:
    def test_forward_shapes(self, small_model, small_graph):
        x = torch.randn(30, 20)
        edge_index, _ = small_graph

        small_model.train()
        out = small_model(x, edge_index)

        E = edge_index.shape[1]
        assert out["x_hat"].shape == (30, 20)
        assert out["mu_node"].shape == (30, 8)
        assert out["var_node"].shape == (30, 8)
        assert out["z_node"].shape == (30, 8)
        assert out["mu_e"].shape == (E, 4)
        assert out["var_e"].shape == (E, 4)
        assert out["l_hat"].shape == (E, 20)

    def test_var_positive(self, small_model, small_graph):
        x = torch.randn(30, 20)
        edge_index, _ = small_graph
        out = small_model(x, edge_index)
        assert (out["var_node"] > 0).all(), "Node variances must be positive."
        assert (out["var_e"] > 0).all(), "Edge variances must be positive."

    def test_edge_decoder_no_bias(self, small_model):
        # EdgeDecoder's weight is stored as _W (private); the public interface
        # is the `weight` property. There is no bias parameter at all.
        # Verify by checking that no named parameter contains 'bias'.
        bias_params = [
            name for name, _ in small_model.edge_decoder.named_parameters()
            if "bias" in name.lower()
        ]
        assert len(bias_params) == 0, (
            f"Edge decoder must have no bias (required by Eckart-Young theorem). "
            f"Found bias parameters: {bias_params}"
        )

    def test_dim_edge_constraint(self):
        with pytest.raises(ValueError, match="dim_edge"):
            SCRVAE(dim_features=10, dim_latent=4, dim_edge=8)

    def test_frame_W_shape(self, small_model):
        W = small_model.frame_W
        assert W.shape == (20, 4), f"W shape should be (G, k) = (20, 4), got {W.shape}"

    def test_gram_matrix_shape(self, small_model):
        G = small_model.gram_matrix()
        assert G.shape == (4, 4), "Gram matrix should be (k, k)."

    def test_eval_uses_mean(self, small_model, small_graph):
        """In eval mode, z_node should equal mu_node (no sampling)."""
        x = torch.randn(30, 20)
        edge_index, _ = small_graph

        small_model.eval()
        with torch.no_grad():
            out = small_model(x, edge_index)

        assert torch.allclose(out["z_node"], out["mu_node"]), \
            "In eval mode, z_node must equal mu_node (no reparameterization)."

    def test_no_gradient_leakage_through_riem_targets(self, small_model, small_graph):
        """Riemannian targets are detached; decoder params must not receive gradient from them."""
        x = torch.randn(30, 20)
        edge_index, _ = small_graph

        small_model.train()
        for p in small_model.node_decoder.parameters():
            p.grad = None

        loss_fn = SCRVAELoss()
        out = small_model(x, edge_index)
        losses = loss_fn(
            x=x, x_hat=out["x_hat"],
            mu_node=out["mu_node"], var_node=out["var_node"],
            mu_e=out["mu_e"], var_e=out["var_e"],
            decoder=small_model.node_decoder,
            z_mu=out["mu_node"].detach(),
            edge_index=edge_index,
            W=small_model.frame_W,
        )
        losses["total"].backward()

        for name, p in small_model.node_decoder.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 1e-10:
                has_grad = True
                break
        else:
            has_grad = False

        assert has_grad, \
            "Node decoder must receive gradient from L_node_recon (not from L_Riemannian)."


class TestLoss:
    def test_reconstruction_loss_zero_perfect(self):
        x = torch.randn(10, 5)
        assert node_reconstruction_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_reconstruction_loss_positive(self):
        x = torch.randn(10, 5)
        x_hat = torch.randn(10, 5)
        assert node_reconstruction_loss(x_hat, x).item() > 0

    def test_kl_loss_zero_at_prior(self):
        mu = torch.zeros(10, 4)
        var = torch.ones(10, 4)
        kl = node_kl_loss(mu, var)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_loss_positive_off_prior(self):
        mu = torch.randn(10, 4)
        var = torch.ones(10, 4) * 2.0
        assert node_kl_loss(mu, var).item() > 0
