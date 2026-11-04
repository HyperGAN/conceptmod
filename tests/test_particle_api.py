"""Keep historical DSL knobs honest against the actual 0.9.4 primitives."""
import pytest
import torch
from particlegan import GANLoss
from particlegan.grad_regularizers import GradientPenalty
from conceptmod.particle_api import make_gan_loss, make_gradient_penalty


@pytest.mark.parametrize("mode,name", [("rp", "relativistic"), ("vanilla", "non_saturating")])
def test_legacy_names_return_upstream_loss_and_gradients(mode, name):
    actual = make_gan_loss("logistic", mode)
    assert type(actual) is GANLoss
    assert actual.loss == name
    scores = torch.tensor([-2., .5, 1.], requires_grad=True)
    real = torch.tensor([.1, 1., -1.])
    loss = actual.g_loss(scores, real)
    expected = GANLoss(name).g_loss(scores, real)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(loss, scores)[0],
                               torch.autograd.grad(expected, scores)[0], rtol=0, atol=0)


def test_fixed_penalty_keeps_upstream_identity_and_rejects_removed_options():
    penalty = make_gradient_penalty(kappa=.2, lazy_k=3)
    assert type(penalty) is GradientPenalty
    assert penalty.arm == "b_cap" and penalty.kappa == .2 and penalty.lazy_k == 3
    for kwargs in ({"norm": "l1"}, {"target_anneal": "linear"}, {"arm": "g_interp_cap"}):
        with pytest.raises(ValueError):
            make_gradient_penalty(**kwargs)
    with pytest.raises(ValueError):
        make_gan_loss("logistic", "typo")
