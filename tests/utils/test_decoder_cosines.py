"""decoder_cosine_matrices: the four .npy artifacts, their exactness, the bitwise transpose."""

import numpy as np
import pytest
import torch as th

from diffing.utils.dictionary.analysis import decoder_cosine_matrices


def unit(w):
    return w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)


@pytest.fixture
def decoders():
    gen = th.Generator().manual_seed(0)
    return {
        "base": th.randn(7, 5, generator=gen),
        "ft": th.randn(7, 5, generator=gen),
    }


def test_the_four_matrices_hold_the_pairwise_cosines(tmp_path, decoders):
    written = decoder_cosine_matrices(decoders, tmp_path, dtype="float32", chunk_size=3)

    assert sorted(written) == [
        "cos_base.npy",
        "cos_base_ft.npy",
        "cos_ft.npy",
        "cos_ft_base.npy",
    ]
    base = np.load(tmp_path / "cos_base.npy")
    assert np.allclose(base, (unit(decoders["base"]) @ unit(decoders["base"]).T).numpy(), atol=1e-6)
    assert np.allclose(np.diag(base), 1.0, atol=1e-6)
    across = np.load(tmp_path / "cos_base_ft.npy")
    assert np.allclose(
        across, (unit(decoders["base"]) @ unit(decoders["ft"]).T).numpy(), atol=1e-6
    )


def test_cos_ft_base_is_the_bitwise_transpose_even_at_float16(tmp_path, decoders):
    decoder_cosine_matrices(decoders, tmp_path, dtype="float16", chunk_size=2, transpose_tile=3)

    forward = np.load(tmp_path / "cos_base_ft.npy")
    backward = np.load(tmp_path / "cos_ft_base.npy")
    assert forward.dtype == np.float16 and backward.dtype == np.float16
    assert np.array_equal(backward, forward.T)


def test_a_single_side_dictionary_writes_cos_base_alone(tmp_path, decoders):
    written = decoder_cosine_matrices({"base": decoders["base"]}, tmp_path, dtype="float32")

    assert sorted(written) == ["cos_base.npy"]
    assert not (tmp_path / "cos_ft.npy").exists()


def test_a_zero_decoder_row_yields_finite_cosines_not_nan(tmp_path, decoders):
    decoders["base"][2] = 0.0
    decoder_cosine_matrices(decoders, tmp_path, dtype="float32")

    assert np.isfinite(np.load(tmp_path / "cos_base.npy")).all()


def test_the_matrices_are_memory_mappable(tmp_path, decoders):
    decoder_cosine_matrices(decoders, tmp_path)

    assert isinstance(np.load(tmp_path / "cos_base.npy", mmap_mode="r"), np.memmap)


def test_mismatched_decoder_shapes_are_refused(tmp_path, decoders):
    with pytest.raises(AssertionError):
        decoder_cosine_matrices({"base": decoders["base"], "ft": th.randn(6, 5)}, tmp_path)
