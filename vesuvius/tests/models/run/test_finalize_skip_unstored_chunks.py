"""finalize_outputs visits only the chunks behind stored input, as blend_logits leaves them."""
import numpy as np
import pytest
import zarr
from vc_reader.reader import ZarrWriter, open_zarr

from vesuvius.models.run.finalize_outputs import finalize_logits


@pytest.fixture
def sparse_merged(tmp_path):
    # Blended logits with the whole volume's shape and only a few chunks stored, like a --bbox run.
    path = str(tmp_path / "merged.zarr")
    logits = np.zeros((2, 8, 8, 16), np.float16)
    logits[1, 0:4, 4:8, 8:12] = 3.0  # foreground; channel 0 stays zero, so only its channel-1 chunk is stored
    logits[0, 4:8, 0:4, 12:16] = 1.0  # background wins here
    logits[1, 5, 1, 13] = 2.0  # except at one voxel
    store = ZarrWriter.create(path, shape=logits.shape, chunks=(1, 4, 4, 4), dtype=np.float16,
                              cname="zstd", level=1, fill_value=0, skip_empty=True)
    store[...] = logits
    assert len(store.stored_chunks()) == 3
    return path, logits


@pytest.mark.parametrize("chunk_size", [None, (2, 8, 8)])
def test_finalize_matches_every_chunk_but_visits_only_stored(tmp_path, sparse_merged, chunk_size, capsys):
    merged, logits = sparse_merged
    final = str(tmp_path / "final.zarr")
    finalize_logits(merged, final, mode="binary", threshold=0.5, chunk_size=chunk_size, num_workers=1)

    output = open_zarr(final)
    expected = np.where(logits[1].astype(np.float32) - logits[0] > 0, 255, 0).astype(np.uint8)
    np.testing.assert_array_equal(output[...], expected[np.newaxis])

    # two of the three stored chunks share a spatial position; a (2, 8, 8) output chunk is
    # half an input chunk deep, so each stored position maps to two output chunks
    visited, total = (2, 16) if chunk_size is None else (4, 8)
    assert f"Processing data in {visited} of {total} chunks" in capsys.readouterr().out
    assert output.attrs["total_chunks"] == total
    assert output.attrs["empty_chunks_skipped"] == total - visited


@pytest.mark.parametrize("fill_value, version", [(np.nan, 2), (0, 3)])
def test_finalize_reads_every_chunk_when_listing_cannot_stand_in(tmp_path, fill_value, version, capsys):
    # A NaN fill is not skipped as empty, so every position is still finalized; a v3 store may be
    # sharded, where listing is not names only. zarr-python writes the inputs: vc_reader cannot
    # create a NaN fill
    merged = str(tmp_path / "merged.zarr")
    logits = np.full((2, 8, 8, 16), fill_value, np.float16)
    logits[0, 0:4, 4:8, 8:12] = 1.0
    logits[1, 0:4, 4:8, 8:12] = 3.0
    store = zarr.create_array(merged, shape=logits.shape, chunks=(1, 4, 4, 4), dtype=np.float16,
                              fill_value=fill_value, zarr_format=version, config={"write_empty_chunks": False})
    store[...] = logits
    assert len(ZarrWriter.open(merged).stored_chunks()) == 2
    final = str(tmp_path / "final.zarr")
    finalize_logits(merged, final, mode="binary", threshold=0.5, num_workers=1)

    output = open_zarr(final)
    expected = np.where(logits[1].astype(np.float32) - logits[0] > 0, 255, 0).astype(np.uint8)
    np.testing.assert_array_equal(output[...], expected[np.newaxis])
    assert "Processing data in 16 of 16 chunks" in capsys.readouterr().out
    # a NaN chunk is finalized (to background) rather than skipped as empty
    assert output.attrs["empty_chunks_skipped"] == (0 if np.isnan(fill_value) else 15)
