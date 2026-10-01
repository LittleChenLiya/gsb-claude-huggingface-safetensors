"""Acceptance tests for the requested-keys / missing-keys list loading mode.

New mode (see ``safe_open.missing_keys`` and
``get_tensors(names, strict=False)``):

  * list *all* requested keys that are absent in one call, instead of failing
    on the first one (the old per-key error is kept for backward
    compatibility);
  * only the file header is consulted, so no tensor bytes are read to find
    the missing keys.

The tests below use a small local file containing tensors of distinct,
known sizes and compare three patterns:

  1. old mode, one ``get_tensor`` per requested key -> raises on the first
     missing key (compatibility);
  2. new mode, ``get_tensors(names, strict=False)`` -> tensors + the full
     missing list, reading only the requested tensors that exist;
  3. ``missing_keys(names)`` alone -> the full missing list with **zero**
     tensor bytes read.

With the ``"pread"`` backend every ``get_tensor`` is exactly one
``read_exact_at`` of that tensor's bytes (see ``Open::get_tensor_pread``), so
the number of bytes read for a request can be predicted exactly from the
header via ``get_tensor_meta``. On Linux we additionally check the kernel's
own accounting (``/proc/self/io`` rchar) against the prediction, closing the
handle (dropping its mmap) before counting so demand paging cannot pollute
it.
"""

import os
import struct
import sys
import tempfile
import unittest

import numpy as np

from safetensors import SafetensorError, safe_open

# Four tensors with distinct, easy-to-reason-about sizes (F32 = 4 bytes each).
TENSORS = {
    "alpha": np.arange(6, dtype=np.float32),  # 24 bytes
    "beta": np.arange(10, dtype=np.float32),  # 40 bytes
    "gamma": np.arange(4, dtype=np.float32),  # 16 bytes
    "delta": np.arange(8, dtype=np.float32),  # 32 bytes
}
REQUEST = ["alpha", "missing_one", "gamma", "missing_two"]
PRESENT_REQUESTED = ["alpha", "gamma"]
MISSING_REQUESTED = ["missing_one", "missing_two"]

BYTES = {name: int(np.prod(t.shape)) * t.dtype.itemsize for name, t in TENSORS.items()}
EXPECTED_PRESENT_BYTES = sum(BYTES[name] for name in PRESENT_REQUESTED)
EXPECTED_ALL_BYTES = sum(BYTES.values())


def _tensor_nbytes(handle, name):
    start, end = handle.get_tensor_meta(name).data_offsets
    return end - start


def _rchar():
    """Bytes read via read()-family syscalls so far, per /proc/self/io (Linux)."""
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("rchar:"):
                return int(line.split()[1])
    raise RuntimeError("rchar missing from /proc/self/io")


HAS_PROC_IO = sys.platform.startswith("linux")


class MissingKeysTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "tiny.safetensors")
        from safetensors.numpy import save_file

        save_file(TENSORS, self.path, metadata={"fixture": "missing-keys"})

    def tearDown(self):
        self.tempdir.cleanup()

    def _meta_bytes(self, names):
        # Header-derived prediction of what strict per-key loading *would*
        # read for the present keys (one read_exact_at per tensor).
        with safe_open(self.path, framework="np", backend="pread") as f:
            return sum(_tensor_nbytes(f, name) for name in names)

    # ------------------------------------------------------------------
    # Baseline: header metadata agrees with the fixture sizes.
    def test_header_byte_sizes_match_fixture(self):
        with safe_open(self.path, framework="np", backend="pread") as f:
            self.assertEqual(set(f.keys()), set(TENSORS))
            for name in TENSORS:
                self.assertEqual(_tensor_nbytes(f, name), BYTES[name])
        self.assertEqual(self._meta_bytes(PRESENT_REQUESTED), EXPECTED_PRESENT_BYTES)

    # ------------------------------------------------------------------
    # Old mode: per-key error, backward compatible, reading one tensor at a
    # time up to the first missing key.
    def test_old_mode_raises_on_first_missing_key(self):
        with safe_open(self.path, framework="np", backend="pread") as f:
            # Existing keys load normally.
            np.testing.assert_array_equal(f.get_tensor("alpha"), TENSORS["alpha"])
            # The first missing key raises the same per-key error as before.
            with self.assertRaisesRegex(
                SafetensorError, "does not contain tensor missing_one"
            ):
                f.get_tensor("missing_one")
            # Failure is per key: an unrelated key still works afterwards.
            np.testing.assert_array_equal(f.get_tensor("gamma"), TENSORS["gamma"])
            with self.assertRaisesRegex(
                SafetensorError, "does not contain tensor missing_two"
            ):
                f.get_tensor("missing_two")

    def test_strict_get_tensors_matches_old_per_key_behavior(self):
        with safe_open(self.path, framework="np", backend="pread") as f:
            with self.assertRaisesRegex(SafetensorError, "missing_one"):
                f.get_tensors(REQUEST, strict=True)
            # The default is strict too.
            with self.assertRaisesRegex(SafetensorError, "missing_one"):
                f.get_tensors(REQUEST)
            # No names: full bulk load, unchanged from the native handle.
            all_tensors = f.get_tensors()
        self.assertEqual(set(all_tensors), set(TENSORS))

    # ------------------------------------------------------------------
    # New mode: one call returns *every* missing key plus present tensors.
    def test_non_strict_get_tensors_lists_all_missing_keys(self):
        with safe_open(self.path, framework="np", backend="pread") as f:
            tensors, missing = f.get_tensors(REQUEST, strict=False)

        self.assertEqual(missing, MISSING_REQUESTED)
        self.assertEqual(set(tensors), set(PRESENT_REQUESTED))
        for name in PRESENT_REQUESTED:
            np.testing.assert_array_equal(tensors[name], TENSORS[name])

    def test_missing_keys_lists_all_without_loading(self):
        with safe_open(self.path, framework="np", backend="pread") as f:
            self.assertEqual(f.missing_keys(REQUEST), MISSING_REQUESTED)
            # Empty and all-present requests.
            self.assertEqual(f.missing_keys([]), [])
            self.assertEqual(f.missing_keys(["alpha", "gamma"]), [])
            # Ordering and duplicates follow the request.
            self.assertEqual(
                f.missing_keys(["missing_two", "alpha", "missing_one", "missing_two"]),
                ["missing_two", "missing_one", "missing_two"],
            )
            # Requesting only missing keys returns them without any tensors.
            tensors, missing = f.get_tensors(MISSING_REQUESTED, strict=False)
            self.assertEqual(missing, MISSING_REQUESTED)
            self.assertEqual(tensors, {})

    # ------------------------------------------------------------------
    # Read accounting via the kernel: finding missing keys must not read
    # tensor bytes, and the list mode reads only the requested present tensors
    # (not all keys, not the missing ones). Uses the pread backend (one
    # read_exact_at per tensor) and /proc/self/io rchar. Each block closes its
    # own handle (dropping the mmap) before measuring, so demand faults on the
    # mapping cannot pollute the counter.
    @unittest.skipUnless(HAS_PROC_IO, "/proc/self/io is only available on Linux")
    def test_read_accounting_rchar_linux(self):
        with safe_open(self.path, framework="np", backend="pread") as f:
            before = _rchar()
            missing = f.missing_keys(REQUEST)
            after = _rchar()
        self.assertEqual(missing, MISSING_REQUESTED)
        self.assertEqual(after - before, 0, "missing_keys read tensor bytes")

        with safe_open(self.path, framework="np", backend="pread") as f:
            before = _rchar()
            tensors, missing = f.get_tensors(REQUEST, strict=False)
            after = _rchar()
        self.assertEqual(missing, MISSING_REQUESTED)
        self.assertEqual(set(tensors), set(PRESENT_REQUESTED))
        self.assertEqual(
            after - before,
            EXPECTED_PRESENT_BYTES,
            "list mode must read only the present requested tensors",
        )
        self.assertLess(EXPECTED_PRESENT_BYTES, EXPECTED_ALL_BYTES)

        # Old mode comparison: walk the request per key; it reads only the
        # first present tensor, then dies at the first missing key.
        with safe_open(self.path, framework="np", backend="pread") as f:
            before = _rchar()
            read_bytes = 0
            for name in REQUEST:
                try:
                    f.get_tensor(name)
                    read_bytes += BYTES[name]
                except SafetensorError:
                    break
            after = _rchar()
        self.assertEqual(read_bytes, BYTES["alpha"])
        self.assertEqual(after - before, BYTES["alpha"])

        # Reference: a full get_tensors() reads every tensor's bytes.
        with safe_open(self.path, framework="np", backend="pread") as f:
            before = _rchar()
            f.get_tensors()
            after = _rchar()
        self.assertEqual(after - before, EXPECTED_ALL_BYTES)

    # Cross-platform accounting via the header: one pread per present tensor,
    # sized by data_offsets. Missing tensors have no bytes to read and never
    # reach get_tensor.
    def test_read_accounting_predicted_from_header(self):
        with safe_open(self.path, framework="np", backend="pread") as f:
            present = [n for n in REQUEST if n not in set(f.missing_keys(REQUEST))]
            predicted = sum(_tensor_nbytes(f, n) for n in present)
            tensors, missing = f.get_tensors(REQUEST, strict=False)
        self.assertEqual(present, PRESENT_REQUESTED)
        self.assertEqual(predicted, EXPECTED_PRESENT_BYTES)
        self.assertEqual(missing, MISSING_REQUESTED)
        self.assertEqual(set(tensors), set(PRESENT_REQUESTED))

        # A request consisting solely of missing keys reads nothing.
        with safe_open(self.path, framework="np", backend="pread") as f:
            tensors, missing = f.get_tensors(MISSING_REQUESTED, strict=False)
        self.assertEqual(tensors, {})
        self.assertEqual(missing, MISSING_REQUESTED)

    # Membership is header-only, so it also works with the mmap default.
    def test_missing_keys_mmap_backend(self):
        with safe_open(self.path, framework="np") as f:
            self.assertEqual(f.missing_keys(REQUEST), MISSING_REQUESTED)
            tensors, missing = f.get_tensors(REQUEST, strict=False)
        self.assertEqual(missing, MISSING_REQUESTED)
        self.assertEqual(set(tensors), set(PRESENT_REQUESTED))
        for name in PRESENT_REQUESTED:
            np.testing.assert_array_equal(tensors[name], TENSORS[name])

    def test_context_manager_and_delegated_api_still_work(self):
        with safe_open(self.path, framework="np", backend="pread") as f:
            self.assertEqual(f.metadata(), {"fixture": "missing-keys"})
            self.assertEqual(set(f.offset_keys()), set(TENSORS))
            meta = f.get_tensor_meta("alpha")
            self.assertEqual(meta.dtype, "F32")
            self.assertEqual(meta.shape, [6])
            part = f.get_slice("beta")[2:5]
            np.testing.assert_array_equal(part, TENSORS["beta"][2:5])

    def test_absent_tensor_bytes_are_not_required(self):
        # The file holds one small, real tensor and two large ones whose data
        # clusters were never allocated (logically present, sparse/zero on
        # disk). Asking for the small tensor together with nonexistent names
        # must succeed without ever touching the unrequested large clusters,
        # and listing the missing names must not read any tensor bytes at all.
        big = 10 * 1024 * 1024  # 10 MiB per large tensor, never written
        small = b"abc"
        # offsets: small first (3 bytes), then big1/big2 over the hole
        header = (
            b'{"small":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},'
            b'"big1":{"dtype":"U8","shape":[%d],"data_offsets":[3,%d]},'
            b'"big2":{"dtype":"U8","shape":[%d],"data_offsets":[%d,%d]}}'
            % (big, 3 + big, big, 3 + big, 3 + 2 * big)
        )
        sparse_path = os.path.join(self.tempdir.name, "sparse.safetensors")
        with open(sparse_path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(small)
            fh.truncate(8 + len(header) + 3 + 2 * big)  # holes beyond `small`
        self.assertEqual(os.path.getsize(sparse_path), 8 + len(header) + 3 + 2 * big)

        with safe_open(sparse_path, framework="np", backend="pread") as f:
            # Pure header listing: big1 exists, the two nope names do not.
            self.assertEqual(
                f.missing_keys(["big1", "nope1", "nope2"]),
                ["nope1", "nope2"],
            )
            # Read only `small`; the big clusters stay unrequested and untouched.
            tensors, missing = f.get_tensors(["small", "nope1", "nope2"], strict=False)
            self.assertEqual(missing, ["nope1", "nope2"])
            self.assertEqual(set(tensors), {"small"})
            np.testing.assert_array_equal(
                tensors["small"], np.frombuffer(small, dtype=np.uint8)
            )
            # Only-missing requests read nothing.
            tensors, missing = f.get_tensors(["nope1", "nope2"], strict=False)
            self.assertEqual(missing, ["nope1", "nope2"])
            self.assertEqual(tensors, {})


if __name__ == "__main__":
    unittest.main()
