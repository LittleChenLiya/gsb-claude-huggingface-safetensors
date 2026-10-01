"""
Tests for the "requested keys -> missing keys list" loading mode.

The new :meth:`safe_open.missing_keys` returns *all* requested tensor names
that are absent from the file in one call, checking only the in-memory header
(no tensor bytes are read). The legacy per-key :meth:`safe_open.get_tensor`
error is unchanged.

Read counts are measured with a small native fixture DLL
(``tests/fixtures/readhook``) that IAT-hooks ``ntdll!NtReadFile`` in the rust
extension, because OS I/O counters are virtualized to zero in some sandboxes.
"""

import ctypes
import os
import subprocess
import unittest

import numpy as np

from safetensors import SafetensorError, _safe_open_handle, safe_open
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "fixtures", "readhook")
HOOK_DLL = os.path.join(FIXTURE_DIR, "readhook.dll")
HOOK_SRC = os.path.join(FIXTURE_DIR, "readhook.c")
# A build script that works from a plain shell (it locates cl.exe itself).
HOOK_BUILD = os.path.join(FIXTURE_DIR, "build.bat")


def _load_readhook():
    """Load the readhook DLL, building it from source if necessary.

    Returns a configured :class:`ctypes.CDLL` or ``None`` if the fixture
    cannot be built/loaded (the read-count tests then skip).
    """
    if not os.path.exists(HOOK_DLL):
        try:
            if os.path.exists(HOOK_BUILD):
                subprocess.run(
                    ["cmd", "//c", HOOK_BUILD],
                    cwd=FIXTURE_DIR,
                    check=True,
                    capture_output=True,
                )
            else:
                subprocess.run(
                    ["cl", "/LD", "/O2", HOOK_SRC],
                    cwd=FIXTURE_DIR,
                    check=True,
                    capture_output=True,
                )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
    if not os.path.exists(HOOK_DLL):
        return None
    try:
        hook = ctypes.CDLL(HOOK_DLL)
    except OSError:
        return None
    hook.readhook_install.argtypes = [ctypes.c_wchar_p]
    hook.readhook_install.restype = ctypes.c_longlong
    hook.readhook_reset.argtypes = []
    hook.readhook_reset.restype = None
    hook.readhook_get_ops.argtypes = []
    hook.readhook_get_ops.restype = ctypes.c_longlong
    hook.readhook_get_bytes.argtypes = []
    hook.readhook_get_bytes.restype = ctypes.c_longlong
    # The extension is already loaded (safetensors was imported above); make
    # sure the hook is attached even if DllMain ran before it loaded.
    hook.readhook_install("_safetensors_rust.pyd")
    return hook


class MissingKeysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hook = _load_readhook()
        # A small local file: two tiny tensors plus a 5 MB one so reads are
        # measurable.
        cls.tensors = {
            "a": np.arange(4, dtype=np.float32),
            "b": np.ones((3,), dtype=np.int32),
            "big": np.zeros((5_000_000,), dtype=np.uint8),
        }
        cls.filename = os.path.join(HERE, "missing_keys_fixture.safetensors")
        save_file(cls.tensors, cls.filename)

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.filename)
        except OSError:
            pass

    def test_missing_keys_lists_all_missing_at_once(self):
        requested = ["a", "ghost1", "b", "ghost2"]
        with safe_open(self.filename, framework="np", backend="pread") as f:
            missing = f.missing_keys(requested)
        # All missing keys reported in a single call, in request order.
        self.assertEqual(missing, ["ghost1", "ghost2"])

    def test_missing_keys_deduplicates_and_preserves_order(self):
        with safe_open(self.filename, framework="np") as f:
            self.assertEqual(f.missing_keys([]), [])
            self.assertEqual(f.missing_keys(["a", "b"]), [])
            self.assertEqual(
                f.missing_keys(["ghost2", "a", "ghost1", "a", "ghost2"]),
                ["ghost2", "ghost1"],
            )

    def test_old_mode_stops_at_first_missing_key(self):
        # Legacy behavior: iterating get_tensor raises on the first missing
        # key, so the caller only ever learns about that one.
        requested = ["a", "ghost1", "b", "ghost2"]
        with safe_open(self.filename, framework="np", backend="pread") as f:
            loaded = {}
            with self.assertRaises(SafetensorError) as ctx:
                for name in requested:
                    loaded[name] = f.get_tensor(name)
            self.assertIn("ghost1", str(ctx.exception))
            self.assertNotIn("ghost2", str(ctx.exception))
            # It had already read "a" before failing on "ghost1".
            self.assertIn("a", loaded)
            self.assertNotIn("ghost1", loaded)

    def test_new_mode_reads_zero_tensor_bytes(self):
        if self.hook is None:
            self.skipTest("readhook DLL fixture unavailable")
        requested = ["a", "ghost1", "b", "ghost2"]
        with safe_open(self.filename, framework="np", backend="pread") as f:
            self.hook.readhook_reset()
            missing = f.missing_keys(requested)
            ops = self.hook.readhook_get_ops()
            nbytes = self.hook.readhook_get_bytes()
        self.assertEqual(missing, ["ghost1", "ghost2"])
        # Header-only: not a single tensor byte was read.
        self.assertEqual(ops, 0)
        self.assertEqual(nbytes, 0)

    def test_get_tensor_reads_exactly_the_tensor_bytes(self):
        if self.hook is None:
            self.skipTest("readhook DLL fixture unavailable")
        with safe_open(self.filename, framework="np", backend="pread") as f:
            self.hook.readhook_reset()
            t = f.get_tensor("big")
            ops = self.hook.readhook_get_ops()
            nbytes = self.hook.readhook_get_bytes()
        self.assertEqual(ops, 1)
        self.assertEqual(nbytes, 5_000_000)
        np.testing.assert_array_equal(t, self.tensors["big"])

    def test_present_tensors_still_load_after_missing_check(self):
        requested = ["a", "ghost1", "b", "ghost2"]
        with safe_open(self.filename, framework="np", backend="pread") as f:
            missing = f.missing_keys(requested)
            self.assertEqual(missing, ["ghost1", "ghost2"])
            loaded = {
                name: f.get_tensor(name) for name in requested if name not in missing
            }
        np.testing.assert_array_equal(loaded["a"], self.tensors["a"])
        np.testing.assert_array_equal(loaded["b"], self.tensors["b"])

    def test_get_slice_missing_key_still_raises(self):
        # Default per-key error behavior is unchanged for get_slice too.
        with safe_open(self.filename, framework="np") as f:
            with self.assertRaises(SafetensorError):
                f.get_slice("ghost")

    def test_safe_open_handle_missing_keys(self):
        with open(self.filename, "rb") as fh:
            with _safe_open_handle(fh, framework="np", backend="pread") as f:
                self.assertEqual(f.missing_keys(["a", "ghost"]), ["ghost"])


if __name__ == "__main__":
    unittest.main()
