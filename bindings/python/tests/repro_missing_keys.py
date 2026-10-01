"""Local reproduction for the requested-keys / missing-keys list mode.

Run (from bindings/python/, with the package sources importable)::

    # build the extension once, then expose the local py_src package:
    maturin build --release
    python -c "import zipfile,glob; w=zipfile.ZipFile(glob.glob('target/wheels/*.whl')[0]); \
        w.extract('safetensors/' + [n for n in w.namelist() if n.endswith(('.pyd','.so'))][0].split('/')[-1], 'py_src')"
    PYTHONPATH=py_src python tests/repro_missing_keys.py

The script writes a tiny local safetensors file (four tensors of distinct
sizes), then compares the old per-key mode with the new list mode and prints
how many tensor bytes each pattern reads. Bytes are counted two ways:

  * exactly from the header (`get_tensor_meta(...).data_offsets`) — every
    `get_tensor` with the "pread" backend is one read_exact_at of that size;
  * from the kernel when available (Linux /proc/self/io `rchar`).
"""

import os
import struct
import sys
import tempfile

import numpy as np

from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

TENSORS = {
    "alpha": np.arange(6, dtype=np.float32),  # 24 B
    "beta": np.arange(10, dtype=np.float32),  # 40 B
    "gamma": np.arange(4, dtype=np.float32),  # 16 B
    "delta": np.arange(8, dtype=np.float32),  # 32 B
}
REQUEST = ["alpha", "missing_one", "gamma", "missing_two"]


def tensor_nbytes(handle, name):
    start, end = handle.get_tensor_meta(name).data_offsets
    return end - start


def rchar():
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("rchar:"):
                return int(line.split()[1])
    return None


def main():
    have_proc_io = sys.platform.startswith("linux")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.safetensors")
        save_file(TENSORS, path)
        all_bytes = sum(t.nbytes for t in TENSORS.values())
        present = ["alpha", "gamma"]
        present_bytes = sum(TENSORS[name].nbytes for name in present)

        print(f"file: {path}")
        print(f"keys in file: {sorted(TENSORS)} (total tensor bytes: {all_bytes})")
        print(f"request: {REQUEST}")
        print()

        # --- Old mode: per-key error, backward compatible -----------------
        print("[old mode] one get_tensor() per requested key")
        with safe_open(path, framework="np", backend="pread") as f:
            old_read = 0
            before = rchar() if have_proc_io else None
            try:
                for name in REQUEST:
                    f.get_tensor(name)
                    old_read += tensor_nbytes(f, name)
                    print(f"  get_tensor({name!r}): ok ({tensor_nbytes(f, name)} B)")
            except SafetensorError as e:
                after = rchar() if have_proc_io else None
                print(f"  get_tensor({REQUEST[1]!r}): raised SafetensorError: {e}")
                print(
                    f"  -> stopped after first missing key; {old_read} B read"
                    + (f"; rchar delta={after - before} B" if have_proc_io else "")
                )
        print()

        # --- New mode: list every missing key in one call -----------------
        print("[new mode] missing_keys(names)")
        with safe_open(path, framework="np", backend="pread") as f:
            before = rchar() if have_proc_io else None
            missing = f.missing_keys(REQUEST)
            after = rchar() if have_proc_io else None
            print(f"  missing_keys({REQUEST}) -> {missing}")
            print(
                "  -> 0 tensor bytes read (header only)"
                + (f"; rchar delta={after - before} B" if have_proc_io else "")
            )
        print()

        print("[new mode] get_tensors(names, strict=False)")
        with safe_open(path, framework="np", backend="pread") as f:
            before = rchar() if have_proc_io else None
            tensors, missing = f.get_tensors(REQUEST, strict=False)
            after = rchar() if have_proc_io else None
            header_predicted = sum(tensor_nbytes(f, name) for name in tensors)
            print(f"  tensors: {sorted(tensors)}")
            print(f"  missing: {missing}")
            print(
                f"  -> {header_predicted} B read "
                f"(present requested only; full file would be {all_bytes} B)"
                + (f"; rchar delta={after - before} B" if have_proc_io else "")
            )
            for name in present:
                np.testing.assert_array_equal(tensors[name], TENSORS[name])
            print("  -> tensor contents verified")
        print()

        # --- Sparse-hole proof: bytes of unrequested tensors never touched -
        big = 10 * 1024 * 1024
        header = (
            b'{"small":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},'
            b'"big1":{"dtype":"U8","shape":[%d],"data_offsets":[3,%d]},'
            b'"big2":{"dtype":"U8","shape":[%d],"data_offsets":[%d,%d]}}'
            % (big, 3 + big, big, 3 + big, 3 + 2 * big)
        )
        sparse = os.path.join(d, "sparse.safetensors")
        with open(sparse, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(b"abc")
            fh.truncate(8 + len(header) + 3 + 2 * big)
        with safe_open(sparse, framework="np", backend="pread") as f:
            tensors, missing = f.get_tensors(["small", "nope"], strict=False)
        print(
            f"[sparse] 2x{big // (1024 * 1024)} MiB clusters unallocated; "
            f"requested small+nope -> tensors={sorted(tensors)}, missing={missing}"
        )

        # Assertions so the script doubles as a smoke check.
        assert old_read == TENSORS["alpha"].nbytes
        assert missing == ["nope"]
        assert present_bytes < all_bytes
        print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
