"""Minimal, repeatable demo for the weight-saving key-name-conflict diagnostic.

It builds a state dict containing:

    * tensors that share storage with each other (overlapping views + a
      duplicate alias of the same tensor),
    * a fully independent tensor,
    * two disjoint slices of the same storage (not a conflict).

It then:

    1. prints the diagnosed conflict groups (without touching the tensors),
    2. shows that a default `save_file` still fails with the original error and
       does not produce a file,
    3. resolves the conflicts by detaching the aliases (`.clone()`) and saves a
       real safetensors file, verifying its keys and values.

Run it from the repository root with:

    python bindings/python/examples/storage_conflicts_demo.py
"""

import os
import tempfile

import torch

from safetensors.torch import (
    find_storage_conflicts,
    format_storage_conflicts,
    load_file,
    save_file,
)


def main():
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    solo = torch.zeros((2, 2))
    # `buffer` itself is NOT saved; only two disjoint slices of it are.
    buffer = torch.arange(8, dtype=torch.float32)

    state_dict = {
        "base": base,  # whole storage
        "row": base[:1, :],  # overlapping view -> conflict with base
        "tail": base[2:, :],  # overlapping view -> conflict with base
        "alias": base,  # duplicate alias -> conflict with base
        "left": buffer[:4],  # disjoint slice of an unsaved storage
        "right": buffer[4:],  # disjoint slice of the same storage, no overlap
        "solo": solo,  # independent storage -> never reported
    }

    print("=" * 70)
    print("1. Diagnosing key-name conflicts (read-only, tensors not modified)")
    print("=" * 70)
    conflicts = find_storage_conflicts(state_dict)
    print(format_storage_conflicts(conflicts))
    assert conflicts == [["alias", "base", "row", "tail"]], conflicts
    # The input dict is untouched: same keys, same storage pointers.
    assert list(state_dict.keys()) == ["base", "row", "tail", "alias", "left", "right", "solo"]
    assert state_dict["alias"].data_ptr() == state_dict["base"].data_ptr()
    # Disjoint slices still share a storage, but do not overlap -> no conflict.
    assert state_dict["left"].untyped_storage().data_ptr() == state_dict["right"].untyped_storage().data_ptr()
    assert not any("left" in group or "right" in group for group in conflicts)

    print()
    print("=" * 70)
    print("2. Default save still fails with the original error")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "broken.safetensors")
        try:
            save_file(state_dict, path)
        except RuntimeError as e:
            print(str(e).strip())
            print("Structured conflict groups:", e.conflict_groups)
            assert e.conflict_groups == [["alias", "base", "row", "tail"]]
            assert not os.path.exists(path), "no file should be written on failure"

    print()
    print("=" * 70)
    print("3. Resolve the conflicts and produce a valid file")
    print("=" * 70)
    resolved = {
        "base": base,
        "row": base[:1, :].clone(),
        "tail": base[2:, :].clone(),
        "alias": base.clone(),
        # Disjoint, non-overlapping slices are already safe to keep.
        "left": buffer[:4],
        "right": buffer[4:],
        "solo": solo,
    }
    assert find_storage_conflicts(resolved) == []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ok.safetensors")
        save_file(resolved, path)
        print(f"Saved {os.path.getsize(path)} bytes to {path!r}")
        loaded = load_file(path)
        print("File keys:", sorted(loaded.keys()))
        assert sorted(loaded.keys()) == ["alias", "base", "left", "right", "row", "solo", "tail"]
        torch.testing.assert_close(loaded["base"], base)
        torch.testing.assert_close(loaded["row"], base[:1, :])
        torch.testing.assert_close(loaded["tail"], base[2:, :])
        torch.testing.assert_close(loaded["alias"], base)
        torch.testing.assert_close(loaded["solo"], solo)

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
