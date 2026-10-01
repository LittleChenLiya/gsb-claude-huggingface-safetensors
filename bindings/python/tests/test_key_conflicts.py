import os
import tempfile
import unittest
from typing import Dict

import torch

from safetensors import safe_open
from safetensors.torch import (
    diagnose_key_conflicts,
    load_file,
    save_file,
    save_model,
)


class _StateDictModule(torch.nn.Module):
    """Minimal module exposing a fixed state dict (no parameter registration)."""

    def __init__(self, tensors: Dict[str, torch.Tensor]):
        super().__init__()
        object.__setattr__(self, "_tensors", dict(tensors))

    def state_dict(self, *args, **kwargs):
        return dict(self._tensors)


class KeyConflictDiagnosisTestCase(unittest.TestCase):
    def test_diagnose_shared_storage_independent_and_aliases(self):
        # Shared storage: B is a view over A's storage.
        A = torch.zeros((4, 4))
        B = A[:1, :]
        # Independent tensors: separate storages, no conflict.
        C = torch.zeros((3, 3))
        D = torch.ones((2, 2))
        # Duplicate aliases: E and F are the exact same tensor object.
        E = torch.arange(5).float()
        F = E

        state_dict = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}

        # Snapshot everything before diagnosis (it must not mutate anything).
        snapshot = {
            k: (v.data_ptr(), v.untyped_storage().data_ptr(), v.nelement(), v.dtype)
            for k, v in state_dict.items()
        }
        values_before = {k: v.clone() for k, v in state_dict.items()}

        conflicts = diagnose_key_conflicts(state_dict)

        # Grouping: A<->B share storage; E<->F are duplicate aliases.
        normalized = sorted(sorted(group) for group in conflicts)
        self.assertEqual(normalized, [["A", "B"], ["E", "F"]])

        # Independent tensors are not reported as conflicts.
        conflicting_keys = {k for group in conflicts for k in group}
        self.assertNotIn("C", conflicting_keys)
        self.assertNotIn("D", conflicting_keys)

        # The diagnosis must not modify the original tensors (nor the dict).
        self.assertEqual(set(state_dict.keys()), {"A", "B", "C", "D", "E", "F"})
        for k, v in state_dict.items():
            data_ptr, storage_ptr, nelem, dtype = snapshot[k]
            self.assertEqual(v.data_ptr(), data_ptr, f"data_ptr changed for {k}")
            self.assertEqual(
                v.untyped_storage().data_ptr(), storage_ptr, f"storage changed for {k}"
            )
            self.assertEqual(v.nelement(), nelem)
            self.assertEqual(v.dtype, dtype)
            torch.testing.assert_close(v, values_before[k], msg=f"values changed for {k}")
        # Duplicate aliases still point to the very same object.
        self.assertIs(state_dict["F"], state_dict["E"])

    def test_non_overlapping_views_are_not_conflicts(self):
        A = torch.zeros((4,))
        B = A[:2]
        C = A[2:]
        # Same storage but no overlap -> no on-disk collision.
        self.assertEqual(diagnose_key_conflicts({"B": B, "C": C}), [])

    def test_clean_state_dict_has_no_conflicts(self):
        self.assertEqual(
            diagnose_key_conflicts({"C": torch.zeros((3, 3)), "D": torch.ones((2, 2))}),
            [],
        )

    def test_default_save_still_gives_original_error(self):
        A = torch.zeros((4, 4))
        B = A[:1, :]
        state_dict = {"A": A, "B": B}
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "conflict.safetensors")
            with self.assertRaises(RuntimeError) as ctx:
                save_file(state_dict, filename)
            # The historical default save failure message is preserved.
            self.assertIn("Some tensors share memory", str(ctx.exception))
            self.assertIn("save_model", str(ctx.exception))
            self.assertFalse(os.path.exists(filename))

    def test_save_model_resolves_conflicts_and_writes_file(self):
        A = torch.zeros((4, 4))
        B = A[:1, :]
        C = torch.zeros((3, 3))
        D = torch.ones((2, 2))
        E = torch.arange(5).float()
        F = E
        state_dict = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "model.safetensors")
            save_model(_StateDictModule(state_dict), filename)

            with safe_open(filename, framework="pt") as f:
                keys = list(f.keys())
                metadata = dict(f.metadata())
                tensors = {k: f.get_tensor(k) for k in keys}

            # One representative per conflict group is kept; independent tensors kept.
            self.assertEqual(sorted(keys), ["A", "C", "D", "E"])
            # Dropped aliases are recorded in metadata.
            self.assertEqual(metadata, {"B": "A", "F": "E"})

            # File contents match the original values (round-trip).
            torch.testing.assert_close(tensors["A"], A)
            torch.testing.assert_close(tensors["C"], C)
            torch.testing.assert_close(tensors["D"], D)
            torch.testing.assert_close(tensors["E"], E)

            # load_file round-trips as well.
            reloaded = load_file(filename)
            torch.testing.assert_close(reloaded["A"], A)
            torch.testing.assert_close(reloaded["E"], E)

        # The original state dict is untouched after saving.
        self.assertEqual(set(state_dict.keys()), {"A", "B", "C", "D", "E", "F"})
        self.assertIs(state_dict["F"], state_dict["E"])

    def test_save_model_without_conflicts(self):
        state_dict = {"C": torch.zeros((3, 3)), "D": torch.ones((2, 2))}
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "clean.safetensors")
            save_model(_StateDictModule(state_dict), filename)
            with safe_open(filename, framework="pt") as f:
                self.assertEqual(sorted(f.keys()), ["C", "D"])
                self.assertEqual(dict(f.metadata() or {}), {})


if __name__ == "__main__":
    unittest.main()
