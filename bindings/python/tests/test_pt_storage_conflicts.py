"""Tests for the weight-saving key-name-conflict (shared storage) diagnostic.

Covers:
    * shared-storage groups are reported correctly (including duplicate aliases),
    * independent tensors (and disjoint slices) are *not* reported,
    * the diagnostic never mutates the input dict or the tensors,
    * the default save path still raises the original error with the same
      message and exposes the structured conflict groups,
    * files produced after resolving the conflicts contain the expected keys
      and values.
"""

import os
import tempfile
import unittest

import torch

from safetensors import safe_open
from safetensors.torch import (
    find_storage_conflicts,
    format_storage_conflicts,
    load_file,
    save_file,
    save_model,
    storage_ptr,
)


class SharedStorageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = self.a  # b.* aliases a.* (shared storage, duplicate alias)


class StorageConflictDiagnosticTestCase(unittest.TestCase):
    def _snapshot(self, tensors):
        return {k: (storage_ptr(v), v.dtype, v.shape, v.detach().clone()) for k, v in tensors.items()}

    def test_groups_shared_independent_and_duplicate_alias(self):
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        independent = torch.zeros((2, 2))
        tensors = {
            "base": base,  # whole storage
            "row": base[:1, :],  # overlapping slice of the same storage
            "tail": base[2:, :],  # overlaps with base
            "alias": base,  # exact duplicate alias of base
            "independent": independent,  # own storage, never a conflict
        }

        conflicts = find_storage_conflicts(tensors)

        # All overlapping names share one storage; the independent tensor is absent.
        self.assertEqual(
            sorted(frozenset(group) for group in conflicts),
            [frozenset({"base", "row", "tail", "alias"})],
        )
        # Output is deterministic and sorted.
        self.assertEqual(conflicts, [sorted(conflicts[0])])

        # A pure duplicate alias (same tensor registered twice) is a conflict too.
        aliased = {"a": independent, "b": independent}
        self.assertEqual(find_storage_conflicts(aliased), [["a", "b"]])

        # No conflicts at all for independent tensors.
        no_conflict = {"a": torch.zeros(2), "b": torch.ones(3)}
        self.assertEqual(find_storage_conflicts(no_conflict), [])

    def test_disjoint_slices_are_not_conflicts(self):
        base = torch.arange(8, dtype=torch.float32)
        tensors = {"left": base[:4], "right": base[4:]}
        # Same storage but the memory ranges do not overlap: no conflict group.
        self.assertEqual(find_storage_conflicts(tensors), [])

    def test_diagnostic_does_not_mutate_input(self):
        base = torch.arange(6, dtype=torch.float32)
        tensors = {"base": base, "slice": base[:3], "solo": torch.zeros(2)}
        keys_before = list(tensors.keys())
        snapshot = self._snapshot(tensors)
        id_before = {k: id(v) for k, v in tensors.items()}

        conflicts = find_storage_conflicts(tensors)
        self.assertEqual(conflicts, [["base", "slice"]])

        # Dict untouched (same keys, same exact tensor objects).
        self.assertEqual(list(tensors.keys()), keys_before)
        self.assertEqual({k: id(v) for k, v in tensors.items()}, id_before)
        for k, v in tensors.items():
            ptr, dtype, shape, data = snapshot[k]
            self.assertEqual(storage_ptr(v), ptr)
            self.assertEqual(v.dtype, dtype)
            self.assertEqual(tuple(v.shape), tuple(shape))
            torch.testing.assert_close(v, data)

    def test_format_storage_conflicts_lists_every_key(self):
        text = format_storage_conflicts([["base", "row"], ["w", "w_alias"]])
        self.assertIn("`base`", text)
        self.assertIn("`row`", text)
        self.assertIn("`w`", text)
        self.assertIn("`w_alias`", text)
        self.assertEqual(len(text.strip().splitlines()), 2)

    def test_default_save_still_raises_original_error(self):
        base = torch.arange(4, dtype=torch.float32)
        tensors = {"w": base, "w_alias": base, "solo": torch.ones(2)}
        keys_before = list(tensors.keys())
        ids_before = {k: id(v) for k, v in tensors.items()}
        values_before = {k: v.detach().clone() for k, v in tensors.items()}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            with self.assertRaises(RuntimeError) as ctx:
                save_file(tensors, path)
            self.assertFalse(os.path.exists(path))

        # The failed diagnostic must not have mutated the caller's dict/tensors.
        self.assertEqual(list(tensors.keys()), keys_before)
        self.assertEqual({k: id(v) for k, v in tensors.items()}, ids_before)
        for k, v in tensors.items():
            torch.testing.assert_close(v, values_before[k])

        message = str(ctx.exception)
        # Original error wording and remediation pointer are preserved.
        self.assertIn("Some tensors share memory", message)
        self.assertIn("duplicate memory on disk", message)
        self.assertIn("use `save_model`", message)
        self.assertIn(
            "https://huggingface.co/docs/safetensors/torch_shared_tensors",
            message,
        )
        # Both members of the conflict group appear; the independent key does not.
        self.assertIn("w_alias", message)
        self.assertNotIn("solo", message)
        # Structured, deterministic diagnostic is attached to the exception.
        self.assertEqual(ctx.exception.conflict_groups, [["w", "w_alias"]])

    def test_non_contiguous_failure_message_unchanged(self):
        model = torch.nn.Linear(4, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nc.safetensors")
            with self.assertRaises(ValueError) as ctx:
                save_file({"w": model.weight.t()}, path)
            self.assertIn("non contiguous tensor", str(ctx.exception))

    def test_file_output_after_resolving_conflicts(self):
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        tensors = {
            "base": base,
            "row": base[:1, :].clone(),  # clone breaks the storage sharing
            "alias": base.clone(),
            "independent": torch.zeros((2, 2)),
        }
        self.assertEqual(find_storage_conflicts(tensors), [])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "resolved.safetensors")
            save_file(tensors, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

            loaded = load_file(path)
            self.assertEqual(
                sorted(loaded.keys()),
                ["alias", "base", "independent", "row"],
            )
            torch.testing.assert_close(loaded["base"], base)
            torch.testing.assert_close(loaded["row"], base[:1, :])
            torch.testing.assert_close(loaded["alias"], base)
            torch.testing.assert_close(loaded["independent"], torch.zeros((2, 2)))

            with safe_open(path, framework="pt") as f:
                self.assertEqual(
                    sorted(f.keys()),
                    ["alias", "base", "independent", "row"],
                )

    def test_save_model_handles_duplicate_alias_groups(self):
        model = SharedStorageModel()
        conflicts = find_storage_conflicts(model.state_dict())
        self.assertEqual(
            sorted(frozenset(g) for g in conflicts),
            [
                frozenset({"a.bias", "b.bias"}),
                frozenset({"a.weight", "b.weight"}),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shared.safetensors")
            save_model(model, path)

            # save_model keeps one canonical key per conflict group and records
            # the dropped aliases in the metadata.
            with safe_open(path, framework="pt") as f:
                self.assertEqual(sorted(f.keys()), ["a.bias", "a.weight"])
                self.assertEqual(
                    f.metadata(),
                    {"b.bias": "a.bias", "b.weight": "a.weight"},
                )

            reloaded = load_file(path)
            state_dict = model.state_dict()
            torch.testing.assert_close(reloaded["a.weight"], state_dict["a.weight"])
            torch.testing.assert_close(reloaded["a.bias"], state_dict["a.bias"])


if __name__ == "__main__":
    unittest.main()
