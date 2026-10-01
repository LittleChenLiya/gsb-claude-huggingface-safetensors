import os
from typing import Any, Iterable, Optional, Union

from ._safetensors_rust import (  # noqa: F401
    PrefetchLoader,
    PySafeSlice,
    SafetensorError,
    TensorMeta,
    TensorSpec,
    __version__,
    deserialize,
    safe_open as _safe_open_rust,
    _safe_open_handle,
    serialize,
    serialize_file,
)

__all__ = [
    "PrefetchLoader",
    "PySafeSlice",
    "SafetensorError",
    "TensorMeta",
    "TensorSpec",
    "safe_open",
    "__version__",
    "deserialize",
    "serialize",
    "serialize_file",
]


class safe_open:
    """
    Opens a safetensors file lazily and returns tensors as asked.

    This is a thin Python wrapper around the native handle. In addition to the
    per-tensor loading API it adds a *requested-keys / missing-keys list* mode:
    :meth:`missing_keys` reports every requested name the file lacks in one
    call, and :meth:`get_tensors` can skip the missing names instead of
    raising. Both only consult the file header, so no tensor bytes are read to
    discover which keys are missing.

    Args:
        filename (`str`, or `os.PathLike`):
            The filename to open
        framework (`str`):
            The framework you want your tensors in. Supported values:
            `pt`, `tf`, `flax`, `numpy`.
        device (`str`, defaults to `"cpu"`):
            The device on which you want the tensors.
        backend (`str`, *keyword-only*, defaults to `"mmap"`):
            Storage backend used to serve tensor bytes. `"mmap"` (the default)
            memory-maps the file; `"pread"` reads tensor bytes with `pread(2)`.

    Example:
    ```python
    from safetensors import safe_open

    with safe_open("model.safetensors", framework="pt") as f:
        tensor = f.get_tensor("embedding")
    ```
    """

    def __init__(
        self,
        filename: Union[str, "os.PathLike[str]"],
        framework: str,
        device: Optional[Union[str, int]] = None,
        *,
        backend: str = "mmap",
    ):
        if device is None:
            self._handle = _safe_open_rust(filename, framework, backend=backend)
        else:
            self._handle = _safe_open_rust(filename, framework, device, backend=backend)

    # -- requested-keys / missing-keys mode --------------------------------

    def missing_keys(self, names: Iterable[str]) -> list:
        """
        Return the requested names the file does not contain, in one pass.

        Only the header (already parsed when the handle was opened) is
        consulted: this never reads tensor bytes, even for a long request
        list. Order and uniqueness follow `names`: the result is a list, so a
        name repeated in `names` is repeated in the result.

        Args:
            names (`Iterable[str]`):
                The tensor names you are looking for.

        Returns:
            (`List[str]`):
                The names from `names` that are absent from the file.

        Example:
        ```python
        from safetensors import safe_open

        with safe_open("model.safetensors", framework="pt") as f:
            missing = f.missing_keys(["embedding", "nonexistent"])
        # missing == ["nonexistent"]
        ```
        """
        available = frozenset(self._handle.keys())
        return [name for name in names if name not in available]

    def get_tensors(
        self,
        names: Optional[Iterable[str]] = None,
        *,
        strict: bool = True,
    ):
        """
        Return tensors from the file, optionally restricted to `names`.

        Args:
            names (`Iterable[str]`, *optional*):
                Names of the tensors to return. `None` (the default) returns
                every tensor in the file, like the native handle.
            strict (`bool`, defaults to `True`):
                Kept for backward compatibility: when `True`, a name missing
                from the file raises [`SafetensorError`] on that key, exactly
                as :meth:`get_tensor` would. When `False`, missing names are
                collected and returned together with the tensors instead of
                raising.

        Returns:
            (`Dict[str, Tensor]` or `(Dict[str, Tensor], List[str])`):
                With `strict=True`, a dict of the requested tensors. With
                `strict=False`, a `(tensors, missing_keys)` pair; only the
                tensors that exist have their bytes read.

        Example:
        ```python
        from safetensors import safe_open

        with safe_open("model.safetensors", framework="pt") as f:
            tensors, missing = f.get_tensors(["embedding", "head"], strict=False)
        ```
        """
        if names is None:
            # Native bulk path: may use framework/device fast paths (e.g. MPS),
            # and stays byte-for-byte compatible with the old behavior.
            return self._handle.get_tensors()

        requested = list(names)
        if strict:
            return {name: self._handle.get_tensor(name) for name in requested}

        # Header-only membership check first: listing missing keys must not
        # touch tensor bytes, so we never call get_tensor for an absent name.
        available = frozenset(self._handle.keys())
        missing = [name for name in requested if name not in available]
        seen = set()
        tensors = {}
        for name in requested:
            if name in available and name not in seen:
                seen.add(name)
                tensors[name] = self._handle.get_tensor(name)
        return tensors, missing

    # -- pass-through API --------------------------------------------------

    def metadata(self):
        return self._handle.metadata()

    def keys(self):
        return self._handle.keys()

    def offset_keys(self):
        return self._handle.offset_keys()

    def get_tensor_meta(self, name: str) -> TensorMeta:
        return self._handle.get_tensor_meta(name)

    def get_tensor(self, name: str) -> Any:
        return self._handle.get_tensor(name)

    def get_slice(self, name: str) -> PySafeSlice:
        return self._handle.get_slice(name)

    def prefetch(self, plan=None, *, device=None, threads: int = 8):
        return self._handle.prefetch(plan, device=device, threads=threads)

    def __enter__(self) -> "safe_open":
        self._handle.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._handle.__exit__(exc_type, exc_value, traceback)
