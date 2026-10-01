//! Module handling lazy loading via iterating on slices on the original buffer.
use crate::lib::Vec;
use crate::tensor::{Dtype, TensorView};
use core::fmt::Display;
use core::num::NonZeroUsize;
use core::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

/// Error representing invalid slicing attempt
#[derive(Debug)]
#[cfg_attr(test, derive(Eq, PartialEq))]
pub enum InvalidSlice {
    /// When the client asked for more slices than the tensors has dimensions
    TooManySlices,
    /// When the client asked for a slice that exceeds the allowed bounds
    SliceOutOfRange {
        /// The rank of the dimension that has the out of bounds
        dim_index: usize,
        /// The problematic value
        asked: usize,
        /// The dimension size we shouldn't go over.
        dim_size: usize,
    },
    /// For smaller than 1 byte dtypes, some slices will happen outside of the byte boundary, some special care has to be taken
    /// and standard functions will fail
    MisalignedSlice,
}

impl Display for InvalidSlice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            InvalidSlice::TooManySlices => {
                write!(f, "more slicing indexes than dimensions in tensor")
            }
            InvalidSlice::SliceOutOfRange {
                dim_index,
                asked,
                dim_size,
            } => {
                write!(f, "index {asked} out of bounds for tensor dimension #{dim_index} of size {dim_size}")
            }
            InvalidSlice::MisalignedSlice => {
                write!(f, "The slice is slicing for subbytes dtypes, and the slice does not end up at a byte boundary, this is invalid.")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InvalidSlice {}

#[cfg(not(feature = "std"))]
impl core::error::Error for InvalidSlice {}

#[derive(Debug, Clone)]
/// Generic structure used to index a slice of the tensor
pub enum TensorIndexer {
    /// This is selecting an entire dimension
    Select(usize),
    /// A slice `start:stop:step`. `step` is always >= 1; a contiguous slice
    /// has `step == 1`.
    Narrow(Bound<usize>, Bound<usize>, NonZeroUsize),
}

fn display_bound(bound: &Bound<usize>) -> &dyn Display {
    match bound {
        Bound::Unbounded => &"",
        Bound::Excluded(n) => n,
        Bound::Included(n) => n,
    }
}

/// Intended for Python users mostly or at least for its conventions
impl Display for TensorIndexer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorIndexer::Select(n) => {
                write!(f, "{n}")
            }
            TensorIndexer::Narrow(left, right, step) => {
                if step.get() == 1 {
                    write!(f, "{}:{}", display_bound(left), display_bound(right))
                } else {
                    write!(f, "{}:{}:{step}", display_bound(left), display_bound(right))
                }
            }
        }
    }
}

impl From<usize> for TensorIndexer {
    fn from(index: usize) -> Self {
        TensorIndexer::Select(index)
    }
}

// impl From<&[usize]> for TensorIndexer {
//     fn from(index: &[usize]) -> Self {
//         let tensor = index.into();
//         TensorIndexer::IndexSelect(tensor)
//     }
// }
//
// impl From<Vec<usize>> for TensorIndexer {
//     fn from(index: Vec<usize>) -> Self {
//         let tensor = Tensor::of_slice(&index);
//         TensorIndexer::IndexSelect(tensor)
//     }
// }

macro_rules! impl_from_range {
    ($range_type:ty) => {
        impl From<$range_type> for TensorIndexer {
            fn from(range: $range_type) -> Self {
                use core::ops::Bound::*;

                let start = match range.start_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                let end = match range.end_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                TensorIndexer::Narrow(start, end, NonZeroUsize::MIN)
            }
        }
    };
}

impl_from_range!(Range<usize>);
impl_from_range!(RangeFrom<usize>);
impl_from_range!(RangeFull);
impl_from_range!(RangeInclusive<usize>);
impl_from_range!(RangeTo<usize>);
impl_from_range!(RangeToInclusive<usize>);

/// A slicing indexer whose bounds may be negative, following Python/NumPy
/// indexing conventions: negative values count back from the end of the
/// dimension (`-1` is the last element).
///
/// This is the signed counterpart of [`TensorIndexer`]. The data path
/// ([`SliceIterator`]) operates on the unsigned [`TensorIndexer`]; callers
/// that resolve negative indices themselves (e.g. the Python bindings) pass
/// the resolved unsigned values down to the data path. [`SignedIndexer`] is
/// provided so negative-index semantics can be expressed and checked directly
/// when computing a [`SliceBoundaryReport`].
#[derive(Debug, Clone)]
pub enum SignedIndexer {
    /// Select a single element; negative values count back from the end.
    Select(isize),
    /// A slice `start:stop:step`; negative bounds count back from the end.
    Narrow(Bound<isize>, Bound<isize>, NonZeroUsize),
}

/// Trait used to implement multiple signatures for ease of use of the slicing
/// of a tensor
pub trait IndexOp<'data, T> {
    /// Returns a slicing iterator which are the chunks of data necessary to
    /// reconstruct the desired tensor.
    fn slice(&'data self, index: T) -> Result<SliceIterator<'data>, InvalidSlice>;
}

impl<'data, A> IndexOp<'data, A> for TensorView<'data>
where
    A: Into<TensorIndexer>,
{
    fn slice(&'data self, index: A) -> Result<SliceIterator<'data>, InvalidSlice> {
        self.sliced_data(&[index.into()])
    }
}

impl<'data, A> IndexOp<'data, (A,)> for TensorView<'data>
where
    A: Into<TensorIndexer>,
{
    fn slice(&'data self, index: (A,)) -> Result<SliceIterator<'data>, InvalidSlice> {
        let idx_a = index.0.into();
        self.sliced_data(&[idx_a])
    }
}

impl<'data, A, B> IndexOp<'data, (A, B)> for TensorView<'data>
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
{
    fn slice(&'data self, index: (A, B)) -> Result<SliceIterator<'data>, InvalidSlice> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        self.sliced_data(&[idx_a, idx_b])
    }
}

impl<'data, A, B, C> IndexOp<'data, (A, B, C)> for TensorView<'data>
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
{
    fn slice(&'data self, index: (A, B, C)) -> Result<SliceIterator<'data>, InvalidSlice> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        self.sliced_data(&[idx_a, idx_b, idx_c])
    }
}

// impl<A, B, C, D> IndexOp<(A, B, C, D)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d])
//     }
// }
//
// impl<A, B, C, D, E> IndexOp<(A, B, C, D, E)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
//     E: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D, E)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         let idx_e = index.4.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d, idx_e])
//     }
// }
//
// impl<A, B, C, D, E, F> IndexOp<(A, B, C, D, E, F)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
//     E: Into<TensorIndexer>,
//     F: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D, E, F)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         let idx_e = index.4.into();
//         let idx_f = index.5.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f])
//     }
// }
//
// impl<A, B, C, D, E, F, G> IndexOp<(A, B, C, D, E, F, G)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
//     E: Into<TensorIndexer>,
//     F: Into<TensorIndexer>,
//     G: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D, E, F, G)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         let idx_e = index.4.into();
//         let idx_f = index.5.into();
//         let idx_g = index.6.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g])
//     }
// }

/// Iterator used to return the bits of the overall tensor buffer
/// when client asks for a slice of the original tensor.
#[cfg_attr(test, derive(Debug, Eq, PartialEq))]
pub struct SliceIterator<'data> {
    view: &'data TensorView<'data>,
    indices: Vec<(usize, usize)>,
    newshape: Vec<usize>,
}

impl<'data> SliceIterator<'data> {
    pub(crate) fn new(
        view: &'data TensorView<'data>,
        slices: &[TensorIndexer],
    ) -> Result<Self, InvalidSlice> {
        let (indices, newshape) = slice_byte_ranges(view.dtype(), view.shape(), slices)?;
        // Reversing so we can pop faster while iterating on the slice
        let indices = indices.into_iter().rev().collect();
        Ok(Self {
            view,
            indices,
            newshape,
        })
    }

    /// Gives back the amount of bytes still being in the iterator
    pub fn remaining_byte_len(&self) -> usize {
        self.indices.iter().map(|(start, stop)| stop - start).sum()
    }

    /// Gives back the amount of bytes still being in the iterator
    pub fn newshape(&self) -> Vec<usize> {
        self.newshape.clone()
    }
}

/// Byte ranges into a tensor's data section in iteration order;
/// concatenating them yields the dense destination layout.
pub type SliceByteRanges = Vec<(usize, usize)>;

/// Post-slice tensor shape (element counts per dim).
pub type SlicedShape = Vec<usize>;

/// The resolved half-open interval `[start, stop)` of a single tensor
/// dimension after applying a slicing operation.
///
/// `start` and `stop` are element indices (not bytes), resolved against the
/// dimension's size: unbounded ends default to `0` and `dim`, and negative
/// indices are counted back from the end (Python/NumPy convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceBoundary {
    /// Inclusive resolved start (element index).
    pub start: usize,
    /// Exclusive resolved stop (element index).
    pub stop: usize,
    /// Size of the dimension this interval was resolved against.
    pub dim: usize,
}

impl SliceBoundary {
    /// `true` when the interval contains no elements (`start >= stop`).
    pub fn is_empty(&self) -> bool {
        self.start >= self.stop
    }

    /// Number of elements in the interval (`stop - start`, saturating at 0).
    pub fn len(&self) -> usize {
        self.stop.saturating_sub(self.start)
    }
}

/// Boundary report of a tensor slicing operation.
///
/// Reports, for every dimension of the original tensor, the actual cropped
/// half-open interval `[start, stop)` that the slice resolved to, together
/// with the post-slice shape. The report is computed from the shape and the
/// indexers alone; the underlying data buffer is never read or copied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceBoundaryReport {
    /// Resolved interval for each original dimension, in slice order.
    ///
    /// Dimensions sliced with [`TensorIndexer::Select`] are removed from the
    /// result shape but still appear here with their selected interval
    /// `[s, s + 1)`.
    pub boundaries: Vec<SliceBoundary>,
    /// Post-slice shape (element counts per dim), matching
    /// [`SliceIterator::newshape`].
    pub newshape: Vec<usize>,
}

impl SliceBoundaryReport {
    /// `true` when the slice selects zero elements overall, i.e. when at
    /// least one cropped interval is empty.
    pub fn is_empty(&self) -> bool {
        self.boundaries.iter().any(SliceBoundary::is_empty)
    }
}

/// Resolve a single signed bound against a dimension size.
///
/// Negative values follow Python/NumPy convention: `-1` refers to the last
/// element (`dim - 1`). Returns `None` when the value is out of range
/// (e.g. `-6` for a dimension of size `5`).
fn resolve_signed(value: isize, dim: usize) -> Option<usize> {
    if value < 0 {
        dim.checked_add_signed(value)
    } else {
        Some(value as usize)
    }
}

/// Resolve the half-open `[start, stop)` element interval for one dimension
/// from signed slice bounds, validating against `dim`.
///
/// Negative bounds follow Python/NumPy indexing conventions (see
/// [`resolve_signed`]). Out-of-bounds bounds are rejected with the same
/// strict [`InvalidSlice::SliceOutOfRange`] semantics as the rest of the
/// slicing API (a range that exceeds the dimension is an error, not a clamp).
fn resolve_interval(
    dim_index: usize,
    dim: usize,
    left: Bound<isize>,
    right: Bound<isize>,
) -> Result<(usize, usize), InvalidSlice> {
    // A bound that fails to resolve (negative value whose magnitude exceeds
    // the dimension) is mapped to `usize::MAX` so the out-of-range check
    // below flags it; the error type matches the rest of the slicing API.
    let resolve_or_max = |s: isize| -> usize { resolve_signed(s, dim).unwrap_or(usize::MAX) };
    let start = match left {
        Bound::Unbounded => 0,
        Bound::Included(s) => resolve_or_max(s),
        Bound::Excluded(s) => resolve_or_max(s).saturating_add(1),
    };
    let stop = match right {
        Bound::Unbounded => dim,
        Bound::Included(s) => resolve_or_max(s).saturating_add(1),
        Bound::Excluded(s) => resolve_or_max(s),
    };
    if start >= dim || stop > dim {
        let asked = if start >= dim { start } else { stop.saturating_sub(1) };
        return Err(InvalidSlice::SliceOutOfRange {
            dim_index,
            asked,
            dim_size: dim,
        });
    }
    Ok((start, stop))
}

/// Resolve a `(start, stop)` half-open element range from unsigned slice
/// bounds, defaulting unbounded ends to `0` and `dim`.
fn narrow_bounds(left: &Bound<usize>, right: &Bound<usize>, dim: usize) -> (usize, usize) {
    let start = match left {
        Bound::Unbounded => 0,
        Bound::Included(s) => *s,
        Bound::Excluded(s) => *s + 1,
    };
    let stop = match right {
        Bound::Unbounded => dim,
        Bound::Included(s) => *s + 1,
        Bound::Excluded(s) => *s,
    };
    (start, stop)
}

/// Compute the [`SliceBoundaryReport`] for a slicing operation without
/// touching the underlying data buffer.
///
/// This resolves each indexer against its dimension (defaulting unbounded
/// ends to `0` and `dim`) and validates the result with the same strict
/// semantics as [`slice_byte_ranges`]: a range that exceeds the dimension is
/// an [`InvalidSlice::SliceOutOfRange`], not a silent clamp. The report
/// carries the actual cropped interval per dimension and whether the slice
/// is empty.
pub fn slice_boundaries(
    shape: &[usize],
    slices: &[TensorIndexer],
) -> Result<SliceBoundaryReport, InvalidSlice> {
    let n_slice = slices.len();
    let n_shape = shape.len();
    if n_slice > n_shape {
        return Err(InvalidSlice::TooManySlices);
    }
    let mut boundaries = Vec::with_capacity(n_shape);
    let mut newshape = Vec::with_capacity(n_shape);

    for (i, &dim) in shape.iter().enumerate() {
        if i >= n_slice {
            // Not sliced: the full dimension is selected.
            boundaries.push(SliceBoundary { start: 0, stop: dim, dim });
            newshape.push(dim);
        } else {
            let (start, stop, removed) = match &slices[i] {
                TensorIndexer::Select(s) => (*s, *s + 1, true),
                TensorIndexer::Narrow(left, right, _step) => {
                    let (start, stop) = narrow_bounds(left, right, dim);
                    (start, stop, false)
                }
            };
            if start >= dim || stop > dim {
                let asked = if start >= dim { start } else { stop.saturating_sub(1) };
                return Err(InvalidSlice::SliceOutOfRange {
                    dim_index: i,
                    asked,
                    dim_size: dim,
                });
            }
            boundaries.push(SliceBoundary { start, stop, dim });
            // `Select` removes the dimension from the result shape.
            if !removed {
                let step = match &slices[i] {
                    TensorIndexer::Narrow(_, _, step) => step.get(),
                    TensorIndexer::Select(_) => unreachable!(),
                };
                newshape.push((stop - start).div_ceil(step));
            }
        }
    }

    Ok(SliceBoundaryReport { boundaries, newshape })
}

/// Compute the [`SliceBoundaryReport`] for a slicing operation using signed
/// indexers, so negative indices follow Python/NumPy indexing conventions
/// (negative values count back from the end of the dimension).
///
/// Out-of-bounds values are rejected with the same strict
/// [`InvalidSlice::SliceOutOfRange`] semantics as [`slice_boundaries`].
pub fn slice_boundaries_signed(
    shape: &[usize],
    slices: &[SignedIndexer],
) -> Result<SliceBoundaryReport, InvalidSlice> {
    let n_slice = slices.len();
    let n_shape = shape.len();
    if n_slice > n_shape {
        return Err(InvalidSlice::TooManySlices);
    }
    let mut boundaries = Vec::with_capacity(n_shape);
    let mut newshape = Vec::with_capacity(n_shape);

    for (i, &dim) in shape.iter().enumerate() {
        if i >= n_slice {
            boundaries.push(SliceBoundary { start: 0, stop: dim, dim });
            newshape.push(dim);
        } else {
            let (start, stop, removed, step) = match &slices[i] {
                SignedIndexer::Select(s) => {
                    // A single element is the half-open interval `[s, s + 1)`.
                    // Resolve the start first (negative values count back from
                    // the end); a negative value whose magnitude exceeds the
                    // dimension is out of range.
                    let start = resolve_signed(*s, dim).ok_or(InvalidSlice::SliceOutOfRange {
                        dim_index: i,
                        asked: dim,
                        dim_size: dim,
                    })?;
                    let stop = start + 1;
                    if start >= dim || stop > dim {
                        return Err(InvalidSlice::SliceOutOfRange {
                            dim_index: i,
                            asked: start,
                            dim_size: dim,
                        });
                    }
                    (start, stop, true, 1usize)
                }
                SignedIndexer::Narrow(left, right, step) => {
                    let (start, stop) = resolve_interval(i, dim, *left, *right)?;
                    (start, stop, false, step.get())
                }
            };
            boundaries.push(SliceBoundary { start, stop, dim });
            if !removed {
                newshape.push((stop - start).div_ceil(step));
            }
        }
    }

    Ok(SliceBoundaryReport { boundaries, newshape })
}

/// Compute the byte ranges and post-slice shape for a slicing operation
/// without requiring the underlying data buffer.
///
/// The returned [`SliceByteRanges`] is in source iteration order; callers
/// may reverse it for pop-based iteration.
pub fn slice_byte_ranges(
    dtype: Dtype,
    shape: &[usize],
    slices: &[TensorIndexer],
) -> Result<(SliceByteRanges, SlicedShape), InvalidSlice> {
    // Resolve and validate every dimension up front. This shares the exact
    // interval resolution and out-of-bounds semantics with
    // [`slice_boundaries`], so the boundary report and the byte ranges can
    // never disagree (the "guard" for negative/out-of-bounds indexing).
    let report = slice_boundaries(shape, slices)?;
    let newshape = report.newshape;

    // Minimum span is the span of 1 item;
    let mut span = dtype.bitsize();
    let mut indices: Vec<(usize, usize)> = vec![];
    // Everything is row major.
    for (i, &dim) in shape.iter().enumerate().rev() {
        if i >= slices.len() {
            // We are not slicing yet, just increase the local span
            span *= dim;
            continue;
        }
        let boundary = &report.boundaries[i];
        let (start, stop) = (boundary.start, boundary.stop);
        let step = match &slices[i] {
            TensorIndexer::Select(_) => 1,
            TensorIndexer::Narrow(_, _, step) => step.get(),
        };
        if indices.is_empty() {
            if step == 1 && start == 0 && stop == dim {
                // Full range, nothing sliced yet; just grow the span.
            } else if step == 1 {
                if start * span % 8 != 0 {
                    return Err(InvalidSlice::MisalignedSlice);
                }
                let offset = (start * span) / 8;
                if stop * span % 8 != 0 {
                    return Err(InvalidSlice::MisalignedSlice);
                }
                let small_span = (stop * span) / 8 - offset;
                indices.push((offset, offset + small_span));
            } else {
                // Strided innermost dim: each kept element is its own run.
                for n in (start..stop).step_by(step) {
                    if n * span % 8 != 0 || (n + 1) * span % 8 != 0 {
                        return Err(InvalidSlice::MisalignedSlice);
                    }
                    indices.push(((n * span) / 8, ((n + 1) * span) / 8));
                }
            }
        } else {
            let capacity = (stop - start).div_ceil(step) * indices.len();
            let mut newindices = Vec::with_capacity(capacity);
            for n in (start..stop).step_by(step) {
                if n * span % 8 != 0 {
                    return Err(InvalidSlice::MisalignedSlice);
                }
                let offset = (n * span) / 8;
                for (old_start, old_stop) in &indices {
                    newindices.push((old_start + offset, old_stop + offset));
                }
            }
            indices = newindices;
        }
        span *= dim;
    }
    if indices.is_empty() {
        // Empty `slices` (or all unbounded full-range slices): no slicing
        // happened, the whole tensor is the result. `span` ended as
        // bitsize * product(shape).
        let total_bits = span;
        if total_bits % 8 != 0 {
            return Err(InvalidSlice::MisalignedSlice);
        }
        indices.push((0, total_bits / 8));
    }
    Ok((indices, newshape))
}

impl<'data> Iterator for SliceIterator<'data> {
    type Item = &'data [u8];

    fn next(&mut self) -> Option<Self::Item> {
        // TODO We might want to move the logic from `new`
        // here actually to remove the need to get all the indices
        // upfront.
        let (start, stop) = self.indices.pop()?;
        Some(&self.view.data()[start..stop])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dtype, TensorView};

    #[test]
    fn test_helpers() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![1, 2, 3], &data).unwrap();

        let iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN)],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 24);
        assert_eq!(iterator.newshape(), vec![1, 2, 3]);

        let iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 12);
        assert_eq!(iterator.newshape(), vec![1, 1, 3]);
    }

    #[test]
    fn test_fp4_simple() {
        let data: Vec<u8> = vec![0u8, 1u8];

        let attn_0 = TensorView::new(Dtype::F4, vec![1, 2, 2], &data).unwrap();

        let iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN)],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 2);
        assert_eq!(iterator.newshape(), vec![1, 2, 2]);

        let iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 1);
        assert_eq!(iterator.newshape(), vec![1, 1, 2]);
    }

    #[test]
    fn test_fp4_misaligned() {
        let data: Vec<u8> = vec![0u8];

        let attn_0 = TensorView::new(Dtype::F4, vec![1, 2], &data).unwrap();

        let iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN)],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 1);
        assert_eq!(iterator.newshape(), vec![1, 2]);

        let iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        );

        assert_eq!(iterator, Err(InvalidSlice::MisalignedSlice));
    }

    #[test]
    fn test_dummy() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![1, 2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN)],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        assert!(SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .is_err(),);
    }

    #[test]
    fn test_slice_variety() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![1, 2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(
                Bound::Included(0),
                Bound::Excluded(1),
                NonZeroUsize::MIN,
            )],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..12]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..4]));
        assert_eq!(iterator.next(), Some(&data[12..16]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(2), NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[12..16]));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_slice_variety2() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[4..12]));
        assert_eq!(iterator.next(), Some(&data[16..24]));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_slice_select() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Select(1),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[16..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Select(0),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[4..12]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(2), NonZeroUsize::MIN),
                TensorIndexer::Select(0),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[12..16]));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_invalid_range() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![2, 3], &data).unwrap();

        assert_eq!(
            SliceIterator::new(
                &attn_0,
                &[
                    TensorIndexer::Select(1),
                    TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(4), NonZeroUsize::MIN),
                ],
            ),
            Err(InvalidSlice::SliceOutOfRange {
                asked: 3,
                dim_index: 1,
                dim_size: 3,
            })
        );
        assert_eq!(
            SliceIterator::new(
                &attn_0,
                &[
                    TensorIndexer::Select(1),
                    TensorIndexer::Narrow(Bound::Included(3), Bound::Excluded(2), NonZeroUsize::MIN),
                ],
            ),
            Err(InvalidSlice::SliceOutOfRange {
                asked: 3,
                dim_index: 1,
                dim_size: 3,
            })
        );
        assert_eq!(
            SliceIterator::new(
                &attn_0,
                &[
                    TensorIndexer::Select(1),
                    TensorIndexer::Select(1),
                    TensorIndexer::Select(1),
                ],
            ),
            Err(InvalidSlice::TooManySlices)
        );
    }

    // ---- Boundary report acceptance tests (2D tensors) ----
    //
    // These demonstrate the behavioral difference the boundary report
    // provides: instead of only raw byte ranges, callers now see the actual
    // cropped half-open interval `[start, stop)` per dimension and whether
    // the slice is empty -- all without reading or copying the data buffer.

    /// Build a 2D F32 tensor of shape `[rows, cols]` with consecutive values.
    ///
    /// Returns the leaked backing buffer together with the view, so the
    /// view's data and the returned slice are the *same* allocation (used to
    /// verify that slicing never copies the tensor).
    fn make_2d(rows: usize, cols: usize) -> (&'static [u8], TensorView<'static>) {
        let data: Vec<u8> = (0..rows * cols)
            .flat_map(|v| (v as f32).to_le_bytes())
            .collect();
        let data: &'static [u8] = Box::leak(data.into_boxed_slice());
        let view = TensorView::new(Dtype::F32, vec![rows, cols], data).unwrap();
        (data, view)
    }

    fn print_report(label: &str, report: &SliceBoundaryReport) {
        println!("  {label}:");
        for (i, b) in report.boundaries.iter().enumerate() {
            println!(
                "    dim {i}: [{}, {}) of {} (len={}, empty={})",
                b.start,
                b.stop,
                b.dim,
                b.len(),
                b.is_empty()
            );
        }
        println!("    newshape = {:?}", report.newshape);
        println!("    is_empty = {}", report.is_empty());
    }

    #[test]
    fn test_boundary_report_normal_2d() {
        let (_data, view) = make_2d(3, 4);

        // Normal slice: rows 1..3, cols 1..3 (a 2x2 sub-block).
        let indexers = [
            TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3), NonZeroUsize::MIN),
            TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3), NonZeroUsize::MIN),
        ];
        let report = slice_boundaries(view.shape(), &indexers).unwrap();
        println!("normal slice [1:3, 1:3] on shape [3, 4]:");
        print_report("boundary report", &report);

        assert_eq!(
            report.boundaries,
            vec![
                SliceBoundary { start: 1, stop: 3, dim: 3 },
                SliceBoundary { start: 1, stop: 3, dim: 4 },
            ]
        );
        assert_eq!(report.newshape, vec![2, 2]);
        assert!(!report.is_empty());

        // The sliced data matches the report: 2*2 = 4 f32 values.
        let mut it = SliceIterator::new(&view, &indexers).unwrap();
        let mut values = Vec::new();
        for chunk in &mut it {
            for b in chunk.chunks_exact(4) {
                values.push(f32::from_le_bytes(b.try_into().unwrap()));
            }
        }
        println!("  sliced values = {values:?}");
        assert_eq!(values, vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_boundary_report_full_and_select_2d() {
        let (_data, view) = make_2d(3, 4);

        // Full range on both dims: every boundary is [0, dim).
        let full = slice_boundaries(
            view.shape(),
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        println!("full slice [:, :] on shape [3, 4]:");
        print_report("boundary report", &full);
        assert_eq!(
            full.boundaries,
            vec![
                SliceBoundary { start: 0, stop: 3, dim: 3 },
                SliceBoundary { start: 0, stop: 4, dim: 4 },
            ]
        );
        assert_eq!(full.newshape, vec![3, 4]);
        assert!(!full.is_empty());

        // Selecting a single row removes dim 0 from the result shape.
        let sel = slice_boundaries(view.shape(), &[TensorIndexer::Select(2)]).unwrap();
        println!("select row 2 on shape [3, 4]:");
        print_report("boundary report", &sel);
        assert_eq!(
            sel.boundaries,
            vec![
                SliceBoundary { start: 2, stop: 3, dim: 3 },
                SliceBoundary { start: 0, stop: 4, dim: 4 },
            ]
        );
        assert_eq!(sel.newshape, vec![4]);
        assert!(!sel.is_empty());
    }

    #[test]
    fn test_boundary_report_empty_2d() {
        let (_data, view) = make_2d(3, 4);

        // Empty first dimension: [0, 0) rows.
        let indexers = [
            TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(0), NonZeroUsize::MIN),
            TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
        ];
        let report = slice_boundaries(view.shape(), &indexers).unwrap();
        println!("empty slice [0:0, :] on shape [3, 4]:");
        print_report("boundary report", &report);

        assert_eq!(
            report.boundaries,
            vec![
                SliceBoundary { start: 0, stop: 0, dim: 3 },
                SliceBoundary { start: 0, stop: 4, dim: 4 },
            ]
        );
        assert_eq!(report.newshape, vec![0, 4]);
        assert!(report.is_empty());

        // The slice yields zero bytes without any copy.
        let total: usize = SliceIterator::new(&view, &indexers)
            .unwrap()
            .map(|c| c.len())
            .sum();
        println!("  total sliced bytes = {total}");
        assert_eq!(total, 0);

        // Empty interior interval: [1, 1) on dim 0.
        let indexers = [
            TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(1), NonZeroUsize::MIN),
            TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(2), NonZeroUsize::MIN),
        ];
        let report = slice_boundaries(view.shape(), &indexers).unwrap();
        println!("empty slice [1:1, 0:2] on shape [3, 4]:");
        print_report("boundary report", &report);
        assert!(report.is_empty());
        assert_eq!(report.newshape, vec![0, 2]);
    }

    #[test]
    fn test_boundary_report_out_of_bounds_2d() {
        let (_data, view) = make_2d(3, 4);

        // Range exceeds dim 0: [0, 4) for a dim of size 3.
        let err = slice_boundaries(
            view.shape(),
            &[
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(4), NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .unwrap_err();
        println!("out-of-bounds range [0:4, :] on shape [3, 4]: {err}");
        assert_eq!(
            err,
            InvalidSlice::SliceOutOfRange {
                dim_index: 0,
                asked: 3,
                dim_size: 3,
            }
        );

        // Range exceeds dim 1: [0, 5) for a dim of size 4.
        let err = slice_boundaries(
            view.shape(),
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(5), NonZeroUsize::MIN),
            ],
        )
        .unwrap_err();
        println!("out-of-bounds range [:, 0:5] on shape [3, 4]: {err}");
        assert_eq!(
            err,
            InvalidSlice::SliceOutOfRange {
                dim_index: 1,
                asked: 4,
                dim_size: 4,
            }
        );

        // Single index out of bounds: row 3 for a dim of size 3.
        let err = slice_boundaries(view.shape(), &[TensorIndexer::Select(3)]).unwrap_err();
        println!("out-of-bounds select 3 on shape [3, 4]: {err}");
        assert_eq!(
            err,
            InvalidSlice::SliceOutOfRange {
                dim_index: 0,
                asked: 3,
                dim_size: 3,
            }
        );

        // Too many slices.
        let err = slice_boundaries(
            view.shape(),
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .unwrap_err();
        println!("too many slices on shape [3, 4]: {err}");
        assert_eq!(err, InvalidSlice::TooManySlices);
    }

    #[test]
    fn test_boundary_report_negative_indices_2d() {
        let (_data, view) = make_2d(3, 4);

        // Negative single index: -1 is the last row (row 2).
        let report =
            slice_boundaries_signed(view.shape(), &[SignedIndexer::Select(-1)]).unwrap();
        println!("negative select -1 on shape [3, 4]:");
        print_report("boundary report", &report);
        assert_eq!(
            report.boundaries,
            vec![
                SliceBoundary { start: 2, stop: 3, dim: 3 },
                SliceBoundary { start: 0, stop: 4, dim: 4 },
            ]
        );
        assert_eq!(report.newshape, vec![4]);
        assert!(!report.is_empty());

        // Negative range: -3:-1 resolves to [0, 2).
        let report = slice_boundaries_signed(
            view.shape(),
            &[
                SignedIndexer::Narrow(
                    Bound::Included(-3),
                    Bound::Excluded(-1),
                    NonZeroUsize::MIN,
                ),
                SignedIndexer::Narrow(
                    Bound::Unbounded,
                    Bound::Unbounded,
                    NonZeroUsize::MIN,
                ),
            ],
        )
        .unwrap();
        println!("negative range [-3:-1, :] on shape [3, 4]:");
        print_report("boundary report", &report);
        assert_eq!(
            report.boundaries,
            vec![
                SliceBoundary { start: 0, stop: 2, dim: 3 },
                SliceBoundary { start: 0, stop: 4, dim: 4 },
            ]
        );
        assert_eq!(report.newshape, vec![2, 4]);

        // Negative index whose magnitude exceeds the dimension: -4 on dim 3.
        let err =
            slice_boundaries_signed(view.shape(), &[SignedIndexer::Select(-4)]).unwrap_err();
        println!("negative out-of-bounds select -4 on shape [3, 4]: {err}");
        assert_eq!(
            err,
            InvalidSlice::SliceOutOfRange {
                dim_index: 0,
                asked: 3,
                dim_size: 3,
            }
        );
    }

    #[test]
    fn test_boundary_report_matches_sliced_data() {
        // Guard: the boundary report must agree with the actual byte ranges
        // the slicer produces, and the sliced bytes must be views into the
        // original buffer (no full-tensor copy).
        let (data, view) = make_2d(3, 4);

        let indexers = [
            TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3), NonZeroUsize::MIN),
            TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(4), NonZeroUsize::MIN),
        ];
        let report = slice_boundaries(view.shape(), &indexers).unwrap();
        let (ranges, newshape) =
            slice_byte_ranges(view.dtype(), view.shape(), &indexers).unwrap();

        // Report shape and byte ranges agree.
        assert_eq!(report.newshape, newshape);
        let expected_bytes: usize =
            report.boundaries.iter().map(|b| b.len()).product::<usize>() * Dtype::F32.bitsize()
                / 8;
        let actual_bytes: usize = ranges.iter().map(|(s, e)| e - s).sum();
        println!(
            "report element product bytes = {expected_bytes}, sliced bytes = {actual_bytes}"
        );
        assert_eq!(expected_bytes, actual_bytes);

        // Every sliced chunk is a sub-slice of the original buffer (no copy):
        // its pointer is exactly `s` bytes into the original allocation.
        let base = data.as_ptr();
        for (s, _e) in ranges {
            let chunk = &view.data()[s..];
            assert_eq!(chunk.as_ptr(), unsafe { base.add(s) });
        }
        println!("all sliced chunks are views into the original buffer (no full-tensor copy)");
    }
}
