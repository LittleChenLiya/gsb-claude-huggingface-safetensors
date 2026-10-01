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

/// Resolve a Python-style scalar index against a dimension.
///
/// Negative indexes count backwards from the end of the dimension
/// (e.g. `-1` is the last element). A scalar index that ends up outside of
/// `0..dim_size` is rejected with [`InvalidSlice::SliceOutOfRange`], matching
/// the original index semantics: scalar indexing out of bounds is an error.
pub fn resolve_index_at(
    index: i64,
    dim_size: usize,
    dim_index: usize,
) -> Result<usize, InvalidSlice> {
    let dim = dim_size as i128;
    let resolved = if index < 0 {
        dim + index as i128
    } else {
        index as i128
    };
    if resolved < 0 || resolved >= dim {
        // Report the past-the-end position when a negative index falls
        // before the start of the dimension.
        let asked = usize::try_from(resolved).unwrap_or(dim_size);
        return Err(InvalidSlice::SliceOutOfRange {
            dim_index,
            asked,
            dim_size,
        });
    }
    Ok(resolved as usize)
}

/// Normalize a Python-style (possibly negative) *range* endpoint against a
/// dimension.
///
/// Unlike scalar indexes, range endpoints never raise out of bounds: they
/// count backwards from the end when negative and are clipped to
/// `0..=dim_size` (e.g. `x[-100..100]` on a size-3 dimension becomes
/// `0..3`). This mirrors the original Python slicing semantics.
pub fn normalize_narrow_index(value: i64, dim_size: usize) -> usize {
    let dim = dim_size as i128;
    let resolved = if value < 0 {
        dim + value as i128
    } else {
        value as i128
    };
    resolved.clamp(0, dim) as usize
}

impl TensorIndexer {
    /// Build a scalar [`TensorIndexer::Select`] from a Python-style (possibly
    /// negative) index. Negative indexes count from the end; an out-of-range
    /// scalar is an error.
    pub fn select(
        index: i64,
        dim_size: usize,
        dim_index: usize,
    ) -> Result<TensorIndexer, InvalidSlice> {
        resolve_index_at(index, dim_size, dim_index).map(TensorIndexer::Select)
    }

    /// Build a half-open [`TensorIndexer::Narrow`] from Python-style (possibly
    /// negative) endpoints. Endpoints are clipped to the dimension rather than
    /// rejected, matching range-slicing semantics.
    pub fn narrow(start: i64, end: i64, step: NonZeroUsize, dim_size: usize) -> TensorIndexer {
        let start = normalize_narrow_index(start, dim_size);
        let end = normalize_narrow_index(end, dim_size);
        TensorIndexer::Narrow(Bound::Included(start), Bound::Excluded(end), step)
    }
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
    boundaries: BoundaryReport,
}

impl<'data> SliceIterator<'data> {
    pub(crate) fn new(
        view: &'data TensorView<'data>,
        slices: &[TensorIndexer],
    ) -> Result<Self, InvalidSlice> {
        let (indices, boundaries) = slice_plan(view.dtype(), view.shape(), slices)?;
        let newshape = boundaries.newshape.clone();
        // Reversing so we can pop faster while iterating on the slice
        let indices = indices.into_iter().rev().collect();
        Ok(Self {
            view,
            indices,
            newshape,
            boundaries,
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

    /// Boundary report: the actual clipped element interval selected on each
    /// dimension, the resulting shape and whether the slice is empty.
    ///
    /// Computed from shape metadata only; the underlying tensor data is never
    /// copied or read to build it.
    pub fn boundaries(&self) -> &BoundaryReport {
        &self.boundaries
    }

    /// Convenience accessor: `true` when this slice selects no element.
    pub fn is_empty_slice(&self) -> bool {
        self.boundaries.is_empty()
    }
}

/// Byte ranges into a tensor's data section in iteration order;
/// concatenating them yields the dense destination layout.
pub type SliceByteRanges = Vec<(usize, usize)>;

/// Post-slice tensor shape (element counts per dim).
pub type SlicedShape = Vec<usize>;

/// How a single dimension was addressed by a slicing operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimSelection {
    /// The dimension was not mentioned in the slice index; it is kept whole.
    Untouched,
    /// A scalar index picked one element, collapsing the dimension.
    Select,
    /// A `start:stop:step` range was applied.
    Narrow,
}

/// Resolved boundary of one dimension after applying a slice.
///
/// For range (`start:stop:step`) slices, `start`/`stop` describe the
/// half-open **clipped** element interval actually read from the source
/// tensor. For a scalar select the interval is `index..index + 1`. For an
/// untouched dimension it is the full `0..dim_size` interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceBoundary {
    /// Position of the dimension in the source tensor.
    pub dim_index: usize,
    /// Size of the dimension before slicing.
    pub dim_size: usize,
    /// How this dimension was addressed (untouched, scalar select or range).
    pub selection: DimSelection,
    /// Inclusive start of the resolved interval in source coordinates.
    pub start: usize,
    /// Exclusive stop of the resolved interval in source coordinates.
    pub stop: usize,
    /// Step used while walking the interval (always >= 1).
    pub step: usize,
    /// Number of elements selected from this dimension. `0` means the whole
    /// slice is empty along this dimension.
    pub count: usize,
}

impl SliceBoundary {
    /// Whether this dimension contributes no element at all (e.g. an
    /// in-bounds empty range such as `2:2` or `2:1`).
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The resolved `start..stop` half-open interval actually read from the
    /// source tensor.
    pub fn clipped_range(&self) -> Range<usize> {
        self.start..self.stop
    }
}

/// Report describing, per dimension, the interval that a slicing operation
/// actually reads, together with the resulting shape.
///
/// The boundaries are ordered by source dimension index (outer to inner).
/// This report is computed from metadata alone: no tensor data is copied to
/// produce it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryReport {
    /// Resolved boundary for every source dimension.
    pub boundaries: Vec<SliceBoundary>,
    /// Shape of the tensor after slicing (scalar-selected dims removed).
    pub newshape: Vec<usize>,
}

impl BoundaryReport {
    /// `true` when the slicing selects no element: at least one dimension has
    /// a zero count.
    pub fn is_empty(&self) -> bool {
        self.boundaries.iter().any(SliceBoundary::is_empty)
    }

    /// Number of elements selected by the slicing operation (product of the
    /// post-slice shape).
    pub fn element_count(&self) -> usize {
        self.newshape.iter().product()
    }
}

/// Per-dimension resolved view produced before the dtype/byte layout pass.
#[derive(Debug, Clone)]
struct ResolvedDim {
    dim_index: usize,
    dim_size: usize,
    selection: DimSelection,
    start: usize,
    stop: usize,
    step: usize,
    count: usize,
}

/// Resolve every slice indexer against `shape`, reporting the clipped
/// interval per dimension without consulting the dtype or data buffer.
///
/// Out-of-range indexes follow the original index semantics and produce an
/// [`InvalidSlice::SliceOutOfRange`]; in-bounds empty or descending ranges
/// (e.g. `2:1`) resolve to an empty interval instead of underflowing.
fn resolve_dimensions(
    shape: &[usize],
    slices: &[TensorIndexer],
) -> Result<Vec<ResolvedDim>, InvalidSlice> {
    let n_slice = slices.len();
    let n_shape = shape.len();
    if n_slice > n_shape {
        return Err(InvalidSlice::TooManySlices);
    }
    shape
        .iter()
        .enumerate()
        .map(|(i, &dim)| {
            if i >= n_slice {
                return Ok(ResolvedDim {
                    dim_index: i,
                    dim_size: dim,
                    selection: DimSelection::Untouched,
                    start: 0,
                    stop: dim,
                    step: 1,
                    count: dim,
                });
            }
            let (selection, start, stop, step) = match &slices[i] {
                TensorIndexer::Select(s) => (DimSelection::Select, *s, *s + 1, 1usize),
                TensorIndexer::Narrow(left, right, step) => {
                    let (start, stop) = narrow_bounds(left, right, dim);
                    (DimSelection::Narrow, start, stop, step.get())
                }
            };
            // Strict bounds check, preserving the original error semantics:
            // scalar indexes and range endpoints past the dimension fail.
            if start >= dim || stop > dim {
                let asked = if start >= dim {
                    start
                } else {
                    stop.saturating_sub(1)
                };
                return Err(InvalidSlice::SliceOutOfRange {
                    dim_index: i,
                    asked,
                    dim_size: dim,
                });
            }
            // In-bounds descending/empty ranges (e.g. `2:1`, `2:2`) select
            // nothing instead of underflowing.
            let count = if stop <= start {
                0
            } else {
                (stop - start).div_ceil(step)
            };
            Ok(ResolvedDim {
                dim_index: i,
                dim_size: dim,
                selection,
                start,
                stop,
                step,
                count,
            })
        })
        .collect()
}

/// Build a [`BoundaryReport`] from already-resolved dimensions.
fn report_from_resolved(resolved: &[ResolvedDim]) -> BoundaryReport {
    let boundaries = resolved
        .iter()
        .map(|d| SliceBoundary {
            dim_index: d.dim_index,
            dim_size: d.dim_size,
            selection: d.selection.clone(),
            start: d.start,
            stop: d.stop,
            step: d.step,
            count: d.count,
        })
        .collect();
    // Scalar-selected dimensions disappear from the resulting shape.
    let newshape = resolved
        .iter()
        .filter(|d| !matches!(d.selection, DimSelection::Select))
        .map(|d| d.count)
        .collect();
    BoundaryReport {
        boundaries,
        newshape,
    }
}

/// Compute the boundary report (actual clipped interval per dimension,
/// resulting shape and emptiness) of a slicing operation from shape metadata
/// alone.
///
/// This never touches the tensor's data buffer and performs no copy; negative
/// indexes can be normalized beforehand with [`resolve_index_at`] /
/// [`normalize_narrow_index`] or the [`TensorIndexer::select`] /
/// [`TensorIndexer::narrow`] constructors.
pub fn slice_boundaries(
    shape: &[usize],
    slices: &[TensorIndexer],
) -> Result<BoundaryReport, InvalidSlice> {
    let resolved = resolve_dimensions(shape, slices)?;
    Ok(report_from_resolved(&resolved))
}

/// Resolve a `(start, stop)` half-open element range from slice bounds,
/// defaulting unbounded ends to `0` and `dim`.
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

/// Compute the byte ranges for already-resolved dimensions.
///
/// The returned [`SliceByteRanges`] is in source iteration order; callers
/// may reverse it for pop-based iteration.
fn byte_ranges_from_resolved(
    dtype: Dtype,
    resolved: &[ResolvedDim],
) -> Result<SliceByteRanges, InvalidSlice> {
    // Minimum span is the span of 1 item;
    let mut span = dtype.bitsize();
    let mut indices: Vec<(usize, usize)> = vec![];
    // Whether any indexer actually restricted the byte layout.
    let mut sliced = false;
    // Set once an (inner) dimension selects no element; outer dimensions
    // cannot bring those bytes back.
    let mut emptied = false;
    // Everything is row major: iterate from the innermost dimension outward.
    for d in resolved.iter().rev() {
        let ResolvedDim {
            dim_size: dim,
            start,
            stop,
            step,
            count,
            ..
        } = d;
        let full_range = *step == 1 && *start == 0 && *stop == *dim;

        if emptied {
            // An inner dimension already selected nothing; stay empty.
            span *= dim;
            continue;
        }

        // An empty interval contributes no bytes; reset so outer dims cannot
        // re-expand the set.
        if *count == 0 {
            indices.clear();
            emptied = true;
            // An explicit empty selection restricts the layout (to nothing),
            // so the whole-tensor fallback must not fire.
            sliced = true;
        } else if indices.is_empty() {
            if full_range {
                // Full range, nothing sliced yet; just grow the span.
            } else if *step == 1 {
                sliced = true;
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
                sliced = true;
                for n in (*start..*stop).step_by(*step) {
                    if n * span % 8 != 0 || (n + 1) * span % 8 != 0 {
                        return Err(InvalidSlice::MisalignedSlice);
                    }
                    indices.push(((n * span) / 8, ((n + 1) * span) / 8));
                }
            }
        } else {
            let mut newindices = Vec::with_capacity(*count * indices.len());
            for n in (*start..*stop).step_by(*step) {
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
    if !sliced {
        // Empty `slices` (or all unbounded full-range slices): no slicing
        // happened, the whole tensor is the result. `span` ended as
        // bitsize * product(shape).
        let total_bits = span;
        if total_bits % 8 != 0 {
            return Err(InvalidSlice::MisalignedSlice);
        }
        indices.push((0, total_bits / 8));
    }
    Ok(indices)
}

/// Resolve slicing once into both its [`BoundaryReport`] and the byte ranges
/// that materialize it, avoiding duplicated index resolution.
pub(crate) fn slice_plan(
    dtype: Dtype,
    shape: &[usize],
    slices: &[TensorIndexer],
) -> Result<(SliceByteRanges, BoundaryReport), InvalidSlice> {
    let resolved = resolve_dimensions(shape, slices)?;
    let report = report_from_resolved(&resolved);
    let indices = byte_ranges_from_resolved(dtype, &resolved)?;
    Ok((indices, report))
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
    let (indices, report) = slice_plan(dtype, shape, slices)?;
    Ok((indices, report.newshape))
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
            &[TensorIndexer::Narrow(
                Bound::Unbounded,
                Bound::Unbounded,
                NonZeroUsize::MIN,
            )],
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
            &[TensorIndexer::Narrow(
                Bound::Unbounded,
                Bound::Unbounded,
                NonZeroUsize::MIN,
            )],
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
            &[TensorIndexer::Narrow(
                Bound::Unbounded,
                Bound::Unbounded,
                NonZeroUsize::MIN,
            )],
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
            &[TensorIndexer::Narrow(
                Bound::Unbounded,
                Bound::Unbounded,
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

    fn range(start: usize, stop: usize) -> TensorIndexer {
        TensorIndexer::Narrow(
            Bound::Included(start),
            Bound::Excluded(stop),
            NonZeroUsize::MIN,
        )
    }

    /// 2D tensor `[3, 4]` of f32 values `row * 10 + col`.
    fn matrix2d() -> (usize, usize, TensorView<'static>) {
        let values: Vec<f32> = (0..12u32).map(|v| (v / 4 * 10 + v % 4) as f32).collect();
        let data: Vec<u8> = values.into_iter().flat_map(|f| f.to_le_bytes()).collect();
        let len = data.len();
        // Leak to obtain a 'static view; keep the base address so tests can
        // prove the iterator only borrows sub-slices of this exact buffer.
        let leaked: &'static [u8] = Box::leak(data.into_boxed_slice());
        let base = leaked.as_ptr() as usize;
        let view = TensorView::new(Dtype::F32, vec![3, 4], leaked).unwrap();
        (base, len, view)
    }

    #[test]
    fn test_boundary_report_normal_slice_2d() {
        // Normal range slice on both dims: tensor[1:3, 1:4]
        let report = slice_boundaries(&[3, 4], &[range(1, 3), range(1, 4)]).unwrap();
        assert!(!report.is_empty());
        assert_eq!(report.newshape, vec![2, 3]);
        assert_eq!(report.element_count(), 6);
        assert_eq!(report.boundaries.len(), 2);

        let row = &report.boundaries[0];
        assert_eq!(row.dim_index, 0);
        assert_eq!(row.dim_size, 3);
        assert_eq!(row.selection, DimSelection::Narrow);
        assert_eq!(row.clipped_range(), 1..3);
        assert_eq!(row.count, 2);
        assert!(!row.is_empty());

        let col = &report.boundaries[1];
        assert_eq!(col.clipped_range(), 1..4);
        assert_eq!(col.count, 3);
    }

    #[test]
    fn test_boundary_report_select_collapses_dim() {
        // Scalar select on the outer dim: tensor[-2, 1:3]
        let select = TensorIndexer::select(-2, 3, 0).unwrap();
        let report = slice_boundaries(&[3, 4], &[select, range(1, 3)]).unwrap();
        assert_eq!(report.newshape, vec![2]);
        assert_eq!(report.element_count(), 2);

        let row = &report.boundaries[0];
        assert_eq!(row.selection, DimSelection::Select);
        // -2 on a size-3 dim resolves to index 1.
        assert_eq!(row.clipped_range(), 1..2);
        assert_eq!(row.count, 1);

        let col = &report.boundaries[1];
        assert_eq!(col.selection, DimSelection::Narrow);
        assert_eq!(col.clipped_range(), 1..3);
    }

    #[test]
    fn test_boundary_report_empty_slice_2d() {
        // In-bounds empty range on the column dim: tensor[0:2, 3:3]
        let report = slice_boundaries(&[3, 4], &[range(0, 2), range(3, 3)]).unwrap();
        assert!(report.is_empty());
        assert_eq!(report.newshape, vec![2, 0]);
        assert_eq!(report.element_count(), 0);
        assert_eq!(report.boundaries[1].clipped_range(), 3..3);
        assert!(report.boundaries[1].is_empty());

        // Descending in-bounds range also resolves to empty instead of
        // underflowing: tensor[:, 2:1].
        let report = slice_boundaries(&[3, 4], &[range(0, 3), range(2, 1)]).unwrap();
        assert!(report.is_empty());
        assert_eq!(report.newshape, vec![3, 0]);

        // Byte layout of an empty slice is empty and yields no chunks.
        let (ranges, shape) =
            slice_byte_ranges(Dtype::F32, &[3, 4], &[range(0, 2), range(3, 3)]).unwrap();
        assert_eq!(shape, vec![2, 0]);
        assert_eq!(ranges, vec![]);
    }

    #[test]
    fn test_boundary_report_out_of_range_2d() {
        // Endpoint past the column dimension.
        let err = slice_boundaries(&[3, 4], &[range(0, 1), range(1, 5)]).unwrap_err();
        assert_eq!(
            err,
            InvalidSlice::SliceOutOfRange {
                dim_index: 1,
                asked: 4,
                dim_size: 4,
            }
        );

        // Start past the row dimension.
        let err = slice_boundaries(&[3, 4], &[range(3, 3), range(0, 1)]).unwrap_err();
        assert_eq!(
            err,
            InvalidSlice::SliceOutOfRange {
                dim_index: 0,
                asked: 3,
                dim_size: 3,
            }
        );

        // Negative scalar still out of range: -5 on a size-3 dim.
        let err = TensorIndexer::select(-5, 3, 0).unwrap_err();
        assert_eq!(
            err,
            InvalidSlice::SliceOutOfRange {
                dim_index: 0,
                asked: 3,
                dim_size: 3,
            }
        );

        // Scalar select past the dimension keeps original semantics.
        let err = slice_boundaries(&[3, 4], &[TensorIndexer::Select(4), range(0, 1)]).unwrap_err();
        assert!(matches!(
            err,
            InvalidSlice::SliceOutOfRange {
                dim_index: 0,
                dim_size: 3,
                ..
            }
        ));
    }

    #[test]
    fn test_negative_indexes_and_range_clipping() {
        // Negative scalar: -1 is the last row.
        assert_eq!(resolve_index_at(-1, 3, 0), Ok(2));
        assert_eq!(resolve_index_at(-3, 3, 0), Ok(0));
        assert!(resolve_index_at(-4, 3, 0).is_err());
        assert!(resolve_index_at(3, 3, 0).is_err());

        // Negative range endpoints count from the end and get clipped; ranges
        // never raise.
        let indexer = TensorIndexer::narrow(-100, 100, NonZeroUsize::MIN, 4);
        let report = slice_boundaries(&[4], &[indexer]).unwrap();
        assert_eq!(report.boundaries[0].clipped_range(), 0..4);
        assert_eq!(report.element_count(), 4);

        let indexer = TensorIndexer::narrow(-2, -1, NonZeroUsize::MIN, 4);
        let report = slice_boundaries(&[4], &[indexer]).unwrap();
        assert_eq!(report.boundaries[0].clipped_range(), 2..3);
    }

    #[test]
    fn test_iterator_boundaries_and_no_full_copy() {
        let (base, len, view) = matrix2d();
        // slice rows 1..3, cols 1..3 -> 4 f32 values: 11, 12, 21, 22.
        let iter = SliceIterator::new(&view, &[range(1, 3), range(1, 3)]).unwrap();
        let report = iter.boundaries();
        assert!(!iter.is_empty_slice());
        assert_eq!(report.newshape, vec![2, 2]);
        assert_eq!(report.boundaries[0].clipped_range(), 1..3);
        assert_eq!(report.boundaries[1].clipped_range(), 1..3);
        // Only 16 bytes for 4 f32, not the full 48-byte tensor.
        assert_eq!(iter.remaining_byte_len(), 4 * 4);
        drop(iter);

        // Verify the iterator borrows sub-slices of the original buffer
        // instead of duplicating the tensor data.
        let mut iter = SliceIterator::new(&view, &[range(1, 3), range(1, 3)]).unwrap();
        let mut gathered: Vec<u8> = vec![];
        for chunk in &mut iter {
            // Every yielded chunk points into the original allocation.
            let p = chunk.as_ptr() as usize;
            assert!((base..base + len).contains(&p));
            gathered.extend_from_slice(chunk);
        }
        let values: Vec<f32> = gathered
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![11.0, 12.0, 21.0, 22.0]);

        // Empty slice iterator yields nothing and reports empty.
        let iter = SliceIterator::new(&view, &[range(0, 3), range(2, 2)]).unwrap();
        assert!(iter.is_empty_slice());
        assert_eq!(iter.remaining_byte_len(), 0);
        assert_eq!(iter.newshape(), vec![3, 0]);
        assert_eq!(iter.count(), 0);
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
                    TensorIndexer::Narrow(
                        Bound::Included(1),
                        Bound::Excluded(4),
                        NonZeroUsize::MIN
                    ),
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
                    TensorIndexer::Narrow(
                        Bound::Included(3),
                        Bound::Excluded(2),
                        NonZeroUsize::MIN
                    ),
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
}
