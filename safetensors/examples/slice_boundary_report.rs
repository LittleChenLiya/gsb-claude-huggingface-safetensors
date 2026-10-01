//! Acceptance demo for "tensor slice boundary reporting".
//!
//! A 2-D `[3, 4]` f32 tensor is sliced three ways; for each slice we print
//! the *actual clipped interval* selected on every dimension and whether the
//! result is empty, plus the byte chunks the lazy iterator borrows (no full
//! tensor copy).
//!
//! Run with:
//!     cargo run --example slice_boundary_report
use safetensors::slice::{slice_boundaries, slice_byte_ranges, InvalidSlice, TensorIndexer};
use safetensors::tensor::TensorView;
use safetensors::Dtype;
use std::num::NonZeroUsize;
use std::ops::Bound;

fn range(start: usize, stop: usize) -> TensorIndexer {
    TensorIndexer::Narrow(
        Bound::Included(start),
        Bound::Excluded(stop),
        NonZeroUsize::MIN,
    )
}

fn values(bytes: &[u8], ranges: &[(usize, usize)]) -> Vec<f32> {
    ranges
        .iter()
        .flat_map(|&(a, b)| {
            bytes[a..b]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        })
        .collect()
}

fn main() {
    // 3 rows x 4 cols, value = row*10 + col
    let data: Vec<u8> = (0u32..12)
        .map(|v| (v / 4 * 10 + v % 4) as f32)
        .flat_map(f32::to_le_bytes)
        .collect();
    let shape = [3usize, 4usize];
    let view = TensorView::new(Dtype::F32, shape.to_vec(), &data).unwrap();

    println!("tensor shape = [3, 4], values:");
    for row in 0..3 {
        let row_vals: Vec<f32> = data[row * 16..(row + 1) * 16]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        println!("  row {row}: {row_vals:?}");
    }

    // --- Case 1: normal slice tensor[1:3, 1:4] ---------------------------
    let slices = vec![range(1, 3), range(1, 4)];
    let report = slice_boundaries(&shape, &slices).unwrap();
    let (ranges, newshape) = slice_byte_ranges(Dtype::F32, &shape, &slices).unwrap();
    println!("\n[1] normal slice [1:3, 1:4]");
    for b in &report.boundaries {
        println!(
            "    dim {} (size {}): clipped interval {:?}, selected {} elements",
            b.dim_index,
            b.dim_size,
            b.clipped_range(),
            b.count
        );
    }
    println!(
        "    is_empty = {}, newshape = {:?}",
        report.is_empty(),
        newshape
    );
    println!(
        "    borrowed byte ranges = {ranges:?}, values = {:?}",
        values(&data, &ranges)
    );
    let iter = view.sliced_data(&slices).unwrap();
    println!(
        "    iterator.remaining_byte_len() = {} (full tensor would be 48)",
        iter.remaining_byte_len()
    );

    // --- Case 2: empty slice tensor[0:2, 3:3] ----------------------------
    let slices = vec![range(0, 2), range(3, 3)];
    let report = slice_boundaries(&shape, &slices).unwrap();
    let (ranges, newshape) = slice_byte_ranges(Dtype::F32, &shape, &slices).unwrap();
    println!("\n[2] empty slice [0:2, 3:3]");
    for b in &report.boundaries {
        println!(
            "    dim {} (size {}): clipped interval {:?}, selected {} elements",
            b.dim_index,
            b.dim_size,
            b.clipped_range(),
            b.count
        );
    }
    println!(
        "    is_empty = {}, newshape = {:?}",
        report.is_empty(),
        newshape
    );
    println!("    borrowed byte ranges = {ranges:?} (no chunks, no copy)");

    // Descending in-bounds range also resolves to empty (previously this
    // underflowed in a debug build).
    let report = slice_boundaries(&shape, &[range(0, 3), range(2, 1)]).unwrap();
    println!(
        "    [:, 2:1] -> is_empty = {}, newshape = {:?}",
        report.is_empty(),
        report.newshape
    );

    // --- Case 3: out-of-range slice tensor[1:3, 1:5] ---------------------
    let err = slice_boundaries(&shape, &[range(1, 3), range(1, 5)]).unwrap_err();
    println!("\n[3] out-of-range slice [1:3, 1:5]");
    match err {
        InvalidSlice::SliceOutOfRange {
            dim_index,
            asked,
            dim_size,
        } => println!(
            "    ERROR: index {asked} out of bounds for dimension #{dim_index} of size {dim_size}"
        ),
        other => println!("    ERROR: {other}"),
    }
    let bad = vec![range(1, 3), range(1, 5)];
    let err = match view.sliced_data(&bad) {
        Err(e) => e,
        Ok(_) => panic!("expected an out-of-range error"),
    };
    println!("    TensorView::slice -> Err({err})");

    // --- Bonus: negative index semantics ---------------------------------
    let select = TensorIndexer::select(-1, 4, 1).unwrap(); // last column
    let report = slice_boundaries(&shape, &[range(0, 3), select]).unwrap();
    println!("\n[bonus] tensor[:, -1] (negative scalar index)");
    println!(
        "    dim 1 clipped interval = {:?}, newshape = {:?}",
        report.boundaries[1].clipped_range(),
        report.newshape
    );
    let clipped = TensorIndexer::narrow(-100, 100, NonZeroUsize::MIN, 4);
    let report = slice_boundaries(&shape, &[range(0, 3), clipped]).unwrap();
    println!(
        "    tensor[:, -100:100] range clips to {:?}, newshape = {:?}",
        report.boundaries[1].clipped_range(),
        report.newshape
    );
}
