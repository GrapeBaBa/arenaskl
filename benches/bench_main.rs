#![feature(test)]

extern crate test;

use criterion::criterion_main;

mod suites;

criterion_main! {
    suites::bench_arena::benches,
    suites::bench_node::benches,
    suites::bench_skl::benches,
}
