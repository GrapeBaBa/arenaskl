#![feature(test)]
#![feature(ptr_internals)]

extern crate num_cpus;
extern crate test;
extern crate threadpool;

use criterion::criterion_main;

mod suites;

criterion_main! {
    suites::bench_arena::benches,
    suites::bench_node::benches,
    suites::bench_skl::benches,
}
