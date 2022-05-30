use arenaskl::arena::Arena;
use arenaskl::arena::ALIGN4;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

fn criterion_benchmark_arena_alloc(c: &mut Criterion) {
    let arena = Arena::new(u32::MAX);
    c.bench_function("arena_alloc", |b| {
        b.iter(|| test::black_box(arena.alloc(100 as u32, ALIGN4, 0)))
    });
}

criterion_group!(benches, criterion_benchmark_arena_alloc);
criterion_main!(benches);
