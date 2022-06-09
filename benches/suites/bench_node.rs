use arenaskl::arena::Arena;
use arenaskl::node::Node;
use arenaskl::skl::{InternalKey, INTERNAL_KEY_KIND_SET};
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

fn criterion_benchmark_new_node(c: &mut Criterion) {
    let mut a = Arena::new(u32::MAX);
    let key = InternalKey::new(&[1u8, 1u8, 1u8, 1u8], 1u64, INTERNAL_KEY_KIND_SET);
    c.bench_function("new_node", |b| {
        b.iter(|| test::black_box(Node::new_node(&mut a, 12, &key, &[1u8, 1u8, 1u8, 1u8, 1u8])))
    });
}

criterion_group!(benches, criterion_benchmark_new_node);
criterion_main!(benches);
