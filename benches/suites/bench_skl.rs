use arenaskl::arena::Arena;
use arenaskl::skl::{InternalKey, Skiplist, INTERNAL_KEY_KIND_SET};
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

fn criterion_benchmark_skl_add(c: &mut Criterion) {
    let mut a = Arena::new(u32::MAX);
    let mut skl = Skiplist::new(&mut a).unwrap();

    c.bench_function("skl_add", |b| {
        b.iter(|| {
            let key = InternalKey::new(&[1u8, 1u8, 1u8, 1u8], 1u64, INTERNAL_KEY_KIND_SET);
            test::black_box(skl.add(&key, &[1u8, 1u8, 1u8, 1u8]))
        })
    });
}

fn criterion_benchmark_skl_iter(c: &mut Criterion) {
    let mut a = Arena::new(u32::MAX);
    let mut skl = Skiplist::new(&mut a).unwrap();
    let key = InternalKey::new(&[1u8, 1u8, 1u8, 1u8], 1u64, INTERNAL_KEY_KIND_SET);
    let _ = skl.add(&key, &[1u8, 1u8, 1u8, 1u8]);

    c.bench_function("skl_iter", |b| {
        b.iter(|| {
            let mut iter = skl.iter(&[] as &[u8], &[] as &[u8]);
            iter.next();
        })
    });
}

criterion_group!(
    benches,
    criterion_benchmark_skl_add,
    criterion_benchmark_skl_iter
);
criterion_main!(benches);
