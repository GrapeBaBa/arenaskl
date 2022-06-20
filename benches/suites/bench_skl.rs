use arenaskl::arena::Arena;
use arenaskl::skl::{InternalKey, Skiplist, INTERNAL_KEY_KIND_SET};
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use std::ptr::Unique;
use std::time::Instant;

fn criterion_benchmark_skl_add(c: &mut Criterion) {
    let mut a = Arena::new(u32::MAX);
    let mut skl = Skiplist::new(&mut a).unwrap();

    c.bench_function("skl_add", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for i in 0..iters {
                let _ = skl.add(&make_int_key(i as usize), &[1u8, 1u8, 1u8, 1u8]);
            }
            start.elapsed()
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

fn criterion_read_write(c: &mut Criterion) {
    let mut a = Arena::new(u32::MAX);
    let mut skl = Skiplist::new(&mut a).unwrap();
    let mut skl = Unique::new(&mut skl as *mut Skiplist).unwrap();
    let pool = threadpool::ThreadPool::new(num_cpus::get());
    for i in 0..=10 {
        let read_frac = i as f32 / 10.0f32;

        c.bench_function(format!("frac_{}", i * 10).as_str(), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    unsafe {
                        pool.execute(move || {
                            let mut iter = skl.as_mut().iter(&[] as &[u8], &[] as &[u8]);
                            let mut buf = [0u8; 8];
                            let mut rng = thread_rng();
                            if rng.gen::<f32>() < read_frac {
                                iter.seek_ge(random_key(rng, &mut buf).user_key.as_slice(), false);
                            } else {
                                let _ = skl.as_mut().add(&random_key(rng, &mut buf), &[] as &[u8]);
                            }
                        });
                    }
                }
                start.elapsed()
            })
        });
    }
}

fn random_key(mut rng: ThreadRng, b: &mut [u8]) -> InternalKey {
    let key = rng.gen::<u32>();
    let key2 = rng.gen::<u32>();
    let a: [u8; 4] = key.to_ne_bytes();
    let a1: [u8; 4] = key2.to_ne_bytes();
    b[..4].copy_from_slice(&a);
    b[4..].copy_from_slice(&a1);
    InternalKey {
        user_key: Vec::from(b),
        trailer: 0,
    }
}

fn make_int_key(i: usize) -> InternalKey {
    InternalKey {
        user_key: Vec::from(format!("{:05}", i)),
        trailer: 0,
    }
}

criterion_group!(
    benches,
    criterion_benchmark_skl_add,
    criterion_benchmark_skl_iter,
    criterion_read_write
);
criterion_main!(benches);
