//! Whether the risks the solver gives are right: over many games, the guesses
//! it rated p should be mines about p of the time. Slow in a debug build:
//!   cargo test --release --test calibration -- --ignored --nocapture
use mines_coach::game::Game;
use mines_coach::play::{FirstClick, play};

#[test]
#[ignore]
fn risks_match_what_happens() {
    let mut buckets = [(0usize, 0usize, 0.0f64); 10];
    for (w, h, m, games) in [(9, 9, 10, 20000), (16, 16, 40, 3000)] {
        for g in 0..games {
            let mut game = Game::new(w, h, m, 1000 + g as u64);
            let first = if g % 2 == 0 {
                FirstClick::Centre
            } else {
                FirstClick::Corner
            };
            for (risk, mine) in play(&mut game, first).guesses {
                let b = ((risk * 10.0) as usize).min(9);
                buckets[b].0 += 1;
                buckets[b].1 += mine as usize;
                buckets[b].2 += risk;
            }
        }
    }
    let mut worst: f64 = 0.0;
    for (i, &(n, hits, sum)) in buckets.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let expected = sum;
        let sd = (sum * (1.0 - sum / n as f64)).max(1.0).sqrt();
        let z = (hits as f64 - expected) / sd;
        worst = worst.max(z.abs());
        println!(
            "risk {:.1}-{:.1}: {n:6} guesses, expected {expected:8.1} mines, hit {hits:6} (z = {z:+.2})",
            i as f64 / 10.0,
            (i + 1) as f64 / 10.0
        );
    }
    assert!(worst < 4.0, "risks off by {worst:.1} standard deviations");
}
