//! The coach's advice played out on real games, to measure it: open a proven
//! safe cell when there is one, else the least risky; every proof checked
//! against where the mines really are.

use crate::board::Cell;
use crate::game::{Game, Opened, State};
use crate::solver::{Pos, analyze};

/// Where the first click goes (the first click is always safe).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstClick {
    Corner,
    Centre,
    Edge,
}

impl FirstClick {
    pub fn cell(self, width: usize, height: usize) -> Pos {
        match self {
            FirstClick::Corner => (0, 0),
            FirstClick::Centre => (width / 2, height / 2),
            FirstClick::Edge => (width / 2, 0),
        }
    }
}

/// How one game went.
#[derive(Debug, Clone, Default)]
pub struct Played {
    pub won: bool,
    /// Moves on proven-safe cells.
    pub proven: usize,
    /// Guesses, with the risk the solver gave each and whether it was a mine.
    pub guesses: Vec<(f64, bool)>,
    /// Proofs the real mines contradicted (must stay zero).
    pub wrong_proofs: usize,
}

/// Plays one game to the end the way the coach would advise.
pub fn play(game: &mut Game, first: FirstClick) -> Played {
    let mut out = Played::default();
    let (x, y) = first.cell(game.width(), game.height());
    game.open(x, y);
    while game.state() == State::Playing {
        let board = game.view();
        let a = analyze(&board, Some(game.total_mines()));
        for d in &a.safe {
            if game.is_mine(d.cell.0, d.cell.1) {
                out.wrong_proofs += 1;
            }
        }
        for d in &a.mines {
            if !game.is_mine(d.cell.0, d.cell.1) {
                out.wrong_proofs += 1;
            }
        }
        // Proven mines get flagged, so that the view carries them (as a player would).
        for d in &a.mines {
            if board.get(d.cell.0, d.cell.1) == Cell::Hidden {
                game.toggle_flag(d.cell.0, d.cell.1);
            }
        }
        let next = a
            .safe
            .iter()
            .map(|d| d.cell)
            .find(|&(x, y)| game.view().get(x, y).is_unknown());
        let target = match next {
            Some(cell) => {
                out.proven += 1;
                cell
            }
            None => match a.guess {
                Some(g) => {
                    out.guesses.push((g.risk, game.is_mine(g.cell.0, g.cell.1)));
                    g.cell
                }
                None => break,
            },
        };
        // A proven-safe cell under the player's flag: take the flag off first.
        if game.view().get(target.0, target.1) == Cell::Flag {
            game.toggle_flag(target.0, target.1);
        }
        if game.open(target.0, target.1) == Opened::Nothing {
            break;
        }
    }
    out.won = game.state() == State::Won;
    out
}

/// Many games of one size: the win rate, and how well the risks matched what happened.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub games: usize,
    pub won: usize,
    pub guesses: usize,
    /// Sum of the risks the solver gave its guesses: the mines it expected to hit.
    pub expected_hits: f64,
    /// Mines its guesses actually hit.
    pub hits: usize,
    pub wrong_proofs: usize,
}

pub fn simulate(
    width: usize,
    height: usize,
    mines: usize,
    games: usize,
    seed: u64,
    first: FirstClick,
) -> Summary {
    let mut s = Summary {
        games,
        ..Default::default()
    };
    for g in 0..games {
        let mut game = Game::new(width, height, mines, seed.wrapping_add(g as u64 * 7919));
        let p = play(&mut game, first);
        s.won += p.won as usize;
        s.guesses += p.guesses.len();
        s.expected_hits += p.guesses.iter().map(|&(r, _)| r).sum::<f64>();
        s.hits += p.guesses.iter().filter(|&&(_, m)| m).count();
        s.wrong_proofs += p.wrong_proofs;
    }
    s
}
