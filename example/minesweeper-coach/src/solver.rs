//! What the numbers on a board prove, and how likely every other hidden cell
//! is to hide a mine.
//!
//! Flags are the player's opinion, not facts: a flagged cell is reasoned
//! about like a hidden one, so a wrong flag shows up as a proven-safe cell.
//!
//! First the rules a person uses, each with its reason so the coach can say
//! why: a number that already touches all its mines clears its other
//! neighbours; a number with exactly as many hidden neighbours as mines left
//! fills them; and two numbers whose neighbourhoods overlap bound how many
//! mines the overlap holds, which can settle the rest of either. Then every
//! way the remaining numbers can be satisfied is counted — per group of
//! hidden cells that share numbers, weighted by how many ways the mines left
//! over fit the cells no number touches — which gives each cell's chance of
//! being a mine and proves the cells whose chance is 0 or 1.

use crate::board::{Board, Cell, Sides};

pub type Pos = (usize, usize);

/// Why a cell is proven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The number at `by` already touches all its mines: its other hidden neighbours are safe.
    Satisfied { by: Pos },
    /// The number at `by` has exactly as many hidden neighbours as mines left: they are all mines.
    Filled { by: Pos },
    /// Two numbers together: what `a` says about the cells it shares with `b` settles `b`'s others.
    Pair { a: Pos, b: Pos },
    /// Only by counting every way the numbers (and the mines left) can be satisfied.
    Counting,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deduction {
    pub cell: Pos,
    pub reason: Reason,
}

/// The least risky cell, when nothing is proven safe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Guess {
    pub cell: Pos,
    /// Chance it is a mine.
    pub risk: f64,
    /// How many other cells share that same lowest risk.
    pub ties: usize,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    width: usize,
    /// Hidden or flagged cells proven safe, in the order they were found.
    pub safe: Vec<Deduction>,
    /// Hidden or flagged cells proven to be mines.
    pub mines: Vec<Deduction>,
    /// Row by row: each unopened cell's chance of being a mine (0 or 1 when
    /// proven); `None` for open cells.
    pub risk: Vec<Option<f64>>,
    pub guess: Option<Guess>,
    /// The numbers cannot all be right: a misread board (or a wrong total).
    pub contradiction: bool,
    /// The chances are exact: every group was counted out and the mine total known.
    pub exact: bool,
}

impl Analysis {
    pub fn risk_at(&self, (x, y): Pos) -> Option<f64> {
        self.risk[y * self.width + x]
    }

    pub fn safe_at(&self, p: Pos) -> Option<&Deduction> {
        self.safe.iter().find(|d| d.cell == p)
    }

    pub fn mine_at(&self, p: Pos) -> Option<&Deduction> {
        self.mines.iter().find(|d| d.cell == p)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Know {
    Open,
    Unknown,
    Safe(Reason),
    Mine(Reason),
    /// A mine the game shows (a lost game).
    Shown,
}

struct Constraint {
    at: Pos,
    need: i32,
    cells: Vec<usize>,
}

/// Mines left over when the total is not known: roughly the density of the
/// standard levels.
const DEFAULT_DENSITY: f64 = 0.16;
/// Enumeration steps allowed per group before its chances are estimated instead.
const STEP_BUDGET: u64 = 3_000_000;

pub fn analyze(board: &Board, total_mines: Option<usize>) -> Analysis {
    let (w, h) = (board.width(), board.height());
    let n = w * h;
    let mut know: Vec<Know> = (0..n)
        .map(|i| match board.at(i) {
            Cell::Hidden | Cell::Flag => Know::Unknown,
            Cell::Open(_) | Cell::WrongFlag => Know::Open,
            Cell::Mine | Cell::Exploded => Know::Shown,
        })
        .collect();
    let mut contradiction = false;
    let mut safe = Vec::new();
    let mut mines = Vec::new();

    let record = |know: &mut Vec<Know>,
                  i: usize,
                  k: Know,
                  safe: &mut Vec<Deduction>,
                  mines: &mut Vec<Deduction>| {
        if know[i] != Know::Unknown {
            return false;
        }
        know[i] = k;
        let cell = (i % w, i / w);
        match k {
            Know::Safe(reason) => safe.push(Deduction { cell, reason }),
            Know::Mine(reason) => mines.push(Deduction { cell, reason }),
            _ => {}
        }
        true
    };

    // ---- the rules a person uses ----------------------------------------------------------------
    loop {
        let (cons, bad) = constraints(board, &know);
        contradiction |= bad;
        let mut progress = false;
        for c in &cons {
            if c.cells.is_empty() {
                continue;
            }
            if c.need == 0 {
                for &i in &c.cells {
                    progress |= record(
                        &mut know,
                        i,
                        Know::Safe(Reason::Satisfied { by: c.at }),
                        &mut safe,
                        &mut mines,
                    );
                }
            } else if c.need as usize == c.cells.len() {
                for &i in &c.cells {
                    progress |= record(
                        &mut know,
                        i,
                        Know::Mine(Reason::Filled { by: c.at }),
                        &mut safe,
                        &mut mines,
                    );
                }
            }
        }
        if progress {
            continue;
        }
        'pairs: for a in &cons {
            for b in &cons {
                if a.at == b.at || a.at.0.abs_diff(b.at.0) > 2 || a.at.1.abs_diff(b.at.1) > 2 {
                    continue;
                }
                let inter: Vec<usize> = a
                    .cells
                    .iter()
                    .copied()
                    .filter(|i| b.cells.contains(i))
                    .collect();
                if inter.is_empty() {
                    continue;
                }
                let a_only = a.cells.len() - inter.len();
                let b_only: Vec<usize> = b
                    .cells
                    .iter()
                    .copied()
                    .filter(|i| !inter.contains(i))
                    .collect();
                if b_only.is_empty() {
                    continue;
                }
                let k = inter.len() as i32;
                let most = k.min(a.need).min(b.need);
                let least = 0
                    .max(a.need - a_only as i32)
                    .max(b.need - b_only.len() as i32);
                if least > most {
                    contradiction = true;
                    continue;
                }
                let reason = Reason::Pair { a: a.at, b: b.at };
                if b.need - least == 0 {
                    for &i in &b_only {
                        progress |= record(&mut know, i, Know::Safe(reason), &mut safe, &mut mines);
                    }
                } else if b.need - most == b_only.len() as i32 {
                    for &i in &b_only {
                        progress |= record(&mut know, i, Know::Mine(reason), &mut safe, &mut mines);
                    }
                }
                if progress {
                    break 'pairs;
                }
            }
        }
        if !progress {
            break;
        }
    }

    // ---- counting every way the rest can be ------------------------------------------------------
    let (cons, bad) = constraints(board, &know);
    contradiction |= bad;
    let mut comp_of = vec![usize::MAX; n];
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        let mut j = i;
        while p[j] != r {
            let next = p[j];
            p[j] = r;
            j = next;
        }
        r
    }
    let mut on_frontier = vec![false; n];
    for c in &cons {
        for &i in &c.cells {
            on_frontier[i] = true;
        }
        for pair in c.cells.windows(2) {
            let (ra, rb) = (find(&mut parent, pair[0]), find(&mut parent, pair[1]));
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, _) in on_frontier.iter().enumerate().filter(|&(_, &f)| f) {
        let r = find(&mut parent, i);
        if comp_of[r] == usize::MAX {
            comp_of[r] = groups.len();
            groups.push(Vec::new());
        }
        groups[comp_of[r]].push(i);
    }
    let interior: Vec<usize> = (0..n)
        .filter(|&i| know[i] == Know::Unknown && !on_frontier[i])
        .collect();
    let known_mines = know
        .iter()
        .filter(|k| matches!(k, Know::Mine(_) | Know::Shown))
        .count();

    let counted: Vec<Counted> = groups
        .iter()
        .map(|cells| {
            let group_cons: Vec<&Constraint> = cons
                .iter()
                .filter(|c| c.cells.iter().any(|i| cells.contains(i)))
                .collect();
            count_group(cells, &group_cons)
        })
        .collect();
    let mut exact = total_mines.is_some() && counted.iter().all(|c| c.exact);
    let mut risk: Vec<Option<f64>> = vec![None; n];
    let left = total_mines.map(|t| t as i64 - known_mines as i64);
    if left.is_some_and(|l| {
        l < 0 || l > (interior.len() + groups.iter().map(Vec::len).sum::<usize>()) as i64
    }) {
        contradiction = true;
    }

    match left {
        Some(left) if left >= 0 && counted.iter().all(|c| c.exact) => {
            // Weights of each group's mine count, the leave-one-out products, and the
            // ways the rest fit the interior: chances exact up to rounding.
            let o = interior.len();
            let lnf = ln_factorials(n + 1);
            let ln_choose = |k: i64| -> Option<f64> {
                if k < 0 || k as usize > o {
                    None
                } else {
                    let k = k as usize;
                    Some(lnf[o] - lnf[k] - lnf[o - k])
                }
            };
            let dists: Vec<Vec<f64>> = counted.iter().map(|c| normalised(&c.totals)).collect();
            let all = dists.iter().fold(vec![1.0], |acc, d| convolve(&acc, d));
            // scale of the binomials: the largest over the reachable totals
            let max_ln = (0..all.len())
                .filter_map(|m| ln_choose(left - m as i64))
                .fold(f64::NEG_INFINITY, f64::max);
            if max_ln.is_finite() {
                let rest = |m: i64| ln_choose(left - m).map_or(0.0, |l| (l - max_ln).exp());
                let den: f64 = all
                    .iter()
                    .enumerate()
                    .map(|(m, &p)| p * rest(m as i64))
                    .sum();
                if den > 0.0 {
                    for (gi, c) in counted.iter().enumerate() {
                        let excl = dists
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != gi)
                            .fold(vec![1.0], |acc, (_, d)| convolve(&acc, d));
                        let scale = c.totals.iter().cloned().fold(0.0, f64::max).max(1e-300);
                        for (k, &cell) in c.cells.iter().enumerate() {
                            let mut num = 0.0;
                            for (m, per) in c.per_cell.iter().enumerate() {
                                if per[k] == 0.0 {
                                    continue;
                                }
                                let tail: f64 = excl
                                    .iter()
                                    .enumerate()
                                    .map(|(mm, &p)| p * rest((m + mm) as i64))
                                    .sum();
                                num += per[k] / scale * tail;
                            }
                            risk[cell] = Some(if num == 0.0 {
                                0.0
                            } else {
                                (num / den).min(1.0)
                            });
                        }
                    }
                    if o > 0 {
                        let expected: f64 = all
                            .iter()
                            .enumerate()
                            .map(|(m, &p)| p * rest(m as i64) * (left - m as i64).max(0) as f64)
                            .sum::<f64>()
                            / den;
                        let p = (expected / o as f64).clamp(0.0, 1.0);
                        for &i in &interior {
                            risk[i] = Some(p);
                        }
                    }
                } else {
                    contradiction = true;
                    exact = false;
                }
            } else {
                contradiction = true;
                exact = false;
            }
        }
        _ => {
            exact = false;
            for c in &counted {
                let total: f64 = c.totals.iter().sum();
                for (k, &cell) in c.cells.iter().enumerate() {
                    risk[cell] = Some(if !c.exact {
                        c.estimate[k]
                    } else if total > 0.0 {
                        c.per_cell.iter().map(|per| per[k]).sum::<f64>() / total
                    } else {
                        0.5
                    });
                }
            }
            let density = match left {
                Some(l) if !interior.is_empty() => {
                    let frontier_expected: f64 = groups
                        .iter()
                        .flatten()
                        .map(|&i| risk[i].unwrap_or(0.0))
                        .sum();
                    ((l as f64 - frontier_expected) / interior.len() as f64).clamp(0.0, 1.0)
                }
                _ => DEFAULT_DENSITY,
            };
            for &i in &interior {
                risk[i] = Some(density);
            }
        }
    }

    // Counted certainties become deductions; proven cells get 0 or 1.
    for i in 0..n {
        if know[i] == Know::Unknown {
            match risk[i] {
                Some(p)
                    if p == 0.0
                        && (exact || on_frontier[i])
                        && groups_exact(&counted, &comp_of, &mut parent, i) =>
                {
                    record(
                        &mut know,
                        i,
                        Know::Safe(Reason::Counting),
                        &mut safe,
                        &mut mines,
                    );
                }
                Some(p)
                    if (p - 1.0).abs() < 1e-12
                        && (exact || on_frontier[i])
                        && groups_exact(&counted, &comp_of, &mut parent, i) =>
                {
                    record(
                        &mut know,
                        i,
                        Know::Mine(Reason::Counting),
                        &mut safe,
                        &mut mines,
                    );
                }
                _ => {}
            }
        }
        risk[i] = match know[i] {
            Know::Open => None,
            Know::Safe(_) => Some(0.0),
            Know::Mine(_) | Know::Shown => Some(1.0),
            Know::Unknown => risk[i].or(Some(DEFAULT_DENSITY)),
        };
    }

    let unknown: Vec<usize> = (0..n).filter(|&i| know[i] == Know::Unknown).collect();
    let guess = if safe.is_empty() && !board.is_fresh() && !board.is_lost() {
        best_guess(board, &risk, &unknown)
    } else {
        None
    };
    let _ = h;
    Analysis {
        width: w,
        safe,
        mines,
        risk,
        guess,
        contradiction,
        exact,
    }
}

/// The same for a board seen only in part: past each open side of it there are
/// more cells, unseen, so the last cells seen are not taken for the board's
/// edge — a number there may have its mine out of view — and the mine total
/// says nothing about the part seen. Everything returned is about the cells
/// seen.
pub fn analyze_seen(board: &Board, cut: Sides, total_mines: Option<usize>) -> Analysis {
    if !cut.any() {
        return analyze(board, total_mines);
    }
    let (l, t) = (cut.left as usize, cut.top as usize);
    let (w, h) = (board.width(), board.height());
    // One row or column of unseen cells past each open side is enough: no number seen touches more.
    let mut padded = Board::new(w + l + cut.right as usize, h + t + cut.bottom as usize);
    for ((x, y), c) in board.iter() {
        padded.set(x + l, y + t, c);
    }
    let a = analyze(&padded, None);
    let seen = |(x, y): Pos| -> Option<Pos> {
        (x >= l && y >= t && x - l < w && y - t < h).then(|| (x - l, y - t))
    };
    let map = |ds: &[Deduction]| -> Vec<Deduction> {
        ds.iter()
            .filter_map(|d| {
                let reason = match d.reason {
                    Reason::Satisfied { by } => Reason::Satisfied { by: seen(by)? },
                    Reason::Filled { by } => Reason::Filled { by: seen(by)? },
                    Reason::Pair { a, b } => Reason::Pair {
                        a: seen(a)?,
                        b: seen(b)?,
                    },
                    Reason::Counting => Reason::Counting,
                };
                Some(Deduction {
                    cell: seen(d.cell)?,
                    reason,
                })
            })
            .collect()
    };
    let (safe, mines) = (map(&a.safe), map(&a.mines));
    let mut risk = vec![None; board.len()];
    for ((x, y), _) in board.iter() {
        risk[board.index(x, y)] = a.risk_at((x + l, y + t));
    }
    let proven = |i: usize| {
        let p = board.position(i);
        safe.iter().chain(&mines).any(|d| d.cell == p)
    };
    let unknown: Vec<usize> = (0..board.len())
        .filter(|&i| board.at(i).is_unknown() && !proven(i))
        .collect();
    let guess = if safe.is_empty() && !board.is_fresh() && !board.is_lost() {
        best_guess(board, &risk, &unknown)
    } else {
        None
    };
    Analysis {
        width: w,
        safe,
        mines,
        risk,
        guess,
        contradiction: a.contradiction,
        exact: false,
    }
}

/// The least risky of the undecided cells; among equals, the one with fewest
/// hidden neighbours (it tells the most when it opens).
fn best_guess(board: &Board, risk: &[Option<f64>], unknown: &[usize]) -> Option<Guess> {
    let w = board.width();
    let best = unknown
        .iter()
        .filter_map(|&i| risk[i].map(|p| (i, p)))
        .fold(None, |acc: Option<(usize, f64)>, (i, p)| match acc {
            None => Some((i, p)),
            Some((j, q)) => {
                if p < q - 1e-9
                    || ((p - q).abs() <= 1e-9 && hidden_around(board, i) < hidden_around(board, j))
                {
                    Some((i, p))
                } else {
                    Some((j, q))
                }
            }
        });
    best.map(|(i, p)| Guess {
        cell: (i % w, i / w),
        risk: p,
        ties: unknown
            .iter()
            .filter(|&&j| j != i && risk[j].is_some_and(|q| (q - p).abs() <= 1e-9))
            .count(),
    })
}

/// Whether the group a frontier cell belongs to was counted out (interior cells: always).
fn groups_exact(counted: &[Counted], comp_of: &[usize], parent: &mut [usize], i: usize) -> bool {
    let mut r = i;
    while parent[r] != r {
        r = parent[r];
    }
    match comp_of.get(r) {
        Some(&g) if g != usize::MAX => counted[g].exact,
        _ => true,
    }
}

fn hidden_around(board: &Board, i: usize) -> usize {
    let (x, y) = board.position(i);
    board
        .neighbors(x, y)
        .filter(|&(nx, ny)| board.get(nx, ny).is_unknown())
        .count()
}

/// Every open number with its undecided neighbours and the mines it still needs.
fn constraints(board: &Board, know: &[Know]) -> (Vec<Constraint>, bool) {
    let mut out = Vec::new();
    let mut bad = false;
    for ((x, y), cell) in board.iter() {
        let Cell::Open(num) = cell else { continue };
        let mut cells = Vec::new();
        let mut mines = 0i32;
        for (nx, ny) in board.neighbors(x, y) {
            let j = board.index(nx, ny);
            match know[j] {
                Know::Unknown => cells.push(j),
                Know::Mine(_) | Know::Shown => mines += 1,
                _ => {}
            }
        }
        let need = num as i32 - mines;
        if need < 0 || need as usize > cells.len() {
            bad = true;
            continue;
        }
        if !cells.is_empty() {
            out.push(Constraint {
                at: (x, y),
                need,
                cells,
            });
        }
    }
    (out, bad)
}

/// A group's solutions, by how many mines they put in it.
struct Counted {
    cells: Vec<usize>,
    /// totals[m]: solutions with m mines in the group.
    totals: Vec<f64>,
    /// per_cell[m][k]: of those, how many have a mine on the group's k-th cell.
    per_cell: Vec<Vec<f64>>,
    exact: bool,
    /// When the count ran out of budget: each cell's chance estimated from its numbers.
    estimate: Vec<f64>,
}

fn count_group(cells: &[usize], cons: &[&Constraint]) -> Counted {
    let k = cells.len();
    let local = |i: usize| cells.iter().position(|&c| c == i);
    // Constraints in local terms; cells ordered so that constraints fill up early.
    let lcons: Vec<(i32, Vec<usize>)> = cons
        .iter()
        .map(|c| (c.need, c.cells.iter().filter_map(|&i| local(i)).collect()))
        .collect();
    let mut order: Vec<usize> = Vec::with_capacity(k);
    let mut placed = vec![false; k];
    for (_, cs) in &lcons {
        for &c in cs {
            if !placed[c] {
                placed[c] = true;
                order.push(c);
            }
        }
    }
    order.extend((0..k).filter(|&c| !placed[c]));
    let mut cons_of: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (ci, (_, cs)) in lcons.iter().enumerate() {
        for &c in cs {
            cons_of[c].push(ci);
        }
    }
    let mut assigned = vec![0i32; lcons.len()];
    let mut open_left: Vec<i32> = lcons.iter().map(|(_, cs)| cs.len() as i32).collect();
    let mut value = vec![0u8; k];
    let mut totals = vec![0.0; k + 1];
    let mut per_cell = vec![vec![0.0; k]; k + 1];
    let mut steps = 0u64;

    #[allow(clippy::too_many_arguments)]
    fn go(
        depth: usize,
        mines: usize,
        order: &[usize],
        cons_of: &[Vec<usize>],
        lcons: &[(i32, Vec<usize>)],
        assigned: &mut [i32],
        open_left: &mut [i32],
        value: &mut [u8],
        totals: &mut [f64],
        per_cell: &mut [Vec<f64>],
        steps: &mut u64,
    ) -> bool {
        *steps += 1;
        if *steps > STEP_BUDGET {
            return false;
        }
        if depth == order.len() {
            totals[mines] += 1.0;
            for (c, &v) in value.iter().enumerate() {
                if v == 1 {
                    per_cell[mines][c] += 1.0;
                }
            }
            return true;
        }
        let c = order[depth];
        for v in [0u8, 1u8] {
            let mut ok = true;
            for &ci in &cons_of[c] {
                assigned[ci] += v as i32;
                open_left[ci] -= 1;
            }
            for &ci in &cons_of[c] {
                let need = lcons[ci].0;
                if assigned[ci] > need || assigned[ci] + open_left[ci] < need {
                    ok = false;
                    break;
                }
            }
            if ok {
                value[c] = v;
                if !go(
                    depth + 1,
                    mines + v as usize,
                    order,
                    cons_of,
                    lcons,
                    assigned,
                    open_left,
                    value,
                    totals,
                    per_cell,
                    steps,
                ) {
                    for &ci in &cons_of[c] {
                        assigned[ci] -= v as i32;
                        open_left[ci] += 1;
                    }
                    return false;
                }
                value[c] = 0;
            }
            for &ci in &cons_of[c] {
                assigned[ci] -= v as i32;
                open_left[ci] += 1;
            }
        }
        true
    }

    let exact = go(
        0,
        0,
        &order,
        &cons_of,
        &lcons,
        &mut assigned,
        &mut open_left,
        &mut value,
        &mut totals,
        &mut per_cell,
        &mut steps,
    );
    let estimate = if exact {
        Vec::new()
    } else {
        (0..k)
            .map(|c| {
                let shares: Vec<f64> = cons_of[c]
                    .iter()
                    .map(|&ci| lcons[ci].0 as f64 / lcons[ci].1.len() as f64)
                    .collect();
                if shares.is_empty() {
                    DEFAULT_DENSITY
                } else {
                    shares.iter().sum::<f64>() / shares.len() as f64
                }
            })
            .collect()
    };
    Counted {
        cells: cells.to_vec(),
        totals,
        per_cell,
        exact,
        estimate,
    }
}

fn ln_factorials(n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n + 1];
    for i in 1..=n {
        v[i] = v[i - 1] + (i as f64).ln();
    }
    v
}

fn normalised(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(0.0, f64::max);
    if max > 0.0 {
        v.iter().map(|x| x / max).collect()
    } else {
        v.to_vec()
    }
}

fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(a: &Analysis, p: Pos) -> f64 {
        a.risk_at(p).unwrap()
    }

    #[test]
    fn an_empty_cell_clears_and_a_full_number_fills() {
        let b = Board::parse("1#\n1#\n..").unwrap();
        let a = analyze(&b, None);
        assert!(
            matches!(a.safe_at((1, 1)).map(|d| d.reason), Some(Reason::Satisfied { by }) if by.1 == 2),
            "{:?}",
            a.safe
        );
        assert!(
            matches!(
                a.mine_at((1, 0)).map(|d| d.reason),
                Some(Reason::Filled { .. })
            ),
            "{:?}",
            a.mines
        );
        assert_eq!(at(&a, (1, 0)), 1.0);
        assert!(a.guess.is_none() && !a.contradiction);
    }

    #[test]
    fn a_fifty_fifty_is_a_guess_with_a_tie() {
        let a = analyze(&Board::parse("#1\n#1").unwrap(), Some(1));
        assert!(a.safe.is_empty() && a.mines.is_empty());
        assert!((at(&a, (0, 0)) - 0.5).abs() < 1e-9 && (at(&a, (0, 1)) - 0.5).abs() < 1e-9);
        let g = a.guess.unwrap();
        assert!((g.risk - 0.5).abs() < 1e-9 && g.ties == 1);
    }

    #[test]
    fn simple_rules_with_their_reasons() {
        let b = Board::parse(
            "
            ##1.
            ##1.
            111.
            ....",
        )
        .unwrap();
        let a = analyze(&b, Some(1));
        assert_eq!(
            a.mine_at((1, 1)).map(|d| d.reason),
            Some(Reason::Filled { by: (2, 2) })
        );
        assert_eq!(
            a.safe_at((1, 0)).map(|d| d.reason),
            Some(Reason::Satisfied { by: (2, 0) })
        );
        assert_eq!(
            a.safe_at((0, 1)).map(|d| d.reason),
            Some(Reason::Satisfied { by: (0, 2) })
        );
        // No number touches the corner: only the count (one mine, already found) clears it.
        assert_eq!(a.safe_at((0, 0)).map(|d| d.reason), Some(Reason::Counting));
        assert!(a.exact && !a.contradiction);
    }

    #[test]
    fn two_numbers_together_settle_a_cell() {
        // Mines at (0,0) and (3,0) under a row of 1s: the 1 at (0,1) sees two cells, the 1 at
        // (1,1) the same two and one more, which must then be safe.
        let b = Board::parse(
            "
            #####
            11111
            .....",
        )
        .unwrap();
        let a = analyze(&b, Some(2));
        let d = a.safe_at((2, 0)).expect("the middle cell is safe");
        assert!(matches!(d.reason, Reason::Pair { .. }), "{:?}", d.reason);
        assert!(
            a.safe.len() == 1 && a.mines.is_empty(),
            "{:?} {:?}",
            a.safe,
            a.mines
        );
        for x in [0, 1, 3, 4] {
            assert!(
                (at(&a, (x, 0)) - 0.5).abs() < 1e-9,
                "({x},0): {}",
                at(&a, (x, 0))
            );
        }
    }

    #[test]
    fn a_flag_on_a_safe_cell_is_proven_safe() {
        let b = Board::parse(
            "
            #1.
            F1.
            ...",
        )
        .unwrap();
        let a = analyze(&b, None);
        assert!(
            a.safe_at((0, 1)).is_some(),
            "the flagged cell is safe: {:?}",
            a.safe
        );
        assert!(a.mine_at((0, 0)).is_some());
    }

    #[test]
    fn counting_uses_the_mines_left() {
        let b = Board::parse("#1.1#\n#1.1#").unwrap();
        let two = analyze(&b, Some(2));
        assert!((at(&two, (0, 0)) - 0.5).abs() < 1e-9 && two.exact && !two.contradiction);
        assert!(analyze(&b, Some(3)).contradiction);
    }

    #[test]
    fn cells_no_number_touches_share_what_is_left() {
        let b = Board::parse(
            "
            ####
            ####
            #1..
            #1..",
        )
        .unwrap();
        let a = analyze(&b, Some(3));
        assert!(a.exact);
        assert!(a.safe_at((1, 1)).is_some() && a.safe_at((3, 1)).is_some());
        assert!(matches!(
            a.safe_at((0, 1)).map(|d| d.reason),
            Some(Reason::Pair { .. })
        ));
        let expected: f64 = a.risk.iter().flatten().sum();
        assert!((expected - 3.0).abs() < 1e-9, "expected mines {expected}");
        assert!((at(&a, (0, 0)) - 0.5).abs() < 1e-9 && (at(&a, (0, 2)) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_fresh_board_has_no_guess_and_no_proof() {
        let a = analyze(&Board::new(9, 9), Some(10));
        assert!(a.safe.is_empty() && a.mines.is_empty() && a.guess.is_none());
        assert!((at(&a, (4, 4)) - 10.0 / 81.0).abs() < 1e-9);
    }

    /// Games stopped part way and seen through a window that cuts them: what the
    /// part seen proves must hold on the whole board, and (to show the cut
    /// matters) taking the window's edge for the board's edge must go wrong
    /// somewhere.
    #[test]
    fn a_board_seen_in_part_proves_only_what_holds() {
        use crate::game::Game;
        let mut rng = crate::game::Rng::new(7);
        let (mut proofs, mut wrong_if_edge) = (0, 0);
        for seed in 0..120u64 {
            let mut game = Game::intermediate(seed);
            game.open(0, 0);
            for _ in 0..1 + rng.below(30) {
                let a = analyze(&game.view(), Some(40));
                match a
                    .safe
                    .iter()
                    .map(|d| d.cell)
                    .find(|&(x, y)| game.view().get(x, y) == Cell::Hidden)
                {
                    Some((x, y)) => {
                        game.open(x, y);
                    }
                    None => break,
                }
            }
            if game.state() != crate::game::State::Playing {
                continue;
            }
            let view = game.view();
            // A window on it: some rows or columns cut off one side or two.
            let (w, h) = (view.width(), view.height());
            let cut = Sides {
                left: rng.below(2) == 0,
                top: rng.below(3) == 0,
                right: rng.below(2) == 0,
                bottom: rng.below(3) == 0,
            };
            if !cut.any() {
                continue;
            }
            let (x0, y0) = (
                if cut.left { 1 + rng.below(4) } else { 0 },
                if cut.top { 1 + rng.below(4) } else { 0 },
            );
            let (x1, y1) = (
                if cut.right { w - 1 - rng.below(4) } else { w },
                if cut.bottom { h - 1 - rng.below(4) } else { h },
            );
            let mut part = Board::new(x1 - x0, y1 - y0);
            for y in y0..y1 {
                for x in x0..x1 {
                    part.set(x - x0, y - y0, view.get(x, y));
                }
            }
            let a = analyze_seen(&part, cut, Some(40));
            for d in &a.safe {
                proofs += 1;
                assert!(
                    !game.is_mine(d.cell.0 + x0, d.cell.1 + y0),
                    "game {seed}: {:?} proven safe is a mine\n{part}",
                    d
                );
            }
            for d in &a.mines {
                proofs += 1;
                assert!(
                    game.is_mine(d.cell.0 + x0, d.cell.1 + y0),
                    "game {seed}: {:?} proven a mine is not\n{part}",
                    d
                );
            }
            let naive = analyze(&part, None);
            wrong_if_edge += naive
                .safe
                .iter()
                .filter(|d| game.is_mine(d.cell.0 + x0, d.cell.1 + y0))
                .count();
            wrong_if_edge += naive
                .mines
                .iter()
                .filter(|d| !game.is_mine(d.cell.0 + x0, d.cell.1 + y0))
                .count();
        }
        eprintln!(
            "{proofs} proofs checked; taking the window's edge for the board's: {wrong_if_edge} wrong"
        );
        assert!(proofs > 300, "only {proofs} proofs checked");
        assert!(
            wrong_if_edge > 0,
            "the window's edge never misled: the test proves nothing"
        );
    }

    #[test]
    fn a_misread_board_is_a_contradiction() {
        assert!(analyze(&Board::parse("3#\n..").unwrap(), None).contradiction);
    }
}
