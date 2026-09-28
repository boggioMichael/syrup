//! A Minesweeper game, for the coach's tests and for measuring its advice:
//! mines placed after the first click (which is always safe, as on
//! minesweeper.online), cells opened with the usual flood of empty cells.

use crate::board::{Board, Cell};

/// A small deterministic generator (xorshift64*): games repeat from a seed.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 11) as u128 * n as u128) >> 53) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Playing,
    Won,
    Lost,
}

/// What opening a cell did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opened {
    /// Safe: this many cells opened (the flood of an empty cell counts them all).
    Safe(usize),
    Boom,
    /// Already open, flagged, or the game is over: nothing happened.
    Nothing,
}

#[derive(Debug, Clone)]
pub struct Game {
    width: usize,
    height: usize,
    total: usize,
    mines: Vec<bool>,
    open: Vec<bool>,
    flag: Vec<bool>,
    exploded: Option<usize>,
    placed: bool,
    state: State,
    rng: Rng,
}

impl Game {
    pub fn new(width: usize, height: usize, mines: usize, seed: u64) -> Self {
        assert!(mines < width * height, "more mines than cells");
        let n = width * height;
        Game {
            width,
            height,
            total: mines,
            mines: vec![false; n],
            open: vec![false; n],
            flag: vec![false; n],
            exploded: None,
            placed: false,
            state: State::Playing,
            rng: Rng::new(seed),
        }
    }

    pub fn beginner(seed: u64) -> Self {
        Game::new(9, 9, 10, seed)
    }

    pub fn intermediate(seed: u64) -> Self {
        Game::new(16, 16, 40, seed)
    }

    pub fn expert(seed: u64) -> Self {
        Game::new(30, 16, 99, seed)
    }

    /// A game with the mines where the text's `*` are (the rest of the cells
    /// hidden): for tests that need an exact position.
    pub fn with_mines(width: usize, height: usize, mines: &[(usize, usize)]) -> Self {
        let mut g = Game::new(width, height, mines.len(), 1);
        for &(x, y) in mines {
            g.mines[y * width + x] = true;
        }
        g.placed = true;
        g
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn total_mines(&self) -> usize {
        self.total
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn is_mine(&self, x: usize, y: usize) -> bool {
        self.mines[y * self.width + x]
    }

    fn place(&mut self, safe: usize) {
        let n = self.width * self.height;
        let mut free: Vec<usize> = (0..n).filter(|&i| i != safe).collect();
        for k in 0..self.total {
            let j = k + self.rng.below(free.len() - k);
            free.swap(k, j);
            self.mines[free[k]] = true;
        }
        self.placed = true;
    }

    fn neighbours(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let (w, h) = (self.width as isize, self.height as isize);
        let (x, y) = ((i % self.width) as isize, (i / self.width) as isize);
        (-1isize..=1)
            .flat_map(move |dy| (-1isize..=1).map(move |dx| (x + dx, y + dy)))
            .filter(move |&(nx, ny)| (nx, ny) != (x, y) && nx >= 0 && ny >= 0 && nx < w && ny < h)
            .map(move |(nx, ny)| ny as usize * w as usize + nx as usize)
    }

    /// How many mines touch this cell.
    pub fn number(&self, x: usize, y: usize) -> u8 {
        let i = y * self.width + x;
        self.neighbours(i).filter(|&j| self.mines[j]).count() as u8
    }

    pub fn open(&mut self, x: usize, y: usize) -> Opened {
        let i = y * self.width + x;
        if self.state != State::Playing || self.open[i] || self.flag[i] {
            return Opened::Nothing;
        }
        if !self.placed {
            self.place(i);
        }
        if self.mines[i] {
            self.exploded = Some(i);
            self.state = State::Lost;
            return Opened::Boom;
        }
        let mut stack = vec![i];
        let mut count = 0;
        while let Some(j) = stack.pop() {
            if self.open[j] || self.flag[j] {
                continue;
            }
            self.open[j] = true;
            count += 1;
            let (jx, jy) = (j % self.width, j / self.width);
            if self.number(jx, jy) == 0 {
                stack.extend(
                    self.neighbours(j)
                        .filter(|&k| !self.open[k] && !self.mines[k]),
                );
            }
        }
        let opened = self.open.iter().filter(|&&o| o).count();
        if opened + self.total == self.width * self.height {
            self.state = State::Won;
        }
        Opened::Safe(count)
    }

    pub fn toggle_flag(&mut self, x: usize, y: usize) {
        let i = y * self.width + x;
        if self.state == State::Playing && !self.open[i] {
            self.flag[i] = !self.flag[i];
        }
    }

    /// What the player sees, as the screen would show it.
    pub fn view(&self) -> Board {
        let mut b = Board::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let i = y * self.width + x;
                let cell = if self.open[i] {
                    Cell::Open(self.number(x, y))
                } else if self.state == State::Lost && Some(i) == self.exploded {
                    Cell::Exploded
                } else if self.state == State::Lost && self.flag[i] && !self.mines[i] {
                    Cell::WrongFlag
                } else if self.state == State::Lost && self.mines[i] && !self.flag[i] {
                    Cell::Mine
                } else if self.flag[i] || (self.state == State::Won && self.mines[i]) {
                    Cell::Flag
                } else {
                    Cell::Hidden
                };
                b.set(x, y, cell);
            }
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_click_is_safe_and_floods_empty_cells() {
        for seed in 0..50 {
            let mut g = Game::beginner(seed);
            match g.open(4, 4) {
                Opened::Safe(n) => assert!(n >= 1),
                other => panic!("first click: {other:?}"),
            }
            assert_eq!(g.mines.iter().filter(|&&m| m).count(), 10);
        }
    }

    #[test]
    fn opening_every_safe_cell_wins() {
        let mut g = Game::with_mines(3, 3, &[(0, 0)]);
        for y in 0..3 {
            for x in 0..3 {
                if (x, y) != (0, 0) {
                    g.open(x, y);
                }
            }
        }
        assert_eq!(g.state(), State::Won);
        assert_eq!(g.view().to_text(), "F1.\n11.\n...\n");
    }

    #[test]
    fn a_mine_loses_and_shows_the_board() {
        let mut g = Game::with_mines(3, 1, &[(0, 0), (2, 0)]);
        g.toggle_flag(2, 0);
        g.toggle_flag(1, 0);
        g.toggle_flag(1, 0);
        assert_eq!(g.open(0, 0), Opened::Boom);
        assert_eq!(g.view().to_text(), "X#F\n");
    }
}
