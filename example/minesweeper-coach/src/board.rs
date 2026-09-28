//! The board as it shows on the screen: what each cell looks like, nothing
//! the player could not see.

use std::fmt;

/// What one cell shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cell {
    /// Not opened, no flag.
    Hidden,
    /// Not opened, flagged by the player (who may be wrong).
    Flag,
    /// Opened: how many of its eight neighbours are mines (0 for an empty cell).
    Open(u8),
    /// A mine the game shows once it is lost.
    Mine,
    /// The mine that was clicked: the game is lost.
    Exploded,
    /// A flag that was not on a mine, shown once the game is lost.
    WrongFlag,
}

impl Cell {
    /// Not opened (hidden or flagged): what the coach reasons about.
    pub fn is_unknown(self) -> bool {
        matches!(self, Cell::Hidden | Cell::Flag)
    }

    /// The one-character form used by [`Board::parse`] and [`Board::to_text`].
    pub fn symbol(self) -> char {
        match self {
            Cell::Hidden => '#',
            Cell::Flag => 'F',
            Cell::Open(0) => '.',
            Cell::Open(n) => (b'0' + n.min(8)) as char,
            Cell::Mine => '*',
            Cell::Exploded => 'X',
            Cell::WrongFlag => '!',
        }
    }

    pub fn from_symbol(c: char) -> Option<Cell> {
        Some(match c {
            '#' => Cell::Hidden,
            'F' | 'f' => Cell::Flag,
            '.' | '0' => Cell::Open(0),
            '1'..='8' => Cell::Open(c as u8 - b'0'),
            '*' => Cell::Mine,
            'X' | 'x' => Cell::Exploded,
            '!' => Cell::WrongFlag,
            _ => return None,
        })
    }
}

/// A rectangular board, row by row; `(x, y)` is column, row from the top left.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Board {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl Board {
    /// A board of hidden cells.
    pub fn new(width: usize, height: usize) -> Self {
        Board {
            width,
            height,
            cells: vec![Cell::Hidden; width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    pub fn position(&self, index: usize) -> (usize, usize) {
        (index % self.width, index / self.width)
    }

    pub fn get(&self, x: usize, y: usize) -> Cell {
        self.cells[self.index(x, y)]
    }

    pub fn at(&self, index: usize) -> Cell {
        self.cells[index]
    }

    pub fn set(&mut self, x: usize, y: usize, cell: Cell) {
        let i = self.index(x, y);
        self.cells[i] = cell;
    }

    /// The (up to eight) cells around `(x, y)`.
    pub fn neighbors(&self, x: usize, y: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let (w, h) = (self.width as isize, self.height as isize);
        let (x, y) = (x as isize, y as isize);
        (-1isize..=1)
            .flat_map(move |dy| (-1isize..=1).map(move |dx| (x + dx, y + dy)))
            .filter(move |&(nx, ny)| (nx, ny) != (x, y) && nx >= 0 && ny >= 0 && nx < w && ny < h)
            .map(|(nx, ny)| (nx as usize, ny as usize))
    }

    /// Every cell with its position, row by row.
    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), Cell)> + '_ {
        self.cells
            .iter()
            .enumerate()
            .map(move |(i, &c)| (self.position(i), c))
    }

    pub fn count(&self, pred: impl Fn(Cell) -> bool) -> usize {
        self.cells.iter().filter(|&&c| pred(c)).count()
    }

    /// Nothing opened yet (flags aside): a new game.
    pub fn is_fresh(&self) -> bool {
        self.cells.iter().all(|c| c.is_unknown())
    }

    /// A mine is showing: the game is over, lost.
    pub fn is_lost(&self) -> bool {
        self.cells
            .iter()
            .any(|c| matches!(c, Cell::Exploded | Cell::Mine | Cell::WrongFlag))
    }

    /// Rows of symbols: `#` hidden, `F` flag, `.` empty, `1`-`8`, `*` mine,
    /// `X` the mine that went off, `!` a wrong flag. Blank lines and
    /// surrounding spaces are ignored.
    pub fn parse(text: &str) -> Result<Board, String> {
        let rows: Vec<Vec<Cell>> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(|l| {
                l.chars()
                    .filter(|c| !c.is_whitespace())
                    .map(|c| {
                        Cell::from_symbol(c).ok_or_else(|| format!("unknown cell symbol {c:?}"))
                    })
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        let height = rows.len();
        let width = rows.first().map_or(0, Vec::len);
        if height == 0 || width == 0 {
            return Err("empty board".into());
        }
        if rows.iter().any(|r| r.len() != width) {
            return Err("rows of different lengths".into());
        }
        Ok(Board {
            width,
            height,
            cells: rows.into_iter().flatten().collect(),
        })
    }

    pub fn to_text(&self) -> String {
        let mut s = String::with_capacity((self.width + 1) * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                s.push(self.get(x, y).symbol());
            }
            s.push('\n');
        }
        s
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

/// Which sides of a board run on out of view (scrolled off, or covered): the
/// cells seen there are not the board's edge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Sides {
    pub left: bool,
    pub top: bool,
    pub right: bool,
    pub bottom: bool,
}

impl Sides {
    pub const NONE: Sides = Sides {
        left: false,
        top: false,
        right: false,
        bottom: false,
    };
    pub const ALL: Sides = Sides {
        left: true,
        top: true,
        right: true,
        bottom: true,
    };

    pub fn any(self) -> bool {
        self.left || self.top || self.right || self.bottom
    }
}

/// The standard sizes and their mine counts, for when the counter cannot be read.
pub fn standard_mines(width: usize, height: usize) -> Option<usize> {
    match (width.max(height), width.min(height)) {
        (9, 9) => Some(10),
        (16, 16) => Some(40),
        (30, 16) => Some(99),
        (8, 8) => Some(10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_round_trip() {
        let text = "#F.1\n2345\n678*\nX!##\n";
        let b = Board::parse(text).unwrap();
        assert_eq!(b.width(), 4);
        assert_eq!(b.height(), 4);
        assert_eq!(b.get(1, 0), Cell::Flag);
        assert_eq!(b.get(3, 2), Cell::Mine);
        assert_eq!(b.to_text(), text);
        assert!(b.is_lost());
    }

    #[test]
    fn neighbours_stay_on_the_board() {
        let b = Board::new(3, 2);
        assert_eq!(b.neighbors(0, 0).count(), 3);
        assert_eq!(b.neighbors(1, 0).count(), 5);
        assert_eq!(b.neighbors(1, 1).count(), 5);
    }

    #[test]
    fn bad_text_is_refused() {
        assert!(Board::parse("##\n#").is_err());
        assert!(Board::parse("#?").is_err());
        assert!(Board::parse("").is_err());
    }
}
