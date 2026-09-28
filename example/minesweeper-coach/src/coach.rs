//! What to say, and what to mark on the board, as a game goes on.
//!
//! The coach is fed what the reader saw, frame by frame, and waits for the
//! board to hold still — a cell held under the mouse, or one mid-animation,
//! is not a move. Then it works the position out (the solver) and picks one
//! thing to say, the most important first: how the game ended; a new game; a
//! flag that is wrong; that part of the board is out of view; a gamble taken
//! while a sure move was there; the next sure move and why it is sure; or,
//! when nothing is sure, the best odds. It marks every sure cell on the
//! board, and speaks only when there is something new. While the player goes
//! from one easy sure move to the next, it keeps quiet — the marks say it —
//! unless they stop for a while; it speaks up again for a move that takes two
//! numbers together, or the mine count, to see.
//!
//! Time is whatever the caller says it is (seconds), so all of this runs the
//! same in tests, on recorded frames, and live.

use std::collections::HashSet;

use crate::board::{Board, Cell, Sides, standard_mines};
use crate::reader::Reading;
use crate::solver::{Analysis, Pos, Reason, analyze_seen};

/// How much the coach talks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Talk {
    /// Guesses, mistakes and the ends of games; sure moves only on the board.
    Quiet,
    /// Also why each new sure move is sure, but not over itself.
    Normal,
    /// Every change, proven mines too.
    Chatty,
}

#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub talk: Talk,
    /// How long the board must hold still to count, in seconds.
    pub settle: f64,
    /// Least time between two lines that are not urgent, in seconds.
    pub gap: f64,
    /// How long without a board before the marks go, in seconds.
    pub gone_after: f64,
    /// How long a player may look at a sure move not taken before the coach
    /// says why it is sure, in seconds.
    pub stuck_after: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            talk: Talk::Normal,
            settle: 0.3,
            gap: 3.0,
            gone_after: 1.0,
            stuck_after: 8.0,
        }
    }
}

/// What a mark on a cell means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mark {
    /// Proven safe.
    Safe,
    /// The safe cell the coach is talking about.
    Next,
    /// Proven to be a mine, and not flagged yet.
    Mine,
    /// A flag on a cell that is proven safe.
    WrongFlag,
    /// Nothing is proven: the least risky cell, with its chance of a mine.
    Guess(f64),
    /// A number the coach's reason is about.
    Reason,
    /// Where to start a new game.
    Start,
}

/// Everything the overlay shows over the board.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Marks {
    pub cells: Vec<(Pos, Mark)>,
    /// What the coach has to say about this position (spoken or not).
    pub caption: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Show {
    /// Nothing new to show.
    Keep,
    /// No board in view: clear the overlay.
    Hide,
    /// New marks.
    Marks(Marks),
}

/// What to do after a frame: say something (or not), change the marks (or not).
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    pub say: Option<String>,
    pub show: Show,
}

impl Update {
    const NOTHING: Update = Update {
        say: None,
        show: Show::Keep,
    };
}

/// What a line is about: a line is not said twice running.
#[derive(Debug, Clone, PartialEq)]
enum Topic {
    /// These three carry the game's number: two games may end alike one after the other.
    NewGame(u32),
    Won(u32),
    Lost(u32),
    WrongFlag(Pos),
    Cut,
    Confused,
    Gamble,
    Safe(Reason),
    Mines(Reason),
    Guess(Pos),
}

#[derive(Debug, Clone)]
struct Line {
    topic: Topic,
    text: String,
    /// Said at once, over anything else.
    urgent: bool,
}

/// The settled position and what was made of it.
struct Position {
    board: Board,
    cut: Sides,
    analysis: Analysis,
    next: Option<Pos>,
    marks: Marks,
}

/// What the coach remembers about the game being played.
#[derive(Default)]
struct Notes {
    game: u32,
    wrong_flags_told: HashSet<Pos>,
    gamble_told: bool,
    cut_told: bool,
    confused_told: bool,
    over: bool,
    /// Moves the player had to guess.
    guesses: usize,
}

pub struct Coach {
    settings: Settings,
    /// The board the reader keeps seeing, with its cut sides, and since when.
    seen: Option<(Board, Sides, f64)>,
    last_seen_at: f64,
    current: Option<Position>,
    notes: Notes,
    said: Option<(Topic, f64)>,
    /// A line held back by the gap, said when the gap is over (if still current).
    pending: Option<Line>,
    /// An easy sure move not spoken while the player keeps moving: said if they stop.
    unspoken: Option<Line>,
    /// When the board last changed.
    settled_at: f64,
    showing: bool,
    lost_before: bool,
    /// Lines said so far: which way to put the next one, so that two alike in a row differ.
    variant: usize,
}

/// The first thing to say, before any board is seen.
pub const HELLO: &str = "Minesweeper coach is on. Open a game and I'll help.";

impl Coach {
    pub fn new(settings: Settings) -> Self {
        Coach {
            settings,
            seen: None,
            last_seen_at: f64::NEG_INFINITY,
            current: None,
            notes: Notes::default(),
            said: None,
            pending: None,
            unspoken: None,
            settled_at: f64::NEG_INFINITY,
            showing: false,
            lost_before: false,
            variant: 0,
        }
    }

    /// The marks for the position the coach has settled on, if any.
    pub fn marks(&self) -> Option<&Marks> {
        self.current.as_ref().map(|p| &p.marks)
    }

    /// One frame: what the reader saw in it (nothing, if no board), at time `now`.
    pub fn see(&mut self, reading: Option<&Reading>, now: f64) -> Update {
        let Some(r) = reading else {
            self.seen = None;
            if self.showing && now - self.last_seen_at >= self.settings.gone_after {
                self.showing = false;
                self.pending = None;
                return Update {
                    say: None,
                    show: Show::Hide,
                };
            }
            return Update::NOTHING;
        };
        self.last_seen_at = now;
        let mut board = unpressed(&r.board);
        for &(x, y) in &r.unsure {
            if x < board.width() && y < board.height() {
                board.set(x, y, Cell::Hidden);
            }
        }
        let still = match &self.seen {
            Some((b, c, since)) if *b == board && *c == r.cut => {
                now - since >= self.settings.settle
            }
            _ => {
                self.seen = Some((board.clone(), r.cut, now));
                self.settings.settle <= 0.0
            }
        };
        let same = self
            .current
            .as_ref()
            .is_some_and(|p| p.board == board && p.cut == r.cut);
        if !still || same {
            let mut update = self.flush(now);
            if !self.showing && same {
                self.showing = true;
                update.show = Show::Marks(self.current.as_ref().unwrap().marks.clone());
            }
            return update;
        }
        self.settle(board, r.cut, now)
    }

    /// A line held back until the gap was over, if it is due.
    fn flush(&mut self, now: f64) -> Update {
        if self.unspoken.is_some() && now - self.settled_at >= self.settings.stuck_after {
            let line = self.unspoken.take().unwrap();
            self.said = Some((line.topic, now));
            self.variant += 1;
            return Update {
                say: Some(line.text),
                show: Show::Keep,
            };
        }
        let due = self
            .said
            .as_ref()
            .is_none_or(|(_, at)| now - at >= self.settings.gap);
        match self.pending.take() {
            Some(line) if due => Update {
                say: self.consider(line, now),
                show: Show::Keep,
            },
            other => {
                self.pending = other;
                Update::NOTHING
            }
        }
    }

    /// Whether to say a line now; remembers it if so, holds it back if it must wait.
    fn consider(&mut self, line: Line, now: f64) -> Option<String> {
        let allowed = match line.topic {
            Topic::Safe(_) | Topic::Mines(_) => self.settings.talk != Talk::Quiet,
            _ => true,
        };
        if !allowed {
            return None;
        }
        // One easy sure move after another: the marks are enough, unless the player stops.
        let easy = matches!(
            line.topic,
            Topic::Safe(Reason::Satisfied { .. } | Reason::Filled { .. })
        );
        if self.settings.talk == Talk::Normal
            && easy
            && matches!(self.said, Some((Topic::Safe(_), _)))
        {
            self.unspoken = Some(line);
            return None;
        }
        if let Some((topic, at)) = &self.said {
            if *topic == line.topic {
                return None;
            }
            if !line.urgent && self.settings.talk != Talk::Chatty && now - at < self.settings.gap {
                self.pending = Some(line);
                return None;
            }
        }
        self.pending = None;
        self.said = Some((line.topic, now));
        self.variant += 1;
        Some(line.text)
    }

    /// The board held still and is new: work it out, choose a line, mark it.
    fn settle(&mut self, board: Board, cut: Sides, now: f64) -> Update {
        let total = if cut.any() {
            None
        } else {
            standard_mines(board.width(), board.height())
        };
        let analysis = analyze_seen(&board, cut, total);
        self.settled_at = now;
        self.unspoken = None;
        let prev = self.current.take();
        let new_game = match &prev {
            None => true,
            Some(p) => {
                p.board.width() != board.width()
                    || p.board.height() != board.height()
                    || (board.is_fresh() && !p.board.is_fresh())
            }
        };
        if new_game {
            self.notes = Notes {
                game: self.notes.game + 1,
                ..Notes::default()
            };
        }
        let prev = if new_game { None } else { prev };
        let (line, next) = self.choose(&board, cut, total, &analysis, prev.as_ref(), new_game);
        let marks = mark(&board, &analysis, next, line.as_ref(), self.notes.over);
        self.current = Some(Position {
            board,
            cut,
            analysis,
            next,
            marks: marks.clone(),
        });
        self.showing = true;
        self.pending = None;
        let say = line.and_then(|l| self.consider(l, now));
        Update {
            say,
            show: Show::Marks(marks),
        }
    }

    /// The one thing worth saying about a new position, and the sure cell to point at.
    fn choose(
        &mut self,
        board: &Board,
        cut: Sides,
        total: Option<usize>,
        a: &Analysis,
        prev: Option<&Position>,
        new_game: bool,
    ) -> (Option<Line>, Option<Pos>) {
        let line = |topic: Topic, text: String, urgent: bool| {
            Some(Line {
                topic,
                text,
                urgent,
            })
        };
        if board.is_lost() {
            if self.notes.over {
                return (None, None);
            }
            self.notes.over = true;
            let text = match prev {
                Some(p) => lost_line(board, p),
                None => "That game is over. Click the face for a new one.".to_string(),
            };
            let text = if self.lost_before {
                text
            } else {
                format!("{text} Click the face to play again.")
            };
            self.lost_before = true;
            return (line(Topic::Lost(self.notes.game), text, true), None);
        }
        if board.is_fresh() {
            if !new_game {
                return (None, None);
            }
            return (
                line(
                    Topic::NewGame(self.notes.game),
                    "New game. Start in a corner: the first click is always safe.".into(),
                    true,
                ),
                None,
            );
        }
        // Won: every cell not opened is a mine (flagged or not), so every safe cell is open.
        let unknown: Vec<Pos> = board
            .iter()
            .filter(|&(_, c)| c.is_unknown())
            .map(|(p, _)| p)
            .collect();
        let won = !cut.any()
            && !a.contradiction
            && unknown.iter().all(|&p| a.mine_at(p).is_some())
            && total.is_none_or(|t| unknown.len() == t);
        if won {
            if self.notes.over {
                return (None, None);
            }
            self.notes.over = true;
            let text = if self.notes.guesses == 0 {
                "Cleared, without a single guess. Well played!"
            } else {
                "Cleared! Well played."
            };
            return (line(Topic::Won(self.notes.game), text.into(), true), None);
        }
        self.notes.over = false;
        if a.contradiction {
            if self.notes.confused_told {
                return (None, None);
            }
            self.notes.confused_told = true;
            return (
                line(
                    Topic::Confused,
                    "The board doesn't add up. Is something covering it?".into(),
                    false,
                ),
                None,
            );
        }

        // What the player did since the last position.
        let opened: Vec<Pos> = match prev {
            Some(p) => board
                .iter()
                .filter(|&((x, y), c)| matches!(c, Cell::Open(_)) && p.board.get(x, y).is_unknown())
                .map(|(pos, _)| pos)
                .collect(),
            None => Vec::new(),
        };
        let gambled = prev.is_some_and(|p| {
            let sure: Vec<Pos> = p
                .analysis
                .safe
                .iter()
                .map(|d| d.cell)
                .filter(|&(x, y)| p.board.get(x, y) == Cell::Hidden)
                .collect();
            !opened.is_empty() && !sure.is_empty() && !opened.iter().any(|c| sure.contains(c))
        });
        if prev.is_some_and(|p| p.analysis.safe.is_empty() && p.analysis.guess.is_some())
            && !opened.is_empty()
        {
            self.notes.guesses += 1;
        }

        // The sure cell to point at: near what the player just opened, or near the last one pointed at.
        let sure: Vec<Pos> = a
            .safe
            .iter()
            .map(|d| d.cell)
            .filter(|&(x, y)| board.get(x, y) == Cell::Hidden)
            .collect();
        let focus = if opened.is_empty() {
            prev.and_then(|p| p.next)
        } else {
            let n = opened.len();
            Some((
                opened.iter().map(|p| p.0).sum::<usize>() / n,
                opened.iter().map(|p| p.1).sum::<usize>() / n,
            ))
        };
        let next = match focus {
            Some((fx, fy)) => sure
                .iter()
                .copied()
                .min_by_key(|&(x, y)| (x.abs_diff(fx).max(y.abs_diff(fy)), y, x)),
            None => sure.first().copied(),
        };

        let wrong: Vec<Pos> = a
            .safe
            .iter()
            .map(|d| d.cell)
            .filter(|&(x, y)| board.get(x, y) == Cell::Flag)
            .collect();
        if let Some(&w) = wrong
            .iter()
            .find(|w| !self.notes.wrong_flags_told.contains(*w))
        {
            self.notes.wrong_flags_told.extend(wrong.iter().copied());
            return (
                line(
                    Topic::WrongFlag(w),
                    "That flag is wrong: the cell under it is safe.".into(),
                    true,
                ),
                next,
            );
        }
        if cut.any() && !self.notes.cut_told {
            self.notes.cut_told = true;
            return (
                line(
                    Topic::Cut,
                    "I can only see part of the board. Scroll so all of it is on screen.".into(),
                    true,
                ),
                next,
            );
        }
        if gambled && !self.notes.gamble_told {
            self.notes.gamble_told = true;
            return (
                line(
                    Topic::Gamble,
                    "That was a gamble: there was a sure cell. Open the green ones first.".into(),
                    false,
                ),
                next,
            );
        }
        if let Some(n) = next {
            let d = a.safe_at(n).expect("a sure cell");
            let count = sure
                .iter()
                .filter(|&&c| a.safe_at(c).is_some_and(|e| e.reason == d.reason))
                .count();
            return (
                line(
                    Topic::Safe(d.reason),
                    safe_line(board, a, d.reason, count, self.variant),
                    false,
                ),
                next,
            );
        }
        if self.settings.talk == Talk::Chatty
            && let Some(m) = a
                .mines
                .iter()
                .find(|m| board.get(m.cell.0, m.cell.1) == Cell::Hidden)
        {
            let count = a
                .mines
                .iter()
                .filter(|e| e.reason == m.reason && board.get(e.cell.0, e.cell.1) == Cell::Hidden)
                .count();
            return (
                line(
                    Topic::Mines(m.reason),
                    mines_line(board, m.reason, count),
                    false,
                ),
                None,
            );
        }
        match a.guess {
            Some(g) => (
                line(
                    Topic::Guess(g.cell),
                    guess_line(board, g.cell, g.risk, g.ties, self.variant),
                    false,
                ),
                None,
            ),
            None => (None, None),
        }
    }
}

/// A cell held down under the mouse looks opened and empty. A real empty cell
/// never has a hidden neighbour — opening it opens them all — so such a cell
/// is still hidden.
pub fn unpressed(board: &Board) -> Board {
    let mut out = board.clone();
    for ((x, y), c) in board.iter() {
        if c == Cell::Open(0)
            && board
                .neighbors(x, y)
                .any(|(nx, ny)| board.get(nx, ny) == Cell::Hidden)
        {
            out.set(x, y, Cell::Hidden);
        }
    }
    out
}

fn number(board: &Board, (x, y): Pos) -> u8 {
    match board.get(x, y) {
        Cell::Open(n) => n,
        _ => 0,
    }
}

/// "about 12 percent"
fn percent(p: f64) -> String {
    let n = (p * 100.0).round() as u32;
    if p < 0.005 {
        "less than one percent".into()
    } else {
        format!("about {n} percent")
    }
}

/// Why the green cells are safe, put one of two ways (`variant`).
fn safe_line(board: &Board, a: &Analysis, reason: Reason, count: usize, variant: usize) -> String {
    let (cells, are) = if count == 1 {
        ("cell", "is")
    } else {
        ("cells", "are")
    };
    let second = variant % 2 == 1;
    match reason {
        Reason::Satisfied { by } => {
            let n = number(board, by);
            let place = if count == 1 {
                "next to it"
            } else {
                "around it"
            };
            match (second, n) {
                (false, 1) => {
                    format!("This 1 already has its mine, so the green {cells} {place} {are} safe.")
                }
                (false, _) => format!(
                    "This {n} already has its {n} mines, so the green {cells} {place} {are} safe."
                ),
                (true, 1) => {
                    format!("This 1's mine is found, so the green {cells} {place} {are} safe.")
                }
                (true, _) => format!(
                    "This {n}'s mines are all found, so the green {cells} {place} {are} safe."
                ),
            }
        }
        Reason::Pair { a: pa, b: pb } => {
            let (na, nb) = (number(board, pa), number(board, pb));
            // The mines b still needs, beyond those already known around it.
            let known = board
                .neighbors(pb.0, pb.1)
                .filter(|&c| a.mine_at(c).is_some() || board.get(c.0, c.1) == Cell::Flag)
                .count();
            let mines = if (nb as usize).saturating_sub(known) >= 2 {
                "mines"
            } else {
                "mine"
            };
            if second {
                format!(
                    "The {na} puts this {nb}'s {mines} in the cells they share, so its other {cells} {are} safe."
                )
            } else {
                format!(
                    "This {nb}'s {mines} must be in the cells it shares with the {na}, so its other {cells} {are} safe."
                )
            }
        }
        Reason::Filled { .. } | Reason::Counting => {
            if second {
                format!("The mine count says the green {cells} {are} safe.")
            } else {
                format!("Counting the mines left, the green {cells} {are} safe.")
            }
        }
    }
}

fn mines_line(board: &Board, reason: Reason, count: usize) -> String {
    let (cells, are, mines) = if count == 1 {
        ("cell", "is", "a mine")
    } else {
        ("cells", "are", "mines")
    };
    match reason {
        Reason::Filled { by } => {
            let n = number(board, by);
            format!(
                "This {n} has only {n} hidden {} left: {}.",
                if n == 1 { "cell" } else { "cells" },
                if n == 1 {
                    "it's a mine"
                } else {
                    "they're all mines"
                }
            )
        }
        Reason::Pair { a, b } => format!(
            "This {} needs more mines than it shares with the {}, so the red {cells} {are} {mines}.",
            number(board, b),
            number(board, a)
        ),
        Reason::Satisfied { .. } | Reason::Counting => {
            format!("Counting the mines left, the red {cells} {are} {mines}.")
        }
    }
}

/// The best odds when nothing is sure, put one of two ways (`variant`).
fn guess_line(board: &Board, cell: Pos, risk: f64, ties: usize, variant: usize) -> String {
    let second = variant % 2 == 1;
    if ties >= 1 && (risk - 0.5).abs() < 0.03 {
        return if second {
            "It's a coin toss now: fifty-fifty. Take the yellow one."
        } else {
            "No sure move here: it's fifty-fifty. Take the yellow one."
        }
        .into();
    }
    let alone = !board
        .neighbors(cell.0, cell.1)
        .any(|(x, y)| matches!(board.get(x, y), Cell::Open(n) if n > 0));
    let p = percent(risk);
    match (alone, second) {
        (true, false) => format!(
            "No sure move. Your best bet is away from the numbers: the yellow cell, {p} risk."
        ),
        (true, true) => {
            format!("Nothing is sure now. Try the yellow cell, away from the numbers: {p} risk.")
        }
        (false, false) => format!("No sure move. Your best bet is the yellow cell: {p} risk."),
        (false, true) => {
            format!("Nothing is sure now. The yellow cell is the safest bet: {p} risk.")
        }
    }
}

fn lost_line(board: &Board, prev: &Position) -> String {
    let Some((hit, _)) = board.iter().find(|&(_, c)| c == Cell::Exploded) else {
        return "Boom.".into();
    };
    let a = &prev.analysis;
    let had_sure = a
        .safe
        .iter()
        .any(|d| prev.board.get(d.cell.0, d.cell.1) == Cell::Hidden);
    let risk = if hit.0 < prev.board.width() && hit.1 < prev.board.height() {
        a.risk_at(hit)
    } else {
        None
    };
    if a.mine_at(hit).is_some() {
        "Boom. That cell was a sure mine.".into()
    } else if had_sure {
        "Boom. There was a sure cell to open instead.".into()
    } else if a.guess.is_some_and(|g| g.cell == hit) {
        format!(
            "Boom. Bad luck: that was the best bet, {} risk.",
            percent(risk.unwrap_or(0.0))
        )
    } else {
        match risk {
            Some(p) => format!("Boom. That cell had {} risk.", percent(p)),
            None => "Boom.".into(),
        }
    }
}

/// The marks for a position: every sure cell, the one pointed at, proven mines
/// not yet flagged, wrong flags, the numbers behind the reason, or the best guess.
fn mark(board: &Board, a: &Analysis, next: Option<Pos>, line: Option<&Line>, over: bool) -> Marks {
    let mut cells = Vec::new();
    let caption = line.map(|l| l.text.clone()).unwrap_or_default();
    if over {
        return Marks { cells, caption };
    }
    if board.is_fresh() {
        let (w, h) = (board.width() - 1, board.height() - 1);
        for c in [(0, 0), (w, 0), (0, h), (w, h)] {
            cells.push((c, Mark::Start));
        }
        return Marks { cells, caption };
    }
    if a.contradiction {
        return Marks { cells, caption };
    }
    for d in &a.safe {
        let (x, y) = d.cell;
        match board.get(x, y) {
            Cell::Hidden => cells.push((
                d.cell,
                if Some(d.cell) == next {
                    Mark::Next
                } else {
                    Mark::Safe
                },
            )),
            Cell::Flag => cells.push((d.cell, Mark::WrongFlag)),
            _ => {}
        }
    }
    for d in &a.mines {
        if board.get(d.cell.0, d.cell.1) == Cell::Hidden {
            cells.push((d.cell, Mark::Mine));
        }
    }
    if let Some(n) = next {
        match a.safe_at(n).map(|d| d.reason) {
            Some(Reason::Satisfied { by }) | Some(Reason::Filled { by }) => {
                cells.push((by, Mark::Reason))
            }
            Some(Reason::Pair { a: pa, b: pb }) => {
                cells.push((pa, Mark::Reason));
                cells.push((pb, Mark::Reason));
            }
            _ => {}
        }
    } else if let Some(g) = a.guess {
        cells.push((g.cell, Mark::Guess(g.risk)));
    }
    Marks { cells, caption }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, State};
    use crate::reader::Grid;

    fn reading(board: &Board) -> Reading {
        Reading {
            board: board.clone(),
            grid: Grid {
                x: 0.0,
                y: 0.0,
                pitch: 20.0,
                cols: board.width(),
                rows: board.height(),
            },
            unsure: Vec::new(),
            cut: Sides::NONE,
        }
    }

    fn instant() -> Settings {
        Settings {
            settle: 0.0,
            gap: 0.0,
            ..Settings::default()
        }
    }

    fn marked(u: &Update, m: Mark) -> Vec<Pos> {
        match &u.show {
            Show::Marks(ms) => ms
                .cells
                .iter()
                .filter(|(_, k)| *k == m)
                .map(|(p, _)| *p)
                .collect(),
            _ => Vec::new(),
        }
    }

    /// A player who does what the marks say, and what the coach says along the way.
    fn follow(seed: u64, settings: Settings) -> (Vec<String>, State) {
        let mut game = Game::beginner(seed);
        let mut coach = Coach::new(settings);
        let mut said = Vec::new();
        let mut t = 0.0;
        for _ in 0..200 {
            let u = coach.see(Some(&reading(&game.view())), t);
            t += 1.0;
            said.extend(u.say.clone());
            let Show::Marks(ms) = &u.show else {
                panic!("no marks for a new position")
            };
            for &(p, m) in &ms.cells {
                match m {
                    Mark::Safe | Mark::Next => assert!(
                        !game.is_mine(p.0, p.1),
                        "game {seed}: {p:?} marked safe is a mine"
                    ),
                    Mark::Mine => assert!(
                        game.is_mine(p.0, p.1),
                        "game {seed}: {p:?} marked a mine is not"
                    ),
                    _ => {}
                }
            }
            if game.state() != State::Playing {
                break;
            }
            let pick = |m: fn(&Mark) -> bool| ms.cells.iter().find(|(_, k)| m(k)).map(|(p, _)| *p);
            let target = pick(|k| *k == Mark::Next)
                .or(pick(|k| matches!(k, Mark::Guess(_))))
                .or(pick(|k| *k == Mark::Start));
            let (x, y) =
                target.unwrap_or_else(|| panic!("game {seed}: nothing to do on\n{}", game.view()));
            game.open(x, y);
        }
        (said, game.state())
    }

    #[test]
    fn a_game_played_as_the_marks_say() {
        let (mut won, mut lost, mut sure) = (0, 0, 0);
        for seed in 0..40 {
            let (said, state) = follow(seed, instant());
            assert_eq!(
                said.first().map(String::as_str),
                Some("New game. Start in a corner: the first click is always safe.")
            );
            for pair in said.windows(2) {
                assert_ne!(
                    pair[0], pair[1],
                    "game {seed} said the same twice running: {said:#?}"
                );
            }
            let last = said.last().unwrap();
            match state {
                State::Won => {
                    won += 1;
                    assert!(last.starts_with("Cleared"), "game {seed}: {said:#?}");
                }
                State::Lost => {
                    lost += 1;
                    assert!(
                        last.starts_with("Boom. Bad luck")
                            || last.starts_with("Boom. That cell had"),
                        "game {seed} lost following the marks: {said:#?}"
                    );
                }
                State::Playing => panic!("game {seed} never ended"),
            }
            sure += said.iter().any(|s| s.contains("green")) as usize;
        }
        assert!(won > 25 && lost > 0, "won {won}, lost {lost}");
        assert!(sure > 30, "only {sure} games of 40 had a sure move to tell");
    }

    #[test]
    fn easy_moves_in_a_row_go_quietly_until_the_player_stops() {
        let mut coach = Coach::new(instant());
        let a = Board::parse(
            "
            ##1.
            ##1.
            111.
            ....",
        )
        .unwrap();
        let first = coach.see(Some(&reading(&a)), 0.0);
        assert!(
            first
                .say
                .is_some_and(|s| s.contains("already has its mine"))
        );
        // The player opens the green cell: another easy sure move, not said...
        let b = Board::parse(
            "
            #11.
            ##1.
            111.
            ....",
        )
        .unwrap();
        let second = coach.see(Some(&reading(&b)), 1.0);
        assert_eq!(second.say, None);
        let Show::Marks(ms) = &second.show else {
            panic!("{second:?}")
        };
        assert!(!ms.caption.is_empty(), "though it is written");
        assert_eq!(coach.see(Some(&reading(&b)), 5.0).say, None);
        // ...until they stop and look at it for a while.
        let nudge = coach.see(Some(&reading(&b)), 9.5).say;
        assert!(nudge.is_some_and(|s| s.contains("green")));
        assert_eq!(coach.see(Some(&reading(&b)), 20.0).say, None, "once");
    }

    #[test]
    fn it_waits_for_the_board_to_hold_still() {
        let mut coach = Coach::new(Settings::default());
        let fresh = Board::new(9, 9);
        assert_eq!(coach.see(Some(&reading(&fresh)), 0.0), Update::NOTHING);
        assert_eq!(coach.see(Some(&reading(&fresh)), 0.2), Update::NOTHING);
        let u = coach.see(Some(&reading(&fresh)), 0.35);
        assert!(u.say.unwrap().starts_with("New game"));
        assert_eq!(
            marked(&coach.see(Some(&reading(&fresh)), 0.5), Mark::Start),
            Vec::<Pos>::new(),
            "no news, no new marks"
        );
        // Gone for a moment: the marks stay; gone for longer: they go.
        assert_eq!(coach.see(None, 1.0), Update::NOTHING);
        assert_eq!(coach.see(None, 1.6).show, Show::Hide);
        // Back: the same marks again, without a word.
        let back = coach.see(Some(&reading(&fresh)), 2.0);
        assert_eq!(
            (back.say.clone(), marked(&back, Mark::Start).len()),
            (None, 4)
        );
    }

    #[test]
    fn a_cell_held_under_the_mouse_is_not_opened() {
        let board = Board::parse(
            "
            #1.
            #1.
            #1.",
        )
        .unwrap();
        // The mouse held on the corner: it looks open and empty, beside hidden cells.
        let mut pressed = board.clone();
        pressed.set(0, 2, Cell::Open(0));
        assert_eq!(unpressed(&pressed), board);
        let mut coach = Coach::new(instant());
        let a = coach.see(Some(&reading(&board)), 0.0);
        let b = coach.see(Some(&reading(&pressed)), 1.0);
        assert!(a.show != Show::Keep && b == Update::NOTHING, "{b:?}");
    }

    #[test]
    fn a_wrong_flag_is_told_once() {
        let mut coach = Coach::new(instant());
        let board = Board::parse(
            "
            ##1.
            ##1.
            111.
            ....",
        )
        .unwrap();
        coach.see(Some(&reading(&board)), 0.0);
        let mut flagged = board.clone();
        flagged.set(1, 0, Cell::Flag);
        let u = coach.see(Some(&reading(&flagged)), 1.0);
        assert_eq!(
            u.say.as_deref(),
            Some("That flag is wrong: the cell under it is safe.")
        );
        assert_eq!(marked(&u, Mark::WrongFlag), vec![(1, 0)]);
        // Flag another cell: the first wrong flag is not told again.
        flagged.set(1, 1, Cell::Flag);
        let again = coach.see(Some(&reading(&flagged)), 2.0);
        assert!(
            again.say.as_deref() != Some("That flag is wrong: the cell under it is safe."),
            "{again:?}"
        );
    }

    #[test]
    fn sure_moves_come_with_their_reason() {
        let mut coach = Coach::new(instant());
        let board = Board::parse(
            "
            ##1.
            ##1.
            111.
            ....",
        )
        .unwrap();
        let u = coach.see(Some(&reading(&board)), 0.0);
        let said = u.say.clone().unwrap();
        assert!(
            said.starts_with("This 1 already has its mine, so the green cell"),
            "{said}"
        );
        let next = marked(&u, Mark::Next);
        assert_eq!(next.len(), 1);
        assert_eq!(marked(&u, Mark::Mine), vec![(1, 1)]);
        assert_eq!(marked(&u, Mark::Reason).len(), 1);
    }

    #[test]
    fn a_gamble_is_noticed_and_a_loss_explained() {
        let mut coach = Coach::new(instant());
        // (1,0) and (0,1) are sure; the player opens the corner, which nothing proves.
        let before = Board::parse(
            "
            ##1.
            ##1.
            111.
            ....",
        )
        .unwrap();
        assert!(coach.see(Some(&reading(&before)), 0.0).say.is_some());
        let mut after = before.clone();
        after.set(0, 0, Cell::Open(1));
        let g = coach.see(Some(&reading(&after)), 1.0);
        assert_eq!(
            g.say.as_deref(),
            Some("That was a gamble: there was a sure cell. Open the green ones first.")
        );
        // Then the proven mine.
        let mut lost = after.clone();
        lost.set(1, 1, Cell::Exploded);
        let l = coach.see(Some(&reading(&lost)), 2.0);
        assert_eq!(
            l.say.as_deref(),
            Some("Boom. That cell was a sure mine. Click the face to play again.")
        );
        assert_eq!(
            l.show,
            Show::Marks(Marks {
                cells: Vec::new(),
                caption: "Boom. That cell was a sure mine. Click the face to play again.".into()
            })
        );
    }

    #[test]
    fn lines_wait_their_turn_but_urgent_ones_do_not() {
        let settings = Settings {
            settle: 0.0,
            gap: 3.0,
            ..Settings::default()
        };
        let mut coach = Coach::new(settings);
        let fresh = Board::new(9, 9);
        assert!(coach.see(Some(&reading(&fresh)), 0.0).say.is_some());
        let mut game = Game::beginner(3);
        game.open(0, 0);
        let opened = game.view();
        // A sure move a second after "new game": held back, then said when the gap is over.
        let u = coach.see(Some(&reading(&opened)), 1.0);
        assert!(u.say.is_none() && matches!(u.show, Show::Marks(_)));
        assert_eq!(coach.see(Some(&reading(&opened)), 2.0).say, None);
        let later = coach.see(Some(&reading(&opened)), 3.1).say;
        assert!(later.is_some(), "the held line is said after the gap");
        // A new game is said at once.
        assert!(coach.see(Some(&reading(&fresh)), 3.5).say.is_some());
    }

    #[test]
    fn a_board_partly_in_view_is_told_and_kept_to() {
        // Whole, the bottom 1 proves the mine above it; seen down to here only, it might be below.
        let board = Board::parse(
            "
            ##1.
            ##1.
            111.",
        )
        .unwrap();
        let mut whole = Coach::new(instant());
        let u = whole.see(Some(&reading(&board)), 0.0);
        assert!(marked(&u, Mark::Mine).contains(&(1, 1)), "{u:?}");
        let mut part = Coach::new(instant());
        let mut r = reading(&board);
        r.cut = Sides {
            bottom: true,
            ..Sides::NONE
        };
        let u = part.see(Some(&r), 0.0);
        assert_eq!(
            u.say.as_deref(),
            Some("I can only see part of the board. Scroll so all of it is on screen.")
        );
        assert!(!marked(&u, Mark::Mine).contains(&(1, 1)), "{u:?}");
        // What holds whatever lies below is still marked: the 1s share their mine.
        assert!(
            marked(&u, Mark::Next).contains(&(0, 1)) || marked(&u, Mark::Safe).contains(&(0, 1)),
            "{u:?}"
        );
    }

    #[test]
    fn the_first_move_marks_the_corners() {
        let mut coach = Coach::new(instant());
        let u = coach.see(Some(&reading(&Board::new(16, 16))), 0.0);
        let mut corners = marked(&u, Mark::Start);
        corners.sort();
        assert_eq!(corners, vec![(0, 0), (0, 15), (15, 0), (15, 15)]);
    }
}
