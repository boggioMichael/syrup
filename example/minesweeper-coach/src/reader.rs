//! The board read off a screenshot: where the cells are, and what each shows.
//!
//! Classic-look boards (Windows' Minesweeper, minesweeper.online's default
//! skins) are grey squares on a grid. Each cell has a flat grey inside —
//! the raised square of a hidden cell, or the open floor of an opened one —
//! so the flat-grey pixels, labelled into connected regions (syrup's
//! components), give one square region per cell. Regions of a common size
//! that sit on a common lattice are the board; the lattice's step is the cell
//! size, so the reading works at any zoom.
//!
//! Each cell is then read from a few bands of it: a hidden cell's top and
//! left edges are white (its bevel), an opened cell's are a thin grey line;
//! inside, the number's colour says which number (1 blue, 2 green, 3 red,
//! 4 navy, 5 maroon, 6 teal, 7 black, 8 grey), red on a hidden cell is a
//! flag, and black with a white glint on an opened one is a mine.
//!
//! A board may be only partly in view — scrolled off the screen, or covered
//! by something drawn over the page. Beyond each side of the cells the frame
//! should follow: a straight band of one grey (the bevel), then a different
//! grey. A side without it is reported open ([`Reading::cut`]), so the solver
//! does not take the last cells seen for the board's edge.

use image::{Rgba, RgbaImage};
use syrup::components::{Connectivity, label_pixels};
use syrup::geometry::Rect;

use crate::board::{Board, Cell, Sides};
use crate::render::DIGIT_COLORS;

/// Where the cells are in the picture: square cells of `pitch` pixels, the
/// first one's top left at `(x, y)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub x: f32,
    pub y: f32,
    pub pitch: f32,
    pub cols: usize,
    pub rows: usize,
}

impl Grid {
    pub fn cell_origin(&self, cx: usize, cy: usize) -> (f32, f32) {
        (
            self.x + cx as f32 * self.pitch,
            self.y + cy as f32 * self.pitch,
        )
    }

    pub fn cell_center(&self, cx: usize, cy: usize) -> (f32, f32) {
        let (x, y) = self.cell_origin(cx, cy);
        (x + self.pitch / 2.0, y + self.pitch / 2.0)
    }

    /// Left, top, right, bottom of the whole board.
    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        (
            self.x,
            self.y,
            self.x + self.cols as f32 * self.pitch,
            self.y + self.rows as f32 * self.pitch,
        )
    }
}

#[derive(Debug, Clone)]
pub struct Reading {
    pub board: Board,
    pub grid: Grid,
    /// Cells whose look fitted no kind clearly: the reading may be wrong there.
    pub unsure: Vec<(usize, usize)>,
    /// Sides where the board runs on out of view.
    pub cut: Sides,
}

fn is_fill(p: &Rgba<u8>) -> bool {
    let [r, g, b, _] = p.0;
    let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
    mx - mn <= 16 && mn >= 172 && mx <= 216
}

struct Blob {
    cx: f32,
    cy: f32,
    size: f32,
}

/// Find the board in a picture (anywhere in it, at any cell size) and read it.
pub fn read(image: &RgbaImage) -> Option<Reading> {
    let grid = find_grid(image)?;
    Some(read_grid(image, &grid))
}

/// Read the board near where it was last time — looking only there is quick —
/// or, when it is not found whole there (moved, grown, or gone), anywhere.
pub fn read_near(image: &RgbaImage, last: &Grid) -> Option<Reading> {
    let m = 2.0 * last.pitch;
    let (l, t, r, b) = last.bounds();
    let (x0, y0) = ((l - m).max(0.0) as u32, (t - m).max(0.0) as u32);
    let (x1, y1) = (
        ((r + m).max(0.0) as u32).min(image.width()),
        ((b + m).max(0.0) as u32).min(image.height()),
    );
    if x1 > x0 + 16 && y1 > y0 + 16 {
        let crop = image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image();
        if let Some(mut found) = read(&crop).filter(|f| !f.cut.any()) {
            found.grid.x += x0 as f32;
            found.grid.y += y0 as f32;
            return Some(found);
        }
    }
    read(image)
}

/// Where the cells are: the squares of flat grey that sit on one lattice.
pub fn find_grid(image: &RgbaImage) -> Option<Grid> {
    let (w, h) = image.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    let (_, comps) = label_pixels(
        image,
        Rect { x: 0, y: 0, w, h },
        Connectivity::Four,
        is_fill,
    );
    let blobs: Vec<Blob> = comps
        .iter()
        .filter_map(|c| {
            let (bw, bh) = (c.bounds.w as f32, c.bounds.h as f32);
            if !(5.0..=240.0).contains(&bw) || !(5.0..=240.0).contains(&bh) {
                return None;
            }
            if bw / bh < 0.78 || bw / bh > 1.28 || (c.area as f32) < 0.45 * bw * bh {
                return None;
            }
            Some(Blob {
                cx: c.bounds.x as f32 + bw / 2.0,
                cy: c.bounds.y as f32 + bh / 2.0,
                size: (bw + bh) / 2.0,
            })
        })
        .collect();
    if blobs.len() < 6 {
        return None;
    }
    // The step: to the nearest similar square on the right or below.
    let mut steps: Vec<f32> = Vec::new();
    for a in &blobs {
        let right = blobs
            .iter()
            .filter(|b| {
                b.cx > a.cx + 0.5 * a.size
                    && (b.cy - a.cy).abs() < 0.3 * a.size
                    && b.cx - a.cx < 1.8 * a.size
                    && (b.size / a.size - 1.0).abs() < 0.4
            })
            .map(|b| b.cx - a.cx)
            .fold(f32::INFINITY, f32::min);
        let below = blobs
            .iter()
            .filter(|b| {
                b.cy > a.cy + 0.5 * a.size
                    && (b.cx - a.cx).abs() < 0.3 * a.size
                    && b.cy - a.cy < 1.8 * a.size
                    && (b.size / a.size - 1.0).abs() < 0.4
            })
            .map(|b| b.cy - a.cy)
            .fold(f32::INFINITY, f32::min);
        for s in [right, below] {
            if s.is_finite() {
                steps.push(s);
            }
        }
    }
    if steps.len() < 4 {
        return None;
    }
    // Several boards-worth of steps may exist (other grids on the page): take the most common.
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pitch = mode_of(&steps)?;
    // The cells of that pitch: a closed cell's grey is its centre, an opened
    // cell's sits a little down and right of it (its grey starts after the line).
    let cells: Vec<(f32, f32)> = blobs
        .iter()
        .filter(|b| b.size > 0.6 * pitch && b.size < 1.05 * pitch)
        .map(|b| {
            if b.size < 0.86 * pitch {
                (b.cx, b.cy)
            } else {
                (b.cx - 0.03 * pitch, b.cy - 0.03 * pitch)
            }
        })
        .collect();
    if cells.len() < 6 {
        return None;
    }
    // The lattice through the cell with the most neighbours at one pitch.
    let anchor = cells.iter().max_by_key(|a| {
        cells
            .iter()
            .filter(|b| {
                ((b.0 - a.0).abs() - pitch).abs() < 0.2 * pitch && (b.1 - a.1).abs() < 0.2 * pitch
                    || ((b.1 - a.1).abs() - pitch).abs() < 0.2 * pitch
                        && (b.0 - a.0).abs() < 0.2 * pitch
            })
            .count()
    })?;
    let (ax, ay) = *anchor;
    let mut placed: Vec<(i64, i64, f32, f32)> = cells
        .iter()
        .filter_map(|&(x, y)| {
            let (i, j) = (((x - ax) / pitch).round(), ((y - ay) / pitch).round());
            let (rx, ry) = (x - (ax + i * pitch), y - (ay + j * pitch));
            (rx.abs() < 0.22 * pitch && ry.abs() < 0.22 * pitch)
                .then_some((i as i64, j as i64, x, y))
        })
        .collect();
    // The biggest connected patch of lattice places: the board (not the face button above it).
    placed.sort_by_key(|p| (p.1, p.0));
    placed.dedup_by_key(|p| (p.0, p.1));
    let set: std::collections::HashSet<(i64, i64)> = placed.iter().map(|p| (p.0, p.1)).collect();
    let mut seen: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
    let mut best: Vec<(i64, i64)> = Vec::new();
    for &(i, j, _, _) in &placed {
        if seen.contains(&(i, j)) {
            continue;
        }
        let mut patch = vec![(i, j)];
        let mut stack = vec![(i, j)];
        seen.insert((i, j));
        while let Some((a, b)) = stack.pop() {
            for (da, db) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
                (2, 0),
                (-2, 0),
                (0, 2),
                (0, -2),
            ] {
                let n = (a + da, b + db);
                if set.contains(&n) && seen.insert(n) {
                    patch.push(n);
                    stack.push(n);
                }
            }
        }
        if patch.len() > best.len() {
            best = patch;
        }
    }
    let (mut i0, mut i1) = (
        best.iter().map(|p| p.0).min()?,
        best.iter().map(|p| p.0).max()?,
    );
    let (mut j0, mut j1) = (
        best.iter().map(|p| p.1).min()?,
        best.iter().map(|p| p.1).max()?,
    );
    // Trim edge rows and columns that are mostly empty (a stray square beside the board).
    let occupied = |ia: i64, ib: i64, ja: i64, jb: i64| {
        best.iter()
            .filter(|p| p.0 >= ia && p.0 <= ib && p.1 >= ja && p.1 <= jb)
            .count()
    };
    loop {
        let (cols, rows) = (i1 - i0 + 1, j1 - j0 + 1);
        if occupied(i0, i0, j0, j1) * 3 < rows as usize && cols > 2 {
            i0 += 1;
        } else if occupied(i1, i1, j0, j1) * 3 < rows as usize && cols > 2 {
            i1 -= 1;
        } else if occupied(i0, i1, j0, j0) * 3 < cols as usize && rows > 2 {
            j0 += 1;
        } else if occupied(i0, i1, j1, j1) * 3 < cols as usize && rows > 2 {
            j1 -= 1;
        } else {
            break;
        }
    }
    let (cols, rows) = ((i1 - i0 + 1) as usize, (j1 - j0 + 1) as usize);
    if cols < 4 || rows < 4 || occupied(i0, i1, j0, j1) * 2 < cols * rows {
        return None;
    }
    // Least squares over the board's cells for the step and the origin.
    let inside: Vec<&(i64, i64, f32, f32)> = placed
        .iter()
        .filter(|p| p.0 >= i0 && p.0 <= i1 && p.1 >= j0 && p.1 <= j1)
        .collect();
    let n = inside.len() as f32;
    let (mi, mj) = (
        inside.iter().map(|p| p.0 as f32).sum::<f32>() / n,
        inside.iter().map(|p| p.1 as f32).sum::<f32>() / n,
    );
    let (mx, my) = (
        inside.iter().map(|p| p.2).sum::<f32>() / n,
        inside.iter().map(|p| p.3).sum::<f32>() / n,
    );
    let (mut sxy, mut sxx) = (0.0f32, 0.0f32);
    for p in &inside {
        let (di, dj) = (p.0 as f32 - mi, p.1 as f32 - mj);
        sxy += di * (p.2 - mx) + dj * (p.3 - my);
        sxx += di * di + dj * dj;
    }
    let step = if sxx > 0.0 { sxy / sxx } else { pitch };
    let step = if (step / pitch - 1.0).abs() < 0.1 {
        step
    } else {
        pitch
    };
    let x = mx - (mi - i0 as f32) * step - step / 2.0;
    let y = my - (mj - j0 as f32) * step - step / 2.0;
    Some(Grid {
        x,
        y,
        pitch: step,
        cols,
        rows,
    })
}

fn mode_of(sorted: &[f32]) -> Option<f32> {
    // the value with the most others within 6% of it
    let mut best = (0usize, 0.0f32);
    for &v in sorted {
        let n = sorted
            .iter()
            .filter(|&&u| (u - v).abs() <= 0.06 * v)
            .count();
        if n > best.0 {
            best = (n, v);
        }
    }
    let near: Vec<f32> = sorted
        .iter()
        .copied()
        .filter(|&u| (u - best.1).abs() <= 0.06 * best.1)
        .collect();
    (!near.is_empty()).then(|| near[near.len() / 2])
}

/// Every cell of a grid, read.
pub fn read_grid(image: &RgbaImage, grid: &Grid) -> Reading {
    let mut board = Board::new(grid.cols, grid.rows);
    let mut unsure = Vec::new();
    for cy in 0..grid.rows {
        for cx in 0..grid.cols {
            let (cell, sure) = read_cell(image, grid, cx, cy);
            board.set(cx, cy, cell);
            if !sure {
                unsure.push((cx, cy));
            }
        }
    }
    Reading {
        board,
        grid: *grid,
        unsure,
        cut: cut_sides(image, grid),
    }
}

/// Which sides of the cells are not followed by the board's frame: a band of
/// one grey right beyond the cells, then a band of another (the bevel, then
/// the frame). Every pixel along each band must be of it — cells running on
/// have their lines between cells; anything drawn over the page, its own
/// pattern — and the bands must be in the picture.
pub fn cut_sides(image: &RgbaImage, g: &Grid) -> Sides {
    let p = g.pitch;
    // A pixel or so into the bevel (the grid can be a pixel out), and twice into the frame.
    let inner = (0.08 * p).max(2.0);
    let outer = [(0.25 * p).max(inner + 4.0), (0.32 * p).max(inner + 5.0)];
    let (l, t, r, b) = g.bounds();
    // One line parallel to a side, `d` pixels beyond it: its usual luminance,
    // if (nearly) all of it is one grey.
    let line = |side: usize, d: f32| -> Option<f32> {
        let len = if matches!(side, 0 | 2) { b - t } else { r - l };
        let n = (0.92 * len).max(8.0) as usize;
        let mut lum = Vec::with_capacity(n);
        for k in 0..n {
            let f = 0.04 + 0.92 * k as f32 / (n - 1) as f32;
            let (x, y) = match side {
                0 => (l - d, t + f * (b - t)),
                1 => (l + f * (r - l), t - d),
                2 => (r - 1.0 + d, t + f * (b - t)),
                _ => (l + f * (r - l), b - 1.0 + d),
            };
            let [pr, pg, pb] = px(image, x, y)?;
            if pr.max(pg).max(pb) - pr.min(pg).min(pb) > 40 {
                return None;
            }
            lum.push((pr as f32 + pg as f32 + pb as f32) / 3.0);
        }
        let mut sorted = lum.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let off = lum.iter().filter(|&&v| (v - median).abs() > 12.0).count();
        (off as f32 <= 0.03 * n as f32).then_some(median)
    };
    let closed = |side: usize| -> bool {
        let Some(a) = line(side, inner).or_else(|| line(side, inner + 1.0)) else {
            return false;
        };
        match (line(side, outer[0]), line(side, outer[1])) {
            (Some(o0), Some(o1)) => (a - o0).abs() >= 30.0 && (o0 - o1).abs() <= 16.0,
            _ => false,
        }
    };
    Sides {
        left: !closed(0),
        top: !closed(1),
        right: !closed(2),
        bottom: !closed(3),
    }
}

fn px(image: &RgbaImage, x: f32, y: f32) -> Option<[u8; 3]> {
    let (x, y) = (x.floor(), y.floor());
    if x < 0.0 || y < 0.0 || x >= image.width() as f32 || y >= image.height() as f32 {
        return None;
    }
    let p = image.get_pixel(x as u32, y as u32).0;
    Some([p[0], p[1], p[2]])
}

/// What one cell shows, and whether it looked clearly like that.
fn read_cell(image: &RgbaImage, g: &Grid, cx: usize, cy: usize) -> (Cell, bool) {
    let (x0, y0) = g.cell_origin(cx, cy);
    let p = g.pitch;
    // The top and left bands, inside the bevel of a hidden cell.
    let band_a = (0.03 * p).max(0.0);
    let band_b = (0.10 * p).max(band_a + 1.0);
    let (mut white, mut band) = (0u32, 0u32);
    let mut t = band_a;
    while t < band_b {
        let mut s = 0.2 * p;
        while s < 0.8 * p {
            for q in [px(image, x0 + s, y0 + t), px(image, x0 + t, y0 + s)]
                .into_iter()
                .flatten()
            {
                band += 1;
                if q.iter().all(|&v| v >= 222) {
                    white += 1;
                }
            }
            s += 1.0;
        }
        t += 1.0;
    }
    let white = if band > 0 {
        white as f32 / band as f32
    } else {
        0.0
    };
    // The middle of the cell.
    let (mut total, mut red, mut pink, mut black, mut light) = (0u32, 0u32, 0u32, 0u32, 0u32);
    let mut votes = [0u32; 9];
    let (a, b) = (0.22 * p, 0.78 * p);
    let mut yy = a;
    while yy < b {
        let mut xx = a;
        while xx < b {
            if let Some([r, gg, bl]) = px(image, x0 + xx, y0 + yy) {
                total += 1;
                let (r, gg, bl) = (r as i32, gg as i32, bl as i32);
                if r >= 190 && gg <= 80 && bl <= 80 {
                    red += 1;
                }
                if r >= 225 && (120..=200).contains(&gg) && (120..=200).contains(&bl) {
                    pink += 1;
                }
                if r.max(gg).max(bl) <= 70 {
                    black += 1;
                }
                if r.min(gg).min(bl) >= 225 {
                    light += 1;
                }
                let off_fill = (r - 198).abs() + (gg - 198).abs() + (bl - 198).abs();
                if off_fill > 110 && r.min(gg).min(bl) < 225 {
                    let mut best = (i32::MAX, 0usize);
                    for (d, c) in DIGIT_COLORS.iter().enumerate().skip(1) {
                        let e = (r - c[0] as i32).pow(2)
                            + (gg - c[1] as i32).pow(2)
                            + (bl - c[2] as i32).pow(2);
                        if e < best.0 {
                            best = (e, d);
                        }
                    }
                    if best.0 < 95 * 95 {
                        votes[best.1] += 1;
                    }
                }
            }
            xx += 1.0;
        }
        yy += 1.0;
    }
    if total == 0 {
        return (Cell::Hidden, false);
    }
    let frac = |n: u32| n as f32 / total as f32;
    if white >= 0.5 {
        let sure = white >= 0.65;
        if frac(pink) > 0.25 {
            return (Cell::WrongFlag, sure);
        }
        if frac(red) >= 0.02 {
            return (Cell::Flag, sure);
        }
        return (Cell::Hidden, sure && frac(red) == 0.0);
    }
    let sure = white <= 0.25;
    // A mine: black with a white glint (a 7 is black without one); on red, the one that went off.
    if frac(black) > 0.08 && frac(red) > 0.08 {
        return (Cell::Exploded, sure);
    }
    if frac(black) > 0.12 && frac(light) > 0.002 {
        return (Cell::Mine, sure);
    }
    let (d, n) = votes
        .iter()
        .enumerate()
        .skip(1)
        .max_by_key(|&(_, &v)| v)
        .map(|(d, &v)| (d, v))
        .unwrap_or((0, 0));
    let enough = (0.012 * total as f32).max(2.0) as u32;
    if n < enough {
        return (Cell::Open(0), sure && n * 3 < enough);
    }
    let runner = votes
        .iter()
        .enumerate()
        .skip(1)
        .filter(|&(e, _)| e != d)
        .map(|(_, &v)| v)
        .max()
        .unwrap_or(0);
    (Cell::Open(d as u8), sure && runner * 3 < n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{draw_board, draw_page};

    const LOST: &str = "
        ..1F2#####
        ..12F#####
        ...1122###
        .....1X###
        11...1*!##
        #1...12345
        #2....1678
        ##1.......";

    #[test]
    fn a_drawn_board_reads_back_at_every_size() {
        let board = Board::parse(LOST).unwrap();
        for s in [14u32, 16, 20, 24, 30, 36, 48] {
            let img = draw_board(&board, s);
            let r = read(&img).unwrap_or_else(|| panic!("no board found at {s} px"));
            assert_eq!((r.grid.cols, r.grid.rows), (10, 8), "size at {s} px");
            assert_eq!(r.board.to_text(), board.to_text(), "cells at {s} px");
            assert!(
                (r.grid.pitch - s as f32).abs() < 0.3,
                "pitch {} at {s} px",
                r.grid.pitch
            );
        }
    }

    #[test]
    fn the_board_is_found_on_a_page_and_not_the_face() {
        let board = Board::parse(LOST).unwrap();
        for s in [16u32, 24, 33] {
            let (img, truth) = draw_page(&board, s, 57, 41, (120, 90));
            let r = read(&img).expect("board on the page");
            assert_eq!(r.board.to_text(), board.to_text(), "at {s} px");
            assert!(
                (r.grid.x - truth.x).abs() < 1.0 && (r.grid.y - truth.y).abs() < 1.0,
                "{:?} vs {:?}",
                r.grid,
                truth
            );
            assert_eq!(r.cut, Sides::NONE, "the frame is all round at {s} px");
        }
    }

    #[test]
    fn a_board_is_followed_when_the_page_moves() {
        let board = Board::parse(LOST).unwrap();
        let (img, truth) = draw_page(&board, 24, 40, 30, (300, 200));
        let first = read(&img).unwrap();
        // The page scrolled a few pixels: the board is read where it now is, not where it was.
        let mut moved =
            RgbaImage::from_pixel(img.width(), img.height(), Rgba([255, 255, 255, 255]));
        image::imageops::overlay(&mut moved, &img, 3, -5);
        let again = read_near(&moved, &first.grid).expect("found again");
        assert_eq!(again.board.to_text(), board.to_text());
        assert!(
            (again.grid.x - (truth.x + 3.0)).abs() < 0.6
                && (again.grid.y - (truth.y - 5.0)).abs() < 0.6,
            "{:?}",
            again.grid
        );
        // A bigger board in its place (a new game at another level) is read whole.
        let big = Board::new(16, 12);
        let (img2, _) = draw_page(&big, 24, 40, 30, (100, 60));
        let grown = read_near(&img2, &first.grid).expect("the bigger board");
        assert_eq!((grown.grid.cols, grown.grid.rows), (16, 12));
    }

    #[test]
    fn cells_alone_run_on_past_every_side() {
        let r = read(&draw_board(&Board::parse(LOST).unwrap(), 24)).unwrap();
        assert_eq!(r.cut, Sides::ALL);
    }

    #[test]
    fn a_banner_over_the_board_cuts_it() {
        let board = Board::parse(LOST).unwrap();
        for s in [20u32, 32, 41] {
            let (mut img, truth) = draw_page(&board, s, 30, 25, (60, 40));
            // A banner from the middle of the sixth row down: pale, with blue words on it.
            let top = (truth.y + 5.4 * s as f32) as i64;
            let (w, h) = (img.width() as i64, img.height() as i64);
            crate::render::fill_rect(&mut img, 0, top, w, h, [240, 244, 248]);
            for k in 0..8i64 {
                let x = 12 + k * (s as i64 * 2);
                crate::render::fill_rect(
                    &mut img,
                    x,
                    top + 6,
                    x + s as i64,
                    top + 6 + s as i64 / 2,
                    [20, 90, 220],
                );
            }
            let r = read(&img).expect("the rows above the banner");
            assert_eq!((r.grid.cols, r.grid.rows), (10, 5), "at {s} px");
            assert_eq!(
                r.cut,
                Sides {
                    bottom: true,
                    ..Sides::NONE
                },
                "at {s} px"
            );
        }
    }
}
