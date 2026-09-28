//! A Minesweeper coach: it watches a game on the screen, works out what the
//! numbers prove, and says what to do next — out loud, with the cell marked
//! on the board. It never clicks.
//!
//! - [`board`]: the board as the screen shows it.
//! - [`solver`]: what the numbers prove, and each hidden cell's chance of a mine.
//! - [`game`]: a game engine, for tests and for measuring the advice.
//! - [`reader`]: the board read off a screenshot.
//! - [`render`]: classic-look boards drawn for tests.
//! - [`coach`]: what to say and what to mark, as a game goes on.
//! - [`overlay`]: the marks, drawn as a picture to lay over the board.
//! - [`play`]: the advice played out on many games, to measure it.

pub mod board;
pub mod coach;
pub mod game;
pub mod overlay;
pub mod play;
pub mod reader;
pub mod render;
pub mod solver;
