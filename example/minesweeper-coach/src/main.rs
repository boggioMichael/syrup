//! mines-coach: watches a Minesweeper game on the screen and coaches it out
//! loud — the next sure move and why it is sure, or the best odds when nothing
//! is — with the cells marked on the board. It never clicks.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::RgbaImage;
use mines_coach::board::Board;
use mines_coach::coach::{Coach, HELLO, Mark, Marks, Settings, Show, Talk};
use mines_coach::game::{Game, State};
use mines_coach::overlay::{composite, layout, paint};
use mines_coach::play::{FirstClick, simulate};
use mines_coach::reader::{self, Grid};
use mines_coach::render::draw_page;

#[cfg(windows)]
mod win;

const USAGE: &str = "\
mines-coach: a Minesweeper coach. It watches the game on your screen and says
what to do next — the next sure move and why it is sure, or the best odds when
nothing is — and marks the cells on the board. It never clicks, and nothing
it sees leaves the computer.

  mines-coach [live] [--quiet | --chatty] [--mute] [--no-marks] [--rate N] [--fps N] [--dump DIR]
                    [--marks-in-screenshots]
      Coach live (Windows): open a game (minesweeper.online, or any classic-look
      Minesweeper) and play. --quiet speaks only for guesses, mistakes and the
      end of a game; --chatty for everything. --rate sets the voice's speed
      (-10 to 10). --dump saves what it saw around the board, and what it said.
      The marks keep out of screenshots unless --marks-in-screenshots.

  mines-coach replay [--quiet | --chatty] [--preview DIR] FRAMES...
      The coach over saved screenshots, in order: what it would have said, and
      (with --preview) each frame with its marks. A frame's time is the last
      number in its name, in milliseconds (frame-000012345.png); otherwise
      they are a second apart.

  mines-coach demo [SEED] [--level beginner|intermediate|expert] [--preview DIR]
      A game played by doing what the coach says, with what it says.

  mines-coach read IMAGES...
      Just the board read off each image (and checked against IMAGE.txt if there is one).

  mines-coach simulate [beginner|intermediate|expert|all] [GAMES]
      The coach's advice played out on many games: how often it wins.
";

struct Options {
    talk: Talk,
    voice: bool,
    marks: bool,
    marks_in_screenshots: bool,
    rate: i32,
    fps: f64,
    dump: Option<PathBuf>,
    preview: Option<PathBuf>,
    level: String,
    rest: Vec<String>,
}

fn parse(args: &[String]) -> Result<(String, Options), String> {
    let mut o = Options {
        talk: Talk::Normal,
        voice: true,
        marks: true,
        marks_in_screenshots: false,
        rate: 1,
        fps: 8.0,
        dump: None,
        preview: None,
        level: "beginner".into(),
        rest: Vec::new(),
    };
    let mut command = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--quiet" => o.talk = Talk::Quiet,
            "--chatty" => o.talk = Talk::Chatty,
            "--mute" => o.voice = false,
            "--no-marks" => o.marks = false,
            "--marks-in-screenshots" => o.marks_in_screenshots = true,
            "--rate" => {
                o.rate = value("--rate")?
                    .parse()
                    .map_err(|_| "--rate takes a number from -10 to 10")?
            }
            "--fps" => {
                o.fps = value("--fps")?
                    .parse()
                    .map_err(|_| "--fps takes a number")?
            }
            "--dump" => o.dump = Some(value("--dump")?.into()),
            "--preview" => o.preview = Some(value("--preview")?.into()),
            "--level" => o.level = value("--level")?,
            "-h" | "--help" | "help" => command = Some("help".to_string()),
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            word if command.is_none()
                && ["live", "replay", "demo", "read", "simulate"].contains(&word) =>
            {
                command = Some(word.to_string())
            }
            word => o.rest.push(word.to_string()),
        }
    }
    Ok((command.unwrap_or_else(|| "live".into()), o))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, o) = match parse(&args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let done = match command.as_str() {
        "live" => live(&o),
        "replay" => replay(&o),
        "demo" => demo(&o),
        "read" => read(&o),
        "simulate" => simulate_levels(&o),
        _ => {
            print!("{USAGE}");
            Ok(())
        }
    };
    if let Err(e) = done {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

/// "01:23.4"
fn clock(t: f64) -> String {
    format!("{:02}:{:04.1}", (t / 60.0) as u32, t % 60.0)
}

fn settings(o: &Options) -> Settings {
    Settings {
        talk: o.talk,
        ..Settings::default()
    }
}

/// What was seen around the board, and what was said, saved as it happens.
#[cfg_attr(not(windows), allow(dead_code))]
struct Dump {
    dir: PathBuf,
    log: std::fs::File,
}

#[cfg_attr(not(windows), allow(dead_code))]
impl Dump {
    fn new(dir: &Path) -> Result<Dump, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("said.txt"))
            .map_err(|e| e.to_string())?;
        Ok(Dump {
            dir: dir.to_path_buf(),
            log,
        })
    }

    fn line(&mut self, now: f64, text: &str) {
        let _ = writeln!(self.log, "{} {text}", clock(now));
    }

    /// The board and a margin round it (enough to see its frame), not the rest of the screen.
    fn frame(&mut self, now: f64, image: &RgbaImage, grid: &Grid) {
        let m = 1.5 * grid.pitch;
        let (l, t, r, b) = grid.bounds();
        let (x0, y0) = ((l - m).max(0.0) as u32, (t - m).max(0.0) as u32);
        let (x1, y1) = (
            ((r + m) as u32).min(image.width()),
            ((b + m) as u32).min(image.height()),
        );
        if x1 > x0 && y1 > y0 {
            let crop = image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image();
            let _ = crop.save(
                self.dir
                    .join(format!("frame-{:09}.png", (now * 1000.0) as u64)),
            );
        }
    }
}

#[cfg(not(windows))]
fn live(o: &Options) -> Result<(), String> {
    let _ = (
        o.dump.as_ref(),
        o.voice,
        o.marks,
        o.marks_in_screenshots,
        o.rate,
        o.fps,
    );
    Err("Coaching live needs Windows (the screen, the voice, the marks). Here, try `mines-coach demo`, or `mines-coach replay` on saved screenshots.".into())
}

#[cfg(windows)]
fn live(o: &Options) -> Result<(), String> {
    use std::time::Duration;
    win::init();
    println!(
        "Minesweeper coach: watching the screen. Open a game and play; close this window to stop."
    );
    println!(
        "Nothing it sees is saved{} or sent anywhere.\n",
        if o.dump.is_some() {
            " (except with --dump, in that folder)"
        } else {
            ""
        }
    );
    let voice = if o.voice {
        win::Voice::start(o.rate)
    } else {
        None
    };
    if o.voice && voice.is_none() {
        println!("(no voice found: the lines are only written here)");
    }
    let mut overlay = if o.marks {
        match win::Overlay::new(!o.marks_in_screenshots) {
            Ok(ov) => Some(ov),
            Err(e) => {
                println!("(no marks on the board: {e})");
                None
            }
        }
    } else {
        None
    };
    if !o.marks_in_screenshots && overlay.as_ref().is_some_and(|ov| !ov.hidden_from_capture) {
        println!(
            "(this Windows cannot keep the marks out of screen captures: they hide while the coach looks)"
        );
    }
    let mut dump = o.dump.as_deref().map(Dump::new).transpose()?;
    let mut coach = Coach::new(settings(o));
    let start = Instant::now();
    let say = |line: &str, now: f64, dump: &mut Option<Dump>| {
        println!("[{}] {line}", clock(now));
        if let Some(v) = &voice {
            v.say(line);
        }
        if let Some(d) = dump {
            d.line(now, line);
        }
    };
    say(HELLO, 0.0, &mut dump);
    let frame = Duration::from_secs_f64(1.0 / o.fps.clamp(1.0, 30.0));
    let mut last: Option<Grid> = None;
    let mut marks: Option<Marks> = None;
    loop {
        let began = Instant::now();
        if let Some(ov) = &overlay {
            ov.pump();
        }
        let now = start.elapsed().as_secs_f64();
        if let Some(ov) = &overlay {
            ov.before_capture();
        }
        let captured = syrup::capture::capture_screen();
        if let Some(ov) = &overlay {
            ov.after_capture();
        }
        if let Some(screen) = captured {
            let reading = match &last {
                Some(g) => reader::read_near(&screen.image, g),
                None => reader::read(&screen.image),
            };
            last = reading.as_ref().map(|r| r.grid);
            let update = coach.see(reading.as_ref(), now);
            if let Some(line) = &update.say {
                say(line, now, &mut dump);
            }
            match &update.show {
                Show::Hide => {
                    marks = None;
                    if let Some(ov) = overlay.as_mut() {
                        ov.hide();
                    }
                }
                Show::Marks(m) => {
                    marks = Some(m.clone());
                    if let (Some(d), Some(r)) = (dump.as_mut(), &reading) {
                        d.frame(now, &screen.image, &r.grid);
                    }
                }
                Show::Keep => {}
            }
            if let (Some(ov), Some(m), Some(r)) = (overlay.as_mut(), &marks, &reading) {
                let l = layout(&r.grid, screen.image.dimensions());
                if let Err(e) = ov.show(l, m, (screen.left, screen.top)) {
                    eprintln!("(could not draw the marks: {e})");
                }
            }
        }
        std::thread::sleep(frame.saturating_sub(began.elapsed()));
    }
}

/// The time a frame was taken, from the last number in its name (milliseconds).
fn frame_time(path: &Path) -> Option<f64> {
    let stem = path.file_stem()?.to_str()?;
    let tail: Vec<char> = stem
        .chars()
        .rev()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let digits: String = tail.into_iter().rev().collect();
    digits.parse::<u64>().ok().map(|ms| ms as f64 / 1000.0)
}

fn replay(o: &Options) -> Result<(), String> {
    let mut frames: Vec<PathBuf> = Vec::new();
    for arg in &o.rest {
        let p = PathBuf::from(arg);
        if p.is_dir() {
            let mut inside: Vec<PathBuf> = std::fs::read_dir(&p)
                .map_err(|e| e.to_string())?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| {
                    f.extension()
                        .is_some_and(|x| x == "png" || x == "jpg" || x == "jpeg")
                })
                .collect();
            inside.sort();
            frames.extend(inside);
        } else {
            frames.push(p);
        }
    }
    if frames.is_empty() {
        return Err("replay: which frames?".into());
    }
    if let Some(dir) = &o.preview {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let timed = frames.iter().all(|f| frame_time(f).is_some());
    let mut coach = Coach::new(Settings {
        settle: 0.0,
        ..settings(o)
    });
    println!("[{}] {HELLO}", clock(0.0));
    let mut last: Option<Grid> = None;
    for (i, path) in frames.iter().enumerate() {
        let now = if timed {
            frame_time(path).unwrap_or(0.0)
        } else {
            i as f64
        };
        let img = image::open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .to_rgba8();
        let reading = match &last {
            Some(g) => reader::read_near(&img, g),
            None => reader::read(&img),
        };
        last = reading.as_ref().map(|r| r.grid);
        let update = coach.see(reading.as_ref(), now);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        match &reading {
            Some(r) => println!(
                "  {name}: {}x{} board{}",
                r.board.width(),
                r.board.height(),
                if r.cut.any() {
                    ", partly out of view"
                } else {
                    ""
                }
            ),
            None => println!("  {name}: no board"),
        }
        if let Some(line) = &update.say {
            println!("[{}] {line}", clock(now));
        }
        if let (Some(dir), Some(r), Some(m)) = (&o.preview, &reading, coach.marks()) {
            let l = layout(&r.grid, img.dimensions());
            let marks = Marks {
                caption: String::new(),
                ..m.clone()
            };
            let out = composite(&img, &paint(&l, &marks), l.left, l.top);
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| format!("{i}"));
            out.save(dir.join(format!("{stem}.png")))
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// A game played by someone who does what the coach says, a move every second
/// and a half, the screen looked at eight times a second.
fn demo(o: &Options) -> Result<(), String> {
    let seed: u64 = o
        .rest
        .first()
        .map(|s| s.parse().map_err(|_| "the seed is a number"))
        .transpose()?
        .unwrap_or(1);
    let mut game = match o.level.as_str() {
        "beginner" => Game::beginner(seed),
        "intermediate" => Game::intermediate(seed),
        "expert" => Game::expert(seed),
        other => return Err(format!("unknown level {other}")),
    };
    if let Some(dir) = &o.preview {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let mut coach = Coach::new(settings(o));
    println!("[{}] {HELLO}", clock(0.0));
    let (tick, pace) = (0.125, 1.5);
    let mut t = 0.0;
    let mut next_move = 1.0;
    let mut shots = 0;
    while t < 1800.0 {
        let view = game.view();
        let (img, grid) = draw_page(&view, 32, 24, 24, (24, 24));
        let reading = reader::read_near(&img, &grid).ok_or("the drawn board was not found")?;
        let update = coach.see(Some(&reading), t);
        if let Some(line) = &update.say {
            println!("[{}] {line}", clock(t));
        }
        if let (Some(dir), Show::Marks(m)) = (&o.preview, &update.show) {
            let l = layout(&reading.grid, img.dimensions());
            let marks = Marks {
                caption: String::new(),
                ..m.clone()
            };
            composite(&img, &paint(&l, &marks), l.left, l.top)
                .save(dir.join(format!("demo-{shots:03}.png")))
                .map_err(|e| e.to_string())?;
            shots += 1;
        }
        if game.state() != State::Playing && update.say.is_some() {
            break;
        }
        if t >= next_move
            && game.state() == State::Playing
            && let Some(m) = coach.marks()
        {
            let pick = |f: fn(&Mark) -> bool| m.cells.iter().find(|(_, k)| f(k)).map(|(p, _)| *p);
            if let Some((x, y)) = pick(|k| *k == Mark::Next)
                .or(pick(|k| matches!(k, Mark::Guess(_))))
                .or(pick(|k| *k == Mark::Start))
            {
                game.open(x, y);
                next_move = t + pace;
            }
        }
        t += tick;
    }
    Ok(())
}

fn read(o: &Options) -> Result<(), String> {
    for path in &o.rest {
        let img = image::open(path)
            .map_err(|e| format!("{path}: {e}"))?
            .to_rgba8();
        let t = Instant::now();
        match reader::read(&img) {
            Some(r) => {
                println!(
                    "{path}: {}x{} cells of {:.2} px at ({:.1}, {:.1}), {:.1} ms, unsure {:?}, cut {:?}",
                    r.grid.cols,
                    r.grid.rows,
                    r.grid.pitch,
                    r.grid.x,
                    r.grid.y,
                    t.elapsed().as_secs_f64() * 1000.0,
                    r.unsure,
                    r.cut
                );
                print!("{}", r.board);
                let truth = Path::new(path).with_extension("txt");
                if let Ok(text) = std::fs::read_to_string(&truth) {
                    let want = Board::parse(&text)?;
                    let same_size =
                        want.width() == r.board.width() && want.height() == r.board.height();
                    let wrong: Vec<_> = want
                        .iter()
                        .filter(|&((x, y), c)| {
                            x >= r.board.width() || y >= r.board.height() || r.board.get(x, y) != c
                        })
                        .collect();
                    println!(
                        "  against {}: {}",
                        truth.display(),
                        if same_size && wrong.is_empty() {
                            "all right".to_string()
                        } else {
                            format!("{} wrong: {:?}", wrong.len(), wrong)
                        }
                    );
                }
            }
            None => println!("{path}: no board found"),
        }
    }
    Ok(())
}

fn simulate_levels(o: &Options) -> Result<(), String> {
    let which = o.rest.first().map(String::as_str).unwrap_or("all");
    let games: usize = o
        .rest
        .get(1)
        .map(|s| s.parse().map_err(|_| "GAMES is a number"))
        .transpose()?
        .unwrap_or(200);
    for (name, w, h, m) in [
        ("beginner", 9, 9, 10),
        ("intermediate", 16, 16, 40),
        ("expert", 30, 16, 99),
    ] {
        if which != "all" && which != name {
            continue;
        }
        let t = Instant::now();
        let s = simulate(w, h, m, games, 42, FirstClick::Corner);
        println!(
            "{name}: won {}/{} ({:.1}%), {} guesses (expected to hit {:.1} mines, hit {}), {} wrong proofs, {:.1} ms a game",
            s.won,
            s.games,
            100.0 * s.won as f64 / s.games as f64,
            s.guesses,
            s.expected_hits,
            s.hits,
            s.wrong_proofs,
            t.elapsed().as_secs_f64() * 1000.0 / s.games as f64
        );
    }
    Ok(())
}
