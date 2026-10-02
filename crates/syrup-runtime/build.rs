// Generated modules must be built for the target this crate runs on.
fn main() {
    let target = std::env::var("TARGET").expect("cargo sets TARGET");
    println!("cargo:rustc-env=SYRUP_HOST_TARGET={target}");
}
