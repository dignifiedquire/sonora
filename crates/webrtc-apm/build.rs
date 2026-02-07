use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = PathBuf::from(&crate_dir).join("include");
    let output_file = output_dir.join("wap_audio_processing.h");

    // Only regenerate if the source changed.
    println!("cargo::rerun-if-changed=src/ffi.rs");
    println!("cargo::rerun-if-changed=src/ffi/types.rs");
    println!("cargo::rerun-if-changed=src/ffi/functions.rs");
    println!("cargo::rerun-if-changed=cbindgen.toml");

    // Create the output directory if it doesn't exist.
    std::fs::create_dir_all(&output_dir).expect("Failed to create include/ directory");

    let config = cbindgen::Config::from_file(PathBuf::from(&crate_dir).join("cbindgen.toml"))
        .expect("Failed to read cbindgen.toml");

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("Failed to generate C header")
        .write_to_file(&output_file);
}
