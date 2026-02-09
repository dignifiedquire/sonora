fn main() {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = PathBuf::from(&crate_dir).join("include");
    let output_file = output_dir.join("wap_audio_processing.h");

    // Only regenerate if the source changed.
    println!("cargo::rerun-if-changed=src/lib.rs");
    println!("cargo::rerun-if-changed=src/types.rs");
    println!("cargo::rerun-if-changed=src/functions.rs");
    println!("cargo::rerun-if-changed=cbindgen.toml");

    // Create the output directory if it doesn't exist.
    fs::create_dir_all(&output_dir).expect("Failed to create include/ directory");

    let config = cbindgen::Config::from_file(PathBuf::from(&crate_dir).join("cbindgen.toml"))
        .expect("Failed to read cbindgen.toml");

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&output_file);
        }
        Err(e) => {
            // During `cargo publish --verify`, dependencies may not be
            // resolvable yet (crates not on crates.io). Skip header
            // generation — the checked-in header is still valid.
            println!("cargo::warning=cbindgen skipped: {e}");
        }
    }
}
