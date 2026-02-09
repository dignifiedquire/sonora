fn main() {
    // Find the C++ library via pkg-config
    let lib = pkg_config::Config::new()
        .atleast_version("3.0")
        .probe("webrtc-audio-processing-3")
        .expect(
            "webrtc-audio-processing-3 not found. Build the C++ library first:\n\
             meson setup builddir && ninja -C builddir install",
        );

    let mut build = cxx_build::bridge("src/bridge.rs");
    build
        .file("cpp/shim.cc")
        .std("c++20")
        .flag_if_supported("-DNDEBUG");

    // Add include paths from pkg-config
    for path in &lib.include_paths {
        build.include(path);
        // On Linux, abseil may be installed as a meson subproject alongside
        // the webrtc library. The pkg-config Cflags only list the
        // webrtc-audio-processing-3 subdirectory, but headers like
        // `absl/base/nullability.h` live one level up. Add the parent
        // include directory so they resolve.
        if let Some(parent) = path.parent() {
            build.include(parent);
        }
    }

    // Also include our own cpp/ directory (shim headers)
    build.include("cpp");

    // Include the C++ submodule root for webrtc headers.
    // Override with WEBRTC_CPP_ROOT env var if the submodule is elsewhere.
    let cpp_root = std::env::var("WEBRTC_CPP_ROOT")
        .unwrap_or_else(|_| format!("{}/../../cpp", env!("CARGO_MANIFEST_DIR")));
    build.include(&cpp_root);
    // Headers in the installed include dir use bare `api/...` paths,
    // which resolve under the `webrtc/` subdirectory of the source tree.
    build.include(format!("{cpp_root}/webrtc"));

    build.compile("sonora_shim");

    println!("cargo:rerun-if-changed=cpp/shim.h");
    println!("cargo:rerun-if-changed=cpp/shim.cc");
    println!("cargo:rerun-if-changed=src/bridge.rs");
}
