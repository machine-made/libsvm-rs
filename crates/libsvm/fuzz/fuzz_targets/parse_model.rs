#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    let _ = libsvm_rs::io::load_model_from_reader(Cursor::new(data));
});
