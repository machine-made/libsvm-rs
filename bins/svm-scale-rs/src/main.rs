#![forbid(unsafe_code)]

use libsvm_rs::io::{format_17g, format_g};
use libsvm_rs::util::MAX_FEATURE_INDEX;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::process;

fn exit_with_help() -> ! {
    print!(
        "\
Usage: svm-scale [options] data_filename
options:
-l lower : x scaling lower limit (default -1)
-u upper : x scaling upper limit (default +1)
-y y_lower y_upper : y scaling limits (default: no y scaling)
-s save_filename : save scaling parameters to save_filename
-r restore_filename : restore scaling parameters from restore_filename
"
    );
    process::exit(1);
}

fn parse_flag_arg<'a>(args: &'a [String], i: &mut usize) -> &'a str {
    if *i >= args.len() {
        exit_with_help();
    }
    let value = &args[*i];
    *i += 1;
    value.as_str()
}

fn parse_feature_index_or_exit(idx_str: &str) -> i32 {
    let index = idx_str.parse::<i32>().unwrap_or_else(|_| {
        eprintln!("invalid feature index: {}", idx_str);
        process::exit(1);
    });
    if index <= 0 {
        eprintln!(
            "feature index {} out of valid range [1, {}]",
            index, MAX_FEATURE_INDEX
        );
        process::exit(1);
    }
    if index > MAX_FEATURE_INDEX {
        eprintln!(
            "feature index {} exceeds limit ({})",
            index, MAX_FEATURE_INDEX
        );
        process::exit(1);
    }
    index
}

fn open_file_or_exit(filename: &str) -> File {
    File::open(filename).unwrap_or_else(|e| {
        eprintln!("can't open file {}: {}", filename, e);
        process::exit(1);
    })
}

fn read_line_or_exit(line: io::Result<String>, filename: &str) -> String {
    line.unwrap_or_else(|e| {
        eprintln!("can't read file {}: {}", filename, e);
        process::exit(1);
    })
}

fn write_or_exit(result: io::Result<()>, filename: &str) {
    result.unwrap_or_else(|e| {
        eprintln!("can't write file {}: {}", filename, e);
        process::exit(1);
    });
}

fn warn_missing_scale_feature(index: i32, data_filename: &str, rfile: &str) {
    eprintln!(
        "WARNING: feature index {} appeared in file {} was not seen in the scaling factor file {}. The feature is scaled to 0.",
        index, data_filename, rfile
    );
}

fn warn_unseen_feature_range(
    first: i32,
    last: i32,
    data_filename: &str,
    rfile: &str,
    feature_min: &mut [f64],
    feature_max: &mut [f64],
) {
    for j in first..=last {
        let ju = j as usize;
        if ju < feature_min.len() && feature_min[ju] != feature_max[ju] {
            warn_missing_scale_feature(j, data_filename, rfile);
            feature_min[j as usize] = 0.0;
            feature_max[j as usize] = 0.0;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut lower = -1.0_f64;
    let mut upper = 1.0_f64;
    let mut y_lower = 0.0_f64;
    let mut y_upper = 0.0_f64;
    let mut y_scaling = false;
    let mut save_filename: Option<String> = None;
    let mut restore_filename: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        if !args[i].starts_with('-') {
            break;
        }
        let flag = &args[i];
        i += 1;
        let value = parse_flag_arg(&args, &mut i);

        if flag.len() < 2 {
            eprintln!("unknown option");
            exit_with_help();
        }
        match flag.as_bytes()[1] {
            b'l' => {
                lower = value.parse().unwrap_or_else(|_| exit_with_help());
            }
            b'u' => {
                upper = value.parse().unwrap_or_else(|_| exit_with_help());
            }
            b'y' => {
                y_lower = value.parse().unwrap_or_else(|_| exit_with_help());
                y_upper = parse_flag_arg(&args, &mut i)
                    .parse()
                    .unwrap_or_else(|_| exit_with_help());
                y_scaling = true;
            }
            b's' => {
                save_filename = Some(value.to_string());
            }
            b'r' => {
                restore_filename = Some(value.to_string());
            }
            _ => {
                eprintln!("unknown option");
                exit_with_help();
            }
        }
    }

    if upper <= lower || (y_scaling && y_upper <= y_lower) {
        eprintln!("inconsistent lower/upper specification");
        process::exit(1);
    }

    if restore_filename.is_some() && save_filename.is_some() {
        eprintln!("cannot use -r and -s simultaneously");
        process::exit(1);
    }

    if i + 1 != args.len() {
        exit_with_help();
    }
    let data_filename = &args[i];

    // ── Pass 1: find max_index ───────────────────────────────────────
    let mut max_index: i32 = 0;
    let mut num_nonzeros: usize = 0;

    // If restoring, scan restore file for max_index first
    if let Some(ref rfile) = restore_filename {
        let f = File::open(rfile).unwrap_or_else(|_| {
            eprintln!("can't open file {}", rfile);
            process::exit(1);
        });
        let reader = BufReader::new(f);
        let mut lines = reader.lines();
        // Check for 'y' header
        if let Some(Ok(first_line)) = lines.next() {
            if first_line.starts_with('y') {
                // Skip y_lower y_upper and y_min y_max lines
                lines.next();
                lines.next();
            }
            // If first char is 'x', skip that line too; otherwise it was 'x' line
            if !first_line.starts_with('y') {
                // First line was the 'x' line, next is lower upper
                let _ = lines.next();
            } else {
                // Already consumed 'y' + 2 lines, need 'x' line + lower/upper line
                lines.next(); // 'x'
                lines.next(); // lower upper
            }
        }
        for line in lines.map_while(Result::ok) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(idx_str) = parts.first() {
                let idx = parse_feature_index_or_exit(idx_str);
                max_index = max_index.max(idx);
            }
        }
    }

    // Scan data file for max_index and count nonzeros
    let f = open_file_or_exit(data_filename);
    let reader = BufReader::new(f);
    for line in reader.lines() {
        let line = read_line_or_exit(line, data_filename);
        let mut parts = line.split_whitespace();
        parts.next(); // skip label
        for token in parts {
            if let Some((idx_str, _)) = token.split_once(':') {
                let idx = parse_feature_index_or_exit(idx_str);
                max_index = max_index.max(idx);
                num_nonzeros += 1;
            }
        }
    }

    // ── Pass 2: find min/max per feature ─────────────────────────────
    let n = (max_index + 1) as usize;
    let mut feature_max = vec![f64::NEG_INFINITY; n];
    let mut feature_min = vec![f64::INFINITY; n];
    let mut y_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;

    let f = open_file_or_exit(data_filename);
    let reader = BufReader::new(f);
    for line in reader.lines() {
        let line = read_line_or_exit(line, data_filename);
        let mut parts = line.split_whitespace();

        // Parse label
        if let Some(label_str) = parts.next() {
            if let Ok(target) = label_str.parse::<f64>() {
                y_max = y_max.max(target);
                y_min = y_min.min(target);
            }
        }

        let mut next_index = 1i32;
        for token in parts {
            if let Some((idx_str, val_str)) = token.split_once(':') {
                let idx = parse_feature_index_or_exit(idx_str);
                let value: f64 = val_str.parse().unwrap_or(0.0);

                // Implicit zeros for skipped indices
                for j in next_index..idx {
                    let ju = j as usize;
                    feature_max[ju] = feature_max[ju].max(0.0);
                    feature_min[ju] = feature_min[ju].min(0.0);
                }

                let iu = idx as usize;
                feature_max[iu] = feature_max[iu].max(value);
                feature_min[iu] = feature_min[iu].min(value);
                next_index = idx + 1;
            }
        }
        // Trailing implicit zeros
        for j in next_index..=max_index {
            let ju = j as usize;
            feature_max[ju] = feature_max[ju].max(0.0);
            feature_min[ju] = feature_min[ju].min(0.0);
        }
    }

    // ── Pass 2.5: save/restore ───────────────────────────────────────
    if let Some(ref rfile) = restore_filename {
        let content = std::fs::read_to_string(rfile).unwrap_or_else(|_| {
            eprintln!("can't open file {}", rfile);
            process::exit(1);
        });
        let mut lines = content.lines();

        if let Some(first) = lines.next() {
            if first.starts_with('y') {
                // y scaling params
                if let Some(bounds) = lines.next() {
                    let parts: Vec<f64> = bounds
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if parts.len() == 2 {
                        y_lower = parts[0];
                        y_upper = parts[1];
                    }
                }
                if let Some(minmax) = lines.next() {
                    let parts: Vec<f64> = minmax
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if parts.len() == 2 {
                        y_min = parts[0];
                        y_max = parts[1];
                    }
                }
                y_scaling = true;
                // Next should be 'x' line
                lines.next();
            }
            // If first line was 'x' or we just consumed 'y' section + 'x'
            // Read lower upper
            if let Some(bounds) = lines.next() {
                let parts: Vec<f64> = bounds
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if parts.len() == 2 {
                    lower = parts[0];
                    upper = parts[1];
                }
            }
            // Read per-feature min/max
            let mut next_restore_index = 1i32;
            for line in lines {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() != 3 {
                    continue;
                }
                let idx = parse_feature_index_or_exit(parts[0]);
                let fmin: f64 = parts[1].parse().unwrap_or(0.0);
                let fmax: f64 = parts[2].parse().unwrap_or(0.0);

                // Warn about features not seen in restore file
                warn_unseen_feature_range(
                    next_restore_index,
                    idx - 1,
                    data_filename,
                    rfile,
                    &mut feature_min,
                    &mut feature_max,
                );

                let iu = idx as usize;
                if iu < n {
                    feature_min[iu] = fmin;
                    feature_max[iu] = fmax;
                }
                next_restore_index = idx + 1;
            }
            // Remaining features
            warn_unseen_feature_range(
                next_restore_index,
                max_index,
                data_filename,
                rfile,
                &mut feature_min,
                &mut feature_max,
            );
        }
    }

    if let Some(ref sfile) = save_filename {
        let mut out = File::create(sfile).unwrap_or_else(|_| {
            eprintln!("can't open file {}", sfile);
            process::exit(1);
        });
        if y_scaling {
            write_or_exit(writeln!(out, "y"), sfile);
            write_or_exit(
                writeln!(out, "{} {}", format_17g(y_lower), format_17g(y_upper)),
                sfile,
            );
            write_or_exit(
                writeln!(out, "{} {}", format_17g(y_min), format_17g(y_max)),
                sfile,
            );
        }
        write_or_exit(writeln!(out, "x"), sfile);
        write_or_exit(
            writeln!(out, "{} {}", format_17g(lower), format_17g(upper)),
            sfile,
        );
        for j in 1..=max_index {
            let ju = j as usize;
            if feature_min[ju] != feature_max[ju] {
                write_or_exit(
                    writeln!(
                        out,
                        "{} {} {}",
                        j,
                        format_17g(feature_min[ju]),
                        format_17g(feature_max[ju])
                    ),
                    sfile,
                );
            }
        }
    }

    // ── Pass 3: scale and output ─────────────────────────────────────
    let mut new_num_nonzeros: usize = 0;

    let f = open_file_or_exit(data_filename);
    let reader = BufReader::new(f);
    for line in reader.lines() {
        let line = read_line_or_exit(line, data_filename);
        let mut parts = line.split_whitespace();

        // Scale target
        if let Some(label_str) = parts.next() {
            if let Ok(mut target) = label_str.parse::<f64>() {
                if y_scaling {
                    if target == y_min {
                        target = y_lower;
                    } else if target == y_max {
                        target = y_upper;
                    } else {
                        target = y_lower + (y_upper - y_lower) * (target - y_min) / (y_max - y_min);
                    }
                }
                print!("{} ", format_17g(target));
            }
        }

        let mut next_index = 1i32;
        for token in parts {
            if let Some((idx_str, val_str)) = token.split_once(':') {
                let idx = parse_feature_index_or_exit(idx_str);
                let value: f64 = val_str.parse().unwrap_or(0.0);

                // Output implicit zeros for skipped features
                for j in next_index..idx {
                    output_feature(
                        j,
                        0.0,
                        &feature_min,
                        &feature_max,
                        lower,
                        upper,
                        &mut new_num_nonzeros,
                    );
                }

                output_feature(
                    idx,
                    value,
                    &feature_min,
                    &feature_max,
                    lower,
                    upper,
                    &mut new_num_nonzeros,
                );
                next_index = idx + 1;
            }
        }
        // Trailing features
        for j in next_index..=max_index {
            output_feature(
                j,
                0.0,
                &feature_min,
                &feature_max,
                lower,
                upper,
                &mut new_num_nonzeros,
            );
        }

        println!();
    }

    if new_num_nonzeros > num_nonzeros {
        eprintln!(
            "WARNING: original #nonzeros {}\n\
             \t > new      #nonzeros {}\n\
             If feature values are non-negative and sparse, use -l 0 rather than the default -l -1",
            num_nonzeros, new_num_nonzeros
        );
    }
}

fn output_feature(
    index: i32,
    value: f64,
    feature_min: &[f64],
    feature_max: &[f64],
    lower: f64,
    upper: f64,
    new_num_nonzeros: &mut usize,
) {
    let iu = index as usize;
    if iu >= feature_min.len() {
        return;
    }

    // Skip single-valued features
    if feature_max[iu] == feature_min[iu] {
        return;
    }

    let scaled = if value == feature_min[iu] {
        lower
    } else if value == feature_max[iu] {
        upper
    } else {
        lower + (upper - lower) * (value - feature_min[iu]) / (feature_max[iu] - feature_min[iu])
    };

    if scaled != 0.0 {
        print!("{}:{} ", index, format_g(scaled));
        *new_num_nonzeros += 1;
    }
}
