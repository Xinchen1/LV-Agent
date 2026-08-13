use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use encoding_rs::UTF_8;
use rayon::prelude::*;
use regex::RegexBuilder;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

#[derive(Debug, Deserialize)]
struct Request {
    action: String,
    path: Option<String>,
    paths: Option<Vec<String>>,
    content: Option<String>,
    offset: Option<usize>,
    limit: Option<usize>,
    pattern: Option<String>,
    diff: Option<String>,
    encoding: Option<String>,
    line_numbers: Option<bool>,
    tag: Option<String>,
    checkpoint_dir: Option<String>,
}

#[derive(Debug, Serialize)]
struct Response {
    success: bool,
    output: String,
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let loop_mode = args.iter().any(|a| a == "--loop");

    if loop_mode {
        run_loop_mode();
    } else {
        run_one_shot();
    }
}

fn run_one_shot() {
    let mut input = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut input) {
        respond(Response {
            success: false,
            output: String::new(),
            error: Some(format!("Failed to read stdin: {}", e)),
            metadata: None,
        });
        return;
    }

    let req: Request = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            respond(Response {
                success: false,
                output: String::new(),
                error: Some(format!("Invalid JSON request: {}", e)),
                metadata: None,
            });
            return;
        }
    };

    respond(handle_request(req));
}

fn run_loop_mode() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout_lock = stdout.lock();
    let reader = stdin.lock();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let _ = writeln!(
                    stdout_lock,
                    "{}",
                    serde_json::to_string(&Response {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Invalid JSON request: {}", e)),
                        metadata: None,
                    })
                    .unwrap()
                );
                continue;
            }
        };
        let resp = handle_request(req);
        let _ = writeln!(stdout_lock, "{}", serde_json::to_string(&resp).unwrap());
        let _ = stdout_lock.flush();
    }
}

fn respond(resp: Response) {
    println!("{}", serde_json::to_string(&resp).unwrap());
}

fn handle_request(req: Request) -> Response {
    match req.action.as_str() {
        "read" => action_read(req),
        "multi_read" => action_multi_read(req),
        "write" => action_write(req),
        "list" => action_list(req),
        "exists" => action_exists(req),
        "analyze" => action_analyze(req),
        "grep" => action_grep(req),
        "find" => action_find(req),
        "apply_diff" => action_apply_diff(req),
        "verify" => action_verify(req),
        "diff" => action_diff(req),
        "backup" => action_backup(req),
        _ => Response {
            success: false,
            output: String::new(),
            error: Some(format!("Unknown action: {}", req.action)),
            metadata: None,
        },
    }
}

fn resolve_path(p: &str) -> PathBuf {
    let expanded = shellexpand::tilde(p);
    PathBuf::from(expanded.as_ref())
}

fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .to_string()
}

fn read_text_smart(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    // Try UTF-8 first.
    let (cow, _encoding_used, had_errors) = UTF_8.decode(&bytes);
    if !had_errors {
        return Ok(cow.into_owned());
    }

    // Fallback encodings.
    for enc in [encoding_rs::GBK, encoding_rs::WINDOWS_1252] {
        let (cow, _, had_errors) = enc.decode(&bytes);
        if !had_errors {
            return Ok(cow.into_owned());
        }
    }

    // Last resort: replace invalid UTF-8 sequences.
    Ok(cow.into_owned())
}

fn is_binary_ext(ext: &str) -> bool {
    let e = ext.to_lowercase();
    [
        "png", "jpg", "jpeg", "gif", "svg", "ico", "webp", "bmp",
        "zip", "gz", "tar", "rar", "7z",
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        "mp3", "mp4", "mov", "avi", "mkv", "wav", "flac",
        "exe", "dll", "so", "dylib", "bin", "dat",
    ]
    .contains(&e.as_str())
}

fn action_read(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for read"),
    };
    let path = resolve_path(&path_str);

    if !path.exists() {
        return error(&format!("File does not exist: {}", path.display()));
    }
    if path.is_dir() {
        return action_list(Request { path: Some(path_str), ..req });
    }

    let max_size = 10 * 1024 * 1024usize;
    match fs::metadata(&path) {
        Ok(m) if m.len() > max_size as u64 => return error("File too large"),
        _ => {}
    }

    let content = match read_text_smart(&path) {
        Ok(c) => c,
        Err(e) => return error(&e),
    };

    let mut lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let offset = req.offset.unwrap_or(0);
    let limit = req.limit.unwrap_or(total_lines);
    lines = lines.into_iter().skip(offset).take(limit).collect();

    let selected: Vec<String> = if req.line_numbers.unwrap_or(false) {
        lines
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{}: {}", offset + i + 1, line))
            .collect()
    } else {
        lines.iter().map(|l| l.to_string()).collect()
    };

    Response {
        success: true,
        output: selected.join("\n"),
        error: None,
        metadata: Some(serde_json::json!({
            "size": fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
            "lines": total_lines,
            "displayed_lines": selected.len(),
            "line_numbers": req.line_numbers.unwrap_or(false),
        })),
    }
}

fn action_multi_read(req: Request) -> Response {
    let paths = match req.paths {
        Some(p) => p,
        None => return error("paths is required for multi_read"),
    };

    let results: Vec<serde_json::Value> = paths
        .par_iter()
        .map(|p| {
            let sub = Request {
                action: "read".to_string(),
                path: Some(p.clone()),
                paths: None,
                content: None,
                offset: req.offset,
                limit: req.limit,
                pattern: None,
                diff: None,
                encoding: req.encoding.clone(),
                line_numbers: req.line_numbers,
                tag: None,
                checkpoint_dir: None,
            };
            let resp = action_read(sub);
            serde_json::json!({
                "path": p,
                "success": resp.success,
                "output": resp.output,
                "error": resp.error,
                "metadata": resp.metadata,
            })
        })
        .collect();

    Response {
        success: true,
        output: format!("Read {} files", results.len()),
        error: None,
        metadata: Some(serde_json::json!({ "results": results })),
    }
}

fn action_write(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for write"),
    };
    let content = match req.content {
        Some(c) => c,
        None => return error("content is required for write"),
    };
    let path = resolve_path(&path_str);

    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return error(&format!("Failed to create directory: {}", e));
        }
    }

    let mut backup_path: Option<PathBuf> = None;
    if path.exists() && path.is_file() {
        let backup = path.parent().unwrap_or(Path::new(".")).join(format!("{}.{}.bak", path.file_name().unwrap().to_string_lossy(), timestamp()));
        if fs::copy(&path, &backup).is_ok() {
            backup_path = Some(backup);
        }
    }

    if let Err(e) = fs::write(&path, &content) {
        if let Some(ref bp) = backup_path {
            let _ = fs::copy(bp, &path);
        }
        return error(&format!("Failed to write {}: {}", path.display(), e));
    }

    let mut output = format!("Written {} bytes to {}", content.len(), path.display());
    if let Some(ref bp) = backup_path {
        output.push_str(&format!("\nBackup saved to {}", bp.display()));
    }

    Response {
        success: true,
        output,
        error: None,
        metadata: None,
    }
}

fn action_list(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for list"),
    };
    let path = resolve_path(&path_str);

    if !path.exists() {
        return error(&format!("Directory does not exist: {}", path.display()));
    }
    if !path.is_dir() {
        return error(&format!("Path is not a directory: {}", path.display()));
    }

    let mut entries: Vec<String> = match fs::read_dir(&path) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
            .map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let meta = e.metadata().ok();
                let suffix = meta
                    .as_ref()
                    .map(|m| {
                        if m.is_dir() {
                            "".to_string()
                        } else {
                            format!(", {} bytes", m.len())
                        }
                    })
                    .unwrap_or_default();
                let prefix = meta.as_ref().map(|m| if m.is_dir() { "[DIR]  " } else { "[FILE] " }).unwrap_or("[?]    ");
                format!("{}{}{}", prefix, name, suffix)
            })
            .collect(),
        Err(e) => return error(&format!("Failed to list {}: {}", path.display(), e)),
    };

    entries.sort();
    Response {
        success: true,
        output: format!("Files in {} ({} entries):\n{}", path.display(), entries.len(), entries.join("\n")),
        error: None,
        metadata: Some(serde_json::json!({ "count": entries.len() })),
    }
}

fn action_exists(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for exists"),
    };
    let path = resolve_path(&path_str);
    Response {
        success: true,
        output: path.exists().to_string(),
        error: None,
        metadata: None,
    }
}

fn action_analyze(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for analyze"),
    };
    let path = resolve_path(&path_str);

    if !path.exists() {
        return error(&format!("Path does not exist: {}", path.display()));
    }

    let lang_map: HashMap<&str, &[&str]> = [
        ("python", &[".py"][..]),
        ("javascript", &[".js"][..]),
        ("typescript", &[".ts", ".tsx"][..]),
        ("java", &[".java"][..]),
        ("cpp", &[".cpp", ".hpp", ".cc", ".cxx"][..]),
        ("c", &[".c", ".h"][..]),
        ("go", &[".go"][..]),
        ("rust", &[".rs"][..]),
        ("markdown", &[".md"][..]),
        ("yaml", &[".yaml", ".yml"][..]),
        ("json", &[".json"][..]),
        ("html", &[".html", ".htm"][..]),
        ("css", &[".css"][..]),
    ]
    .into_iter()
    .collect();

    let entries: Vec<_> = WalkDir::new(&path)
        .into_iter()
        .filter_entry(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path().to_path_buf())
        .collect();

    let stats: Vec<_> = entries
        .par_iter()
        .filter_map(|fp| {
            let name = fp.file_name()?.to_string_lossy();
            if name.starts_with('.') {
                return None;
            }
            let ext = fp.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default();
            let ext_lower = ext.to_lowercase();
            let mut lines = 0usize;
            if !is_binary_ext(&ext_lower) {
                if let Ok(content) = read_text_smart(fp) {
                    lines = content.lines().count();
                }
            }
            let size = fs::metadata(fp).map(|m| m.len()).unwrap_or(0);
            Some((ext_lower, lines, size))
        })
        .collect();

    let total_files = stats.len();
    let mut total_lines = 0usize;
    let mut total_size = 0u64;
    let mut by_extension: HashMap<String, usize> = HashMap::new();
    let mut by_language: HashMap<String, usize> = HashMap::new();

    for (ext_lower, lines, size) in stats {
        total_lines += lines;
        total_size += size;
        *by_extension.entry(ext_lower.clone()).or_insert(0) += 1;
        for (lang, exts) in &lang_map {
            if exts.iter().any(|e| *e == format!(".{}", ext_lower)) {
                *by_language.entry(lang.to_string()).or_insert(0) += 1;
            }
        }
    }

    let mut output = format!(
        "Project Analysis for: {}\n{}\nTotal Files: {}\nTotal Lines: {}\nTotal Size: {:.2} MB\n\nBy Language:\n",
        path.display(),
        "=".repeat(50),
        total_files,
        total_lines,
        total_size as f64 / 1024.0 / 1024.0
    );

    let mut lang_vec: Vec<_> = by_language.iter().collect();
    lang_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (lang, count) in lang_vec.iter().take(12) {
        output.push_str(&format!("  {}: {} files\n", lang, count));
    }

    output.push_str("\nBy Extension:\n");
    let mut ext_vec: Vec<_> = by_extension.iter().collect();
    ext_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (ext, count) in ext_vec.iter().take(12) {
        let label = if ext.is_empty() { "(no ext)" } else { ext };
        output.push_str(&format!("  {}: {} files\n", label, count));
    }

    Response {
        success: true,
        output,
        error: None,
        metadata: Some(serde_json::json!({
            "total_files": total_files,
            "total_lines": total_lines,
            "total_size": total_size,
        })),
    }
}

fn action_grep(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for grep"),
    };
    let pattern = match req.pattern {
        Some(p) => p,
        None => return error("pattern is required for grep"),
    };
    let path = resolve_path(&path_str);

    // Prefer ripgrep when available for large directories.
    if path.is_dir() {
        if let Ok(output) = run_external_grep(&path, &pattern) {
            return Response {
                success: true,
                output: format!("Grep results for '{}' in {}:\n{}", pattern, path.display(), output),
                error: None,
                metadata: Some(serde_json::json!({ "matches": output.lines().count(), "pattern": pattern, "engine": "ripgrep" })),
            };
        }
    }

    let re = match RegexBuilder::new(&pattern).case_insensitive(true).build() {
        Ok(r) => r,
        Err(e) => return error(&format!("Invalid regex: {}", e)),
    };

    let entries: Vec<_> = if path.is_file() {
        vec![path.clone()]
    } else {
        WalkDir::new(&path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect()
    };

    let all_matches: Vec<(PathBuf, usize, String)> = entries
        .par_iter()
        .flat_map(|fp| {
            let mut local = Vec::new();
            let ext = fp.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default().to_lowercase();
            if is_binary_ext(&ext) {
                return local;
            }
            if let Ok(content) = read_text_smart(fp) {
                for (i, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        local.push((fp.clone(), i + 1, line[..line.len().min(200)].to_string()));
                        if local.len() >= 50 {
                            break;
                        }
                    }
                }
            }
            local
        })
        .collect();

    let matches: Vec<String> = all_matches
        .into_iter()
        .take(100)
        .map(|(fp, line_no, line)| {
            let rel = fp.strip_prefix(&path).unwrap_or(&fp).display().to_string();
            format!("{}:{}: {}", rel, line_no, line)
        })
        .collect();

    Response {
        success: true,
        output: format!(
            "Grep results for '{}' in {}:\n{}",
            pattern,
            path.display(),
            if matches.is_empty() { "No matches found".to_string() } else { matches.join("\n") }
        ),
        error: None,
        metadata: Some(serde_json::json!({ "matches": matches.len(), "pattern": pattern, "engine": "rust" })),
    }
}

fn run_external_grep(path: &Path, pattern: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut cmd = Command::new("rg");
    cmd.arg("--no-heading")
        .arg("--line-number")
        .arg("--max-columns")
        .arg("200")
        .arg("--max-count")
        .arg("50")
        .arg("--hidden")
        .arg("--glob")
        .arg("!.git")
        .arg("-i")
        .arg(pattern)
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = cmd.output()?;
    if !output.status.success() && output.stdout.is_empty() {
        return Err("ripgrep error".into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn glob_to_regex(pattern: &str) -> String {
    let mut re = String::from("^");
    for ch in pattern.chars() {
        match ch {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            '.' => re.push_str("\\."),
            '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
                re.push('\\');
                re.push(ch);
            }
            _ => re.push(ch),
        }
    }
    re.push('$');
    re
}

fn action_find(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for find"),
    };
    let path = resolve_path(&path_str);
    let pattern = req.pattern.as_deref().unwrap_or("*");

    // Prefer fd when available.
    if let Ok(output) = run_external_find(&path, pattern) {
        let lines: Vec<&str> = output.lines().collect();
        return Response {
            success: true,
            output: format!("Found {} files in {}:\n{}", lines.len(), path.display(), output),
            error: None,
            metadata: Some(serde_json::json!({ "count": lines.len(), "engine": "fd" })),
        };
    }

    let re = match RegexBuilder::new(&glob_to_regex(pattern)).case_insensitive(true).build() {
        Ok(r) => r,
        Err(e) => return error(&format!("Invalid pattern: {}", e)),
    };

    let mut results = Vec::new();
    for entry in WalkDir::new(&path)
        .into_iter()
        .filter_entry(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue;
            }
            if re.is_match(&name) {
                let rel = entry.path().strip_prefix(&path).unwrap_or(entry.path()).display().to_string();
                results.push(rel);
            }
        }
        if results.len() >= 200 {
            break;
        }
    }

    Response {
        success: true,
        output: format!("Found {} files in {}:\n{}", results.len(), path.display(), results.join("\n")),
        error: None,
        metadata: Some(serde_json::json!({ "count": results.len(), "engine": "rust" })),
    }
}

fn run_external_find(path: &Path, pattern: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut cmd = Command::new("fd");
    cmd.arg("--type")
        .arg("file")
        .arg("--hidden")
        .arg("--exclude")
        .arg(".git")
        .arg("--max-results")
        .arg("200")
        .arg(pattern)
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = cmd.output()?;
    if !output.status.success() && output.stdout.is_empty() {
        return Err("fd error".into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn parse_diff_blocks(diff: &str) -> Vec<(String, String)> {
    let mut blocks = Vec::new();
    let lines: Vec<&str> = diff.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() == "<<<<<<< SEARCH" {
            let mut search = Vec::new();
            i += 1;
            while i < lines.len() && lines[i].trim() != "=======" {
                search.push(lines[i]);
                i += 1;
            }
            i += 1; // skip =======
            let mut replace = Vec::new();
            while i < lines.len() && lines[i].trim() != ">>>>>>> REPLACE" {
                replace.push(lines[i]);
                i += 1;
            }
            i += 1; // skip >>>>>>> REPLACE
            blocks.push((search.join("\n"), replace.join("\n")));
        } else {
            i += 1;
        }
    }
    blocks
}

fn find_similar_line(text_lines: &[&str], needle: &str) -> isize {
    let needle = needle.trim();
    if needle.is_empty() {
        return -1;
    }
    let mut best_score = 0.0f64;
    let mut best_idx = -1isize;
    let needle_tokens: std::collections::HashSet<&str> = needle.split_whitespace().collect();
    for (i, line) in text_lines.iter().enumerate() {
        if line.contains(needle) {
            return (i + 1) as isize;
        }
        let line_tokens: std::collections::HashSet<&str> = line.trim().split_whitespace().collect();
        if line_tokens.is_empty() || needle_tokens.is_empty() {
            continue;
        }
        let common = line_tokens.intersection(&needle_tokens).count();
        let score = common as f64 / needle_tokens.len().max(1) as f64;
        if score > best_score {
            best_score = score;
            best_idx = (i + 1) as isize;
        }
    }
    if best_score > 0.3 {
        best_idx
    } else {
        -1
    }
}

fn apply_block(current: &str, search: &str, replace: &str) -> Option<String> {
    if current.contains(search) {
        return Some(current.replacen(search, replace, 1));
    }

    let current_lines: Vec<&str> = current.split('\n').collect();
    let search_lines: Vec<&str> = search.split('\n').collect();
    'outer: for i in 0..current_lines.len().saturating_sub(search_lines.len() - 1) {
        for j in 0..search_lines.len() {
            if current_lines[i + j].trim_end() != search_lines[j].trim_end() {
                continue 'outer;
            }
        }
        let mut new_lines: Vec<&str> = current_lines[..i].to_vec();
        new_lines.extend(replace.split('\n'));
        new_lines.extend(&current_lines[i + search_lines.len()..]);
        return Some(new_lines.join("\n"));
    }
    None
}

fn action_apply_diff(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for apply_diff"),
    };
    let diff = match req.diff {
        Some(d) => d,
        None => return error("diff is required for apply_diff"),
    };
    let path = resolve_path(&path_str);

    let content = match read_text_smart(&path) {
        Ok(c) => c,
        Err(e) => return error(&e),
    };

    let blocks = parse_diff_blocks(&diff);
    if blocks.is_empty() {
        return error("No valid diff blocks found");
    }

    let mut current = content.clone();
    let mut applied = 0usize;

    for (i, (search, replace)) in blocks.iter().enumerate() {
        match apply_block(&current, search, replace) {
            Some(next) => {
                current = next;
                applied += 1;
            }
            None => {
                let hint = {
                    let lines: Vec<&str> = content.split('\n').collect();
                    let mut hint_line = -1isize;
                    for line in search.lines() {
                        hint_line = find_similar_line(&lines, line);
                        if hint_line > 0 {
                            break;
                        }
                    }
                    if hint_line > 0 {
                        format!(" (closest match around line {})", hint_line)
                    } else {
                        String::new()
                    }
                };
                return Response {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Block {}/{} not found{}:\n{}", i + 1, blocks.len(), hint, &search[..search.len().min(500)])),
                    metadata: Some(serde_json::json!({ "applied": applied })),
                };
            }
        }
    }

    let backup = path.parent().unwrap_or(Path::new(".")).join(format!(
        "{}.{}.bak",
        path.file_name().unwrap().to_string_lossy(),
        timestamp()
    ));
    let backup_written = fs::copy(&path, &backup).is_ok();

    if let Err(e) = fs::write(&path, &current) {
        if backup_written {
            let _ = fs::copy(&backup, &path);
        }
        return error(&format!("Failed to write {}: {}. Original content restored.", path.display(), e));
    }

    let mut output = format!("Applied {} diff block(s) to {}", applied, path.display());
    if backup_written {
        output.push_str(&format!("\nBackup saved to {}", backup.display()));
    }

    Response {
        success: true,
        output,
        error: None,
        metadata: Some(serde_json::json!({ "applied": applied, "backup": backup_written.then(|| backup.display().to_string()) })),
    }
}

fn action_verify(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for verify"),
    };
    let path = resolve_path(&path_str);
    let ext = path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default().to_lowercase();

    let content = match read_text_smart(&path) {
        Ok(c) => c,
        Err(e) => return error(&e),
    };

    let result = match ext.as_str() {
        "py" => verify_python(&content),
        "json" => verify_json(&content),
        "yaml" | "yml" => verify_yaml(&content),
        _ => Ok(format!("No syntax verification available for file type: {}", ext)),
    };

    match result {
        Ok(msg) => Response {
            success: true,
            output: msg,
            error: None,
            metadata: None,
        },
        Err(e) => Response {
            success: false,
            output: String::new(),
            error: Some(e),
            metadata: None,
        },
    }
}

fn verify_python(content: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .args(["-c", "import ast, sys; ast.parse(sys.stdin.read())"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn python3: {}", e))?;

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(content.as_bytes());
    }

    let output = child.wait_with_output().map_err(|e| format!("Failed to read output: {}", e))?;
    if output.status.success() {
        Ok("Python syntax OK".to_string())
    } else {
        Err(format!("Python syntax error: {}", String::from_utf8_lossy(&output.stderr)))
    }
}

fn verify_json(content: &str) -> Result<String, String> {
    serde_json::from_str::<serde_json::Value>(content)
        .map(|_| "JSON syntax OK".to_string())
        .map_err(|e| format!("JSON syntax error: {}", e))
}

fn verify_yaml(content: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .args(["-c", "import yaml, sys; yaml.safe_load(sys.stdin.read())"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn python3: {}", e))?;

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(content.as_bytes());
    }

    let output = child.wait_with_output().map_err(|e| format!("Failed to read output: {}", e))?;
    if output.status.success() {
        Ok("YAML syntax OK".to_string())
    } else {
        Err(format!("YAML syntax error: {}", String::from_utf8_lossy(&output.stderr)))
    }
}

fn action_diff(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for diff"),
    };
    let path = resolve_path(&path_str);

    if !path.exists() {
        return error(&format!("Path does not exist: {}", path.display()));
    }

    let mut current: HashMap<String, serde_json::Value> = HashMap::new();
    let collect = |p: &Path, map: &mut HashMap<String, serde_json::Value>| {
        if p.is_file() {
            if let Ok(meta) = p.metadata() {
                map.insert(
                    p.to_string_lossy().to_string(),
                    serde_json::json!({
                        "mtime": meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()).unwrap_or(0.0),
                        "size": meta.len(),
                    }),
                );
            }
        } else if p.is_dir() {
            for entry in WalkDir::new(p)
                .into_iter()
                .filter_entry(|e| !e.file_name().to_string_lossy().starts_with('.'))
                .filter_map(|e| e.ok())
            {
                if entry.file_type().is_file() {
                    let name = entry.file_name().to_string_lossy();
                    if name.starts_with('.') {
                        continue;
                    }
                    if let Ok(meta) = entry.metadata() {
                        map.insert(
                            entry.path().to_string_lossy().to_string(),
                            serde_json::json!({
                                "mtime": meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()).unwrap_or(0.0),
                                "size": meta.len(),
                            }),
                        );
                    }
                }
            }
        }
    };

    collect(&path, &mut current);

    let cache_file = if path.is_file() {
        path.parent().unwrap_or(Path::new(".")).join(".file_cache.json")
    } else {
        path.join(".file_cache.json")
    };

    let mut changes: Vec<String> = Vec::new();
    if cache_file.exists() {
        if let Ok(text) = fs::read_to_string(&cache_file) {
            if let Ok(old) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&text) {
                for fp in old.keys() {
                    if !current.contains_key(fp) {
                        changes.push(format!("DELETED: {}", Path::new(fp).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| fp.clone())));
                    }
                }
                for (fp, info) in &current {
                    if let Some(old_info) = old.get(fp) {
                        let old_mtime = old_info.get("mtime").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let new_mtime = info.get("mtime").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        if (new_mtime - old_mtime).abs() > 1.0 {
                            changes.push(format!("MODIFIED: {}", Path::new(fp).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| fp.clone())));
                        }
                    } else {
                        changes.push(format!("NEW: {}", Path::new(fp).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| fp.clone())));
                    }
                }
            } else {
                changes = current.keys().map(|f| format!("NEW: {}", Path::new(f).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| f.clone()))).collect();
            }
        } else {
            changes = current.keys().map(|f| format!("NEW: {}", Path::new(f).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| f.clone()))).collect();
        }
    } else {
        changes = current.keys().map(|f| format!("NEW: {}", Path::new(f).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| f.clone()))).collect();
    }

    let _ = fs::write(&cache_file, serde_json::to_string_pretty(&current).unwrap_or_default());

    Response {
        success: true,
        output: format!(
            "File Changes in {}:\n{}\n{}\n\nTotal changes: {}",
            path.display(),
            "=".repeat(50),
            if changes.is_empty() { "No changes detected".to_string() } else { changes[..changes.len().min(50)].join("\n") },
            changes.len()
        ),
        error: None,
        metadata: Some(serde_json::json!({ "changes": changes.len() })),
    }
}

fn action_backup(req: Request) -> Response {
    let path_str = match req.path {
        Some(p) => p,
        None => return error("path is required for backup"),
    };
    let path = resolve_path(&path_str);

    if !path.exists() {
        return error(&format!("Path does not exist: {}", path.display()));
    }

    let backup_base = dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".file_backups");
    if let Err(e) = fs::create_dir_all(&backup_base) {
        return error(&format!("Failed to create backup directory: {}", e));
    }

    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "backup".to_string());
    let backup_path = backup_base.join(format!("{}_{}", name, timestamp()));

    let result = if path.is_dir() {
        fs::create_dir_all(&backup_path)
            .and_then(|_| copy_dir_recursively(&path, &backup_path))
            .map(|_| format!("Directory backed up to: {}", backup_path.display()))
    } else {
        fs::copy(&path, &backup_path).map(|_| format!("File backed up to: {}", backup_path.display()))
    };

    match result {
        Ok(msg) => Response {
            success: true,
            output: msg,
            error: None,
            metadata: Some(serde_json::json!({ "backup_path": backup_path.display().to_string() })),
        },
        Err(e) => error(&format!("Backup failed: {}", e)),
    }
}

fn copy_dir_recursively(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursively(&entry.path(), &dest)?;
        } else {
            fs::copy(&entry.path(), &dest)?;
        }
    }
    Ok(())
}

fn error(msg: &str) -> Response {
    Response {
        success: false,
        output: String::new(),
        error: Some(msg.to_string()),
        metadata: None,
    }
}
