// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Shared canonical primitives for build-time provenance producers.
//!
//! This file is included by build scripts and by the qualification build
//! identity tool. It deliberately depends only on `std` and `sha2` so the
//! producers use one path, mode, tree, and digest encoding without creating a
//! runtime dependency edge.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

pub const MAX_CANONICAL_ITEMS: usize = 200_000;
pub const MAX_CANONICAL_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_CANONICAL_TREE_BYTES: usize = 256 * 1024 * 1024;
const JOINED_PATH_OPTION_PREFIXES: [&str; 14] = [
    "-I",
    "-L",
    "-F",
    "-isystem",
    "--sysroot=",
    "@",
    "/I",
    "/Fo",
    "/Fd",
    "/Fe",
    "/LIBPATH:",
    "/OUT:",
    "/PDB:",
    "/IMPLIB:",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeEntry {
    pub path: String,
    pub mode: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathToken {
    pub token: String,
    pub root: PathBuf,
}

impl PathToken {
    pub fn new(token: &str, root: &Path) -> Result<Self, String> {
        if token.len() < 3
            || !token.starts_with('<')
            || !token.ends_with('>')
            || token.bytes().any(|byte| byte.is_ascii_whitespace())
        {
            return Err(format!("invalid canonical path token {token:?}"));
        }
        let root = fs::canonicalize(root)
            .map_err(|err| format!("cannot canonicalize {}: {err}", root.display()))?;
        Ok(Self {
            token: token.to_owned(),
            root,
        })
    }
}

#[derive(Default)]
pub struct CanonicalEncoder {
    bytes: Vec<u8>,
}

impl CanonicalEncoder {
    pub fn domain(&mut self, value: &str) {
        self.text(value);
    }

    #[allow(dead_code)] // build-script consumers do not all encode booleans
    pub fn bool(&mut self, value: bool) {
        self.bytes.push(u8::from(value));
    }

    pub fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn bytes(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.bytes.extend_from_slice(value);
    }

    pub fn text(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    pub fn finish(self) -> [u8; 32] {
        Sha256::digest(self.bytes).into()
    }
}

pub fn canonical_tree_digest(entries: &[TreeEntry]) -> Result<[u8; 32], String> {
    if entries.len() > MAX_CANONICAL_ITEMS {
        return Err(format!(
            "canonical tree exceeds {MAX_CANONICAL_ITEMS} entries"
        ));
    }
    let mut budget = TreeByteBudget::default();
    let mut prior = None;
    let mut encoder = CanonicalEncoder::default();
    encoder.domain("mlxcel.canonical-tree.v1");
    encoder.u64(entries.len() as u64);
    for entry in entries {
        budget.add(entry.bytes.len())?;
        validate_relative_label(&entry.path)?;
        if prior.is_some_and(|value: &str| value >= entry.path.as_str()) {
            return Err("canonical tree entries are not strictly path-sorted".to_owned());
        }
        if !matches!(entry.mode, 0o100644 | 0o100755 | 0o120000) {
            return Err(format!("unsupported tree mode {:o}", entry.mode));
        }
        encoder.text(&entry.path);
        encoder.u64(u64::from(entry.mode));
        encoder.bytes(&entry.bytes);
        prior = Some(&entry.path);
    }
    Ok(encoder.finish())
}

pub fn directory_tree_entries(root: &Path) -> Result<Vec<TreeEntry>, String> {
    let root = fs::canonicalize(root)
        .map_err(|err| format!("cannot canonicalize {}: {err}", root.display()))?;
    let mut pending = vec![root.clone()];
    let mut entries = Vec::new();
    let mut budget = TreeByteBudget::default();
    while let Some(directory) = pending.pop() {
        let children = fs::read_dir(&directory)
            .map_err(|err| format!("cannot enumerate {}: {err}", directory.display()))?;
        for child in children {
            let child =
                child.map_err(|err| format!("cannot enumerate {}: {err}", directory.display()))?;
            if child.file_name() == ".git" {
                continue;
            }
            let path = child.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|err| format!("cannot inspect {}: {err}", path.display()))?;
            if metadata.is_dir() {
                pending.push(path);
                continue;
            }
            if !metadata.is_file() && !metadata.file_type().is_symlink() {
                return Err(format!("unsupported tree entry {}", path.display()));
            }
            let relative = path
                .strip_prefix(&root)
                .map_err(|_| format!("tree entry escaped {}", root.display()))?;
            let label = strict_relative_path(relative)?;
            let (mode, bytes) = if metadata.file_type().is_symlink() {
                let target = fs::read_link(&path)
                    .map_err(|err| format!("cannot read symlink {}: {err}", path.display()))?;
                let bytes = strict_path_text(&target)?.into_bytes();
                budget.add(bytes.len())?;
                (0o120000, bytes)
            } else {
                (
                    filesystem_mode(&metadata),
                    bounded_file(&path, &mut budget)?,
                )
            };
            entries.push(TreeEntry {
                path: label,
                mode,
                bytes,
            });
            if entries.len() > MAX_CANONICAL_ITEMS {
                return Err(format!(
                    "canonical tree exceeds {MAX_CANONICAL_ITEMS} entries"
                ));
            }
        }
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(entries)
}

pub fn directory_tree_digest(root: &Path) -> Result<[u8; 32], String> {
    canonical_tree_digest(&directory_tree_entries(root)?)
}

pub fn git_candidate_tree_digest(root: &Path) -> Result<[u8; 32], String> {
    let mut budget = TreeByteBudget::default();
    let mut entries = git_head_entries(root, &mut budget)?;
    let skip = git_skip_worktree_paths(root)?;
    for entry in entries.values_mut() {
        let absolute = root.join(&entry.path);
        match fs::symlink_metadata(&absolute) {
            Ok(metadata) => {
                budget.remove(entry.bytes.len())?;
                let (mode, bytes) = candidate_file(&absolute, &metadata, entry.mode, &mut budget)?;
                entry.mode = mode;
                entry.bytes = bytes;
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                if !skip.contains(&entry.path) {
                    budget.remove(entry.bytes.len())?;
                    entry.bytes.clear();
                    entry.mode = 0;
                }
            }
            Err(err) => return Err(format!("cannot inspect {}: {err}", absolute.display())),
        }
    }
    entries.retain(|_, entry| entry.mode != 0);
    for relative in git_untracked_paths(root)? {
        let absolute = root.join(&relative);
        let metadata = fs::symlink_metadata(&absolute)
            .map_err(|err| format!("cannot inspect {}: {err}", absolute.display()))?;
        let (mode, bytes) = candidate_file(&absolute, &metadata, 0o100644, &mut budget)?;
        if entries
            .insert(
                relative.clone(),
                TreeEntry {
                    path: relative,
                    mode,
                    bytes,
                },
            )
            .is_some()
        {
            return Err("untracked path aliases a tracked path".to_owned());
        }
    }
    canonical_tree_digest(&entries.into_values().collect::<Vec<_>>())
}

pub fn git_head_tree_digest(root: &Path) -> Result<[u8; 32], String> {
    let mut budget = TreeByteBudget::default();
    canonical_tree_digest(
        &git_head_entries(root, &mut budget)?
            .into_values()
            .collect::<Vec<_>>(),
    )
}

pub fn canonicalize_argument(value: &str, roots: &[PathToken]) -> Result<String, String> {
    if value.len() > MAX_CANONICAL_TEXT_BYTES {
        return Err("build argument exceeds 64 KiB".to_owned());
    }
    let mut roots = roots.to_vec();
    roots.sort_by(|left, right| {
        path_sort_key(&right.root)
            .len()
            .cmp(&path_sort_key(&left.root).len())
    });
    let role_tokens = roots
        .iter()
        .map(|mapping| mapping.token.clone())
        .collect::<Vec<_>>();
    let mut canonical = value.replace('\\', "/");
    for mapping in roots {
        let physical = path_sort_key(&mapping.root);
        canonical = replace_case_aware(&canonical, &physical, &mapping.token);
    }
    if contains_unmapped_absolute_path(&canonical, &role_tokens) {
        return Err(format!(
            "build argument retains an unmapped absolute path: {canonical}"
        ));
    }
    Ok(canonical)
}

pub fn validate_relative_label(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.starts_with('/')
        || value.starts_with('\\')
        || value.contains('\\')
        || value.bytes().any(|byte| byte == 0 || byte < 0x20)
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || has_windows_drive_prefix(value)
    {
        return Err(format!(
            "path label is not canonical and relative: {value:?}"
        ));
    }
    Ok(())
}

pub fn strict_path_text(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("path is not strict UTF-8: {}", path.display()))
}

pub fn strict_relative_path(path: &Path) -> Result<String, String> {
    if path.components().any(|component| {
        matches!(
            component,
            Component::Prefix(_) | Component::RootDir | Component::ParentDir
        )
    }) {
        return Err(format!("path is not relative: {}", path.display()));
    }
    let value = strict_path_text(path)?.replace('\\', "/");
    validate_relative_label(&value)?;
    Ok(value)
}

pub fn file_sha256(path: &Path) -> Result<[u8; 32], String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|err| format!("cannot inspect {}: {err}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "build input is not a regular file: {}",
            path.display()
        ));
    }
    let bytes = fs::read(path).map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    Ok(Sha256::digest(bytes).into())
}

pub fn canonical_tool_output(bytes: Vec<u8>, label: &str) -> Result<String, String> {
    let text = String::from_utf8(bytes).map_err(|_| format!("{label} is not strict UTF-8"))?;
    if text.bytes().any(|byte| byte == 0) {
        return Err(format!("{label} contains NUL"));
    }
    let canonical = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    if canonical.is_empty() || canonical.len() > MAX_CANONICAL_TEXT_BYTES {
        return Err(format!("{label} is empty or exceeds 64 KiB"));
    }
    Ok(canonical)
}

pub fn hex_digest(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn parse_sha256(value: &str, label: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} is not a SHA-256 digest"));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let pair = std::str::from_utf8(pair).map_err(|_| format!("{label} is not ASCII"))?;
        digest[index] = u8::from_str_radix(pair, 16)
            .map_err(|_| format!("{label} contains invalid hexadecimal"))?;
    }
    Ok(digest)
}

pub fn is_full_lower_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn git_head_entries(
    root: &Path,
    budget: &mut TreeByteBudget,
) -> Result<BTreeMap<String, TreeEntry>, String> {
    let output = git_output(root, &["ls-tree", "-r", "-z", "HEAD"])?;
    let mut entries = BTreeMap::new();
    for record in output
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
    {
        let record = std::str::from_utf8(record)
            .map_err(|_| "tracked Git path is not strict UTF-8".to_owned())?;
        let (metadata, path) = record
            .split_once('\t')
            .ok_or_else(|| "malformed git ls-tree record".to_owned())?;
        let mut metadata = metadata.split_whitespace();
        let mode = u32::from_str_radix(
            metadata
                .next()
                .ok_or_else(|| "tracked Git mode is missing".to_owned())?,
            8,
        )
        .map_err(|_| "tracked Git mode is invalid".to_owned())?;
        if metadata.next() != Some("blob") {
            return Err("non-blob tracked Git entry is unsupported".to_owned());
        }
        let object = metadata
            .next()
            .ok_or_else(|| "tracked Git object is missing".to_owned())?;
        let path = path.replace('\\', "/");
        validate_relative_label(&path)?;
        let bytes = git_blob(root, object, budget)?;
        entries.insert(path.clone(), TreeEntry { path, mode, bytes });
    }
    Ok(entries)
}

fn git_skip_worktree_paths(root: &Path) -> Result<BTreeSet<String>, String> {
    let output = git_output(root, &["ls-files", "-v", "-z"])?;
    output
        .split(|byte| *byte == 0)
        .filter(|record| record.starts_with(b"S "))
        .map(|record| {
            let value = std::str::from_utf8(&record[2..])
                .map_err(|_| "skip-worktree path is not strict UTF-8".to_owned())?
                .replace('\\', "/");
            validate_relative_label(&value)?;
            Ok(value)
        })
        .collect()
}

fn git_untracked_paths(root: &Path) -> Result<Vec<String>, String> {
    let output = git_output(root, &["ls-files", "--others", "--exclude-standard", "-z"])?;
    let mut paths = output
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
        .map(|record| {
            let value = std::str::from_utf8(record)
                .map_err(|_| "untracked path is not strict UTF-8".to_owned())?
                .replace('\\', "/");
            validate_relative_label(&value)?;
            Ok(value)
        })
        .collect::<Result<Vec<_>, String>>()?;
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn git_output(root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = git_command(root)
        .args(args)
        .output()
        .map_err(|err| format!("cannot execute Git: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "Git command failed: git -C {} {}",
            root.display(),
            args.join(" ")
        ));
    }
    Ok(output.stdout)
}

fn git_command(root: &Path) -> Command {
    let mut command = Command::new("git");
    command.arg("-C").arg(root);
    for name in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    ] {
        command.env_remove(name);
    }
    command
}

fn candidate_file(
    path: &Path,
    metadata: &fs::Metadata,
    base_mode: u32,
    budget: &mut TreeByteBudget,
) -> Result<(u32, Vec<u8>), String> {
    if metadata.file_type().is_symlink() {
        let target = fs::read_link(path)
            .map_err(|err| format!("cannot read symlink {}: {err}", path.display()))?;
        let bytes = strict_path_text(&target)?.into_bytes();
        budget.add(bytes.len())?;
        return Ok((0o120000, bytes));
    }
    if !metadata.is_file() {
        return Err(format!("candidate entry is not a file: {}", path.display()));
    }
    #[cfg(unix)]
    let mode = {
        let _ = base_mode;
        filesystem_mode(metadata)
    };
    #[cfg(not(unix))]
    let mode = if base_mode == 0o100755 {
        0o100755
    } else {
        0o100644
    };
    let bytes = bounded_file(path, budget)?;
    Ok((mode, bytes))
}

#[derive(Default)]
struct TreeByteBudget {
    retained: usize,
}

impl TreeByteBudget {
    fn add(&mut self, bytes: usize) -> Result<(), String> {
        self.retained = self
            .retained
            .checked_add(bytes)
            .ok_or_else(|| "canonical tree byte total overflowed".to_owned())?;
        if self.retained > MAX_CANONICAL_TREE_BYTES {
            return Err(format!(
                "canonical tree exceeds {MAX_CANONICAL_TREE_BYTES} content bytes"
            ));
        }
        Ok(())
    }

    fn remove(&mut self, bytes: usize) -> Result<(), String> {
        self.retained = self
            .retained
            .checked_sub(bytes)
            .ok_or_else(|| "canonical tree byte accounting underflowed".to_owned())?;
        Ok(())
    }

    fn remaining(&self) -> usize {
        MAX_CANONICAL_TREE_BYTES - self.retained
    }
}

fn bounded_file(path: &Path, budget: &mut TreeByteBudget) -> Result<Vec<u8>, String> {
    let limit = budget
        .remaining()
        .checked_add(1)
        .ok_or_else(|| "canonical tree read bound overflowed".to_owned())?;
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(|err| format!("cannot open {}: {err}", path.display()))?
        .take(limit as u64)
        .read_to_end(&mut bytes)
        .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    budget.add(bytes.len())?;
    Ok(bytes)
}

fn git_blob(root: &Path, object: &str, budget: &mut TreeByteBudget) -> Result<Vec<u8>, String> {
    let size = git_output(root, &["cat-file", "-s", object])?;
    let size = std::str::from_utf8(&size)
        .map_err(|_| "Git blob size is not strict UTF-8".to_owned())?
        .trim()
        .parse::<usize>()
        .map_err(|_| "Git blob size is not canonical".to_owned())?;
    budget.add(size)?;
    let bytes = git_output(root, &["cat-file", "blob", object])?;
    if bytes.len() != size {
        return Err("Git blob size changed while reading canonical tree".to_owned());
    }
    Ok(bytes)
}

fn filesystem_mode(metadata: &fs::Metadata) -> u32 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            0o100644
        } else {
            0o100755
        }
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        0o100644
    }
}

fn path_sort_key(path: &Path) -> String {
    let value = strict_path_text(path)
        .unwrap_or_else(|err| panic!("{err}"))
        .replace('\\', "/");
    if cfg!(windows) {
        value.strip_prefix("//?/").unwrap_or(&value).to_owned()
    } else {
        value
    }
}

fn replace_case_aware(value: &str, needle: &str, replacement: &str) -> String {
    let search_value = if cfg!(windows) {
        value.to_ascii_lowercase()
    } else {
        value.to_owned()
    };
    let search_needle = if cfg!(windows) {
        needle.to_ascii_lowercase()
    } else {
        needle.to_owned()
    };
    let mut result = String::new();
    let mut offset = 0;
    while let Some(relative) = search_value[offset..].find(&search_needle) {
        let found = offset + relative;
        let end = found + needle.len();
        if path_root_boundary(value, found, end) {
            result.push_str(&value[offset..found]);
            result.push_str(replacement);
            offset = end;
        } else {
            let width = value[found..]
                .chars()
                .next()
                .map(char::len_utf8)
                .unwrap_or(1);
            result.push_str(&value[offset..found + width]);
            offset = found + width;
        }
    }
    result.push_str(&value[offset..]);
    result
}

fn path_root_boundary(value: &str, start: usize, end: usize) -> bool {
    let after = value[end..].chars().next();
    let after_is_boundary = after.is_none_or(|character| {
        character.is_ascii_whitespace()
            || matches!(character, '/' | '"' | '\'' | '=' | ',' | ';' | '>' | ':')
    });
    path_prefix_boundary(value, start) && after_is_boundary
}

fn path_prefix_boundary(value: &str, start: usize) -> bool {
    value[..start].chars().next_back().is_none_or(|character| {
        character.is_ascii_whitespace()
            || matches!(character, '"' | '\'' | '=' | ',' | ';' | '>' | ':')
            || JOINED_PATH_OPTION_PREFIXES
                .into_iter()
                .any(|prefix| value[..start].ends_with(prefix))
    })
}

fn contains_unmapped_absolute_path(value: &str, role_tokens: &[String]) -> bool {
    // `compile_commands.json` stores one logical argument per array entry, so
    // an include path containing whitespace can arrive as
    // `-I"/ambient root/include"`. Splitting on whitespace before examining
    // the joined option tears that path apart and used to let it through.
    // Quotes do not contribute to path identity; examine the intact argument
    // once with them removed before the component-oriented checks below.
    let dequoted = value.replace(['\'', '"'], "");
    let begins_with_quote = matches!(value.trim_start().as_bytes().first(), Some(b'"' | b'\''));
    let has_joined_path_option = JOINED_PATH_OPTION_PREFIXES
        .into_iter()
        .any(|prefix| dequoted.starts_with(prefix));
    if has_windows_drive_prefix(value)
        || value.starts_with("//")
        || contains_unrecognized_angle_syntax(value, role_tokens)
        || contains_embedded_unix_absolute_path(value)
        || ((begins_with_quote || has_joined_path_option) && unix_absolute_path(&dequoted))
    {
        return true;
    }
    value
        .split(|character: char| character.is_ascii_whitespace() || matches!(character, '=' | ','))
        .any(|part| {
            part.split(';').any(|part| {
                let part = part.trim_matches(['\'', '"']);
                if role_tokens.iter().any(|token| part.contains(token)) {
                    return false;
                }
                // MSVC options legitimately start with one slash (`/std:c++20`,
                // `/D...`, `/DEFAULTLIB:...`). Check only positions that can begin
                // an absolute path: the whole argument or a known joined path
                // option. A stable relative argument such as `../mlx-cpp/turbo`
                // is not an ambient absolute path. CMake semicolon lists are
                // checked one element at a time so a mapped element cannot hide
                // a later unmapped absolute path.
                unix_absolute_path(part)
            })
        })
}

fn contains_embedded_unix_absolute_path(value: &str) -> bool {
    if value.contains("$<")
        && value.split(';').any(|part| {
            let part = part.trim_matches(['\'', '"']);
            part.starts_with('/') && unix_absolute_path(part)
        })
    {
        return true;
    }
    value.match_indices(":/").any(|(index, _)| {
        let tail = &value[index + 1..];
        let candidate = tail
            .split(|character: char| {
                character.is_ascii_whitespace() || matches!(character, ';' | ',' | '>')
            })
            .next()
            .unwrap_or(tail);
        if candidate.starts_with("//") {
            return value.trim_start().starts_with("$<");
        }
        unix_absolute_path(candidate)
    })
}

fn contains_unrecognized_angle_syntax(value: &str, role_tokens: &[String]) -> bool {
    let mut remaining = value.to_owned();
    for token in role_tokens {
        remaining = remaining.replace(token, "");
    }
    let openings = remaining.match_indices("$<").count();
    let closings = remaining.bytes().filter(|byte| *byte == b'>').count();
    let without_openings = remaining.replace("$<", "");
    without_openings.contains('<') || openings != closings
}

fn unix_absolute_path(part: &str) -> bool {
    let joined = JOINED_PATH_OPTION_PREFIXES
        .into_iter()
        .find_map(|prefix| part.strip_prefix(prefix));
    joined.unwrap_or(part).split(';').any(|candidate| {
        let candidate = candidate.trim_matches(['\'', '"']).trim_end_matches('>');
        if candidate.starts_with("//") {
            return true;
        }
        candidate.starts_with('/')
            && (joined.is_some()
                || candidate[1..].contains('/')
                || (!cfg!(windows) && !known_msvc_switch(candidate)))
    })
}

fn known_msvc_switch(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "/debug" | "/incremental" | "/utf-8" | "/bigobj"
    ) || lower.starts_with("/std:")
        || lower.starts_with("/defaultlib:")
        || lower.starts_with("/wd")
        || lower.starts_with("/zc:")
}

fn has_windows_drive_prefix(value: &str) -> bool {
    value
        .as_bytes()
        .windows(3)
        .enumerate()
        .any(|(index, window)| {
            window[0].is_ascii_alphabetic()
                && window[1] == b':'
                && window[2] == b'/'
                && path_prefix_boundary(value, index)
        })
}

#[allow(dead_code)]
fn _assert_send_safe_types(_: OsString, _: &OsStr) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_tree_binds_path_mode_content_and_order() {
        let baseline = vec![TreeEntry {
            path: "src/a".to_owned(),
            mode: 0o100644,
            bytes: b"a".to_vec(),
        }];
        let digest = canonical_tree_digest(&baseline).unwrap();
        for changed in [
            TreeEntry {
                path: "src/b".to_owned(),
                ..baseline[0].clone()
            },
            TreeEntry {
                mode: 0o100755,
                ..baseline[0].clone()
            },
            TreeEntry {
                bytes: b"b".to_vec(),
                ..baseline[0].clone()
            },
        ] {
            assert_ne!(canonical_tree_digest(&[changed]).unwrap(), digest);
        }
    }

    #[test]
    fn canonical_tree_rejects_content_beyond_its_byte_ceiling() {
        let mut budget = TreeByteBudget {
            retained: MAX_CANONICAL_TREE_BYTES,
        };
        assert!(budget.add(1).is_err());
    }

    #[test]
    fn canonical_tree_accepts_its_exact_content_byte_ceiling() {
        let mut budget = TreeByteBudget {
            retained: MAX_CANONICAL_TREE_BYTES - 1,
        };
        assert!(budget.add(1).is_ok());
        assert_eq!(budget.retained, MAX_CANONICAL_TREE_BYTES);
    }

    #[test]
    fn path_tokenization_rejects_unmapped_checkout_paths() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        fs::create_dir(&root).unwrap();
        let mapping = PathToken::new("<workspace>", &root).unwrap();
        let argument = format!("-I{}/include", mapping.root.display());
        assert_eq!(
            canonicalize_argument(&argument, &[mapping]).unwrap(),
            "-I<workspace>/include"
        );
        let nested_mapping = PathToken::new("<workspace>", &root).unwrap();
        let nested_argument = format!("-I{}/include/generated", nested_mapping.root.display());
        assert_eq!(
            canonicalize_argument(&nested_argument, &[nested_mapping]).unwrap(),
            "-I<workspace>/include/generated"
        );
        let unmapped = if cfg!(windows) {
            "-IC:/ambient/include"
        } else {
            "-I/ambient/include"
        };
        assert!(canonicalize_argument(unmapped, &[]).is_err());
        assert!(canonicalize_argument("/ambient/include", &[]).is_err());
        assert!(canonicalize_argument("-I/ambient/include", &[]).is_err());
        assert!(canonicalize_argument("-I\"/ambient root/include\"", &[]).is_err());
        assert!(canonicalize_argument("/I/ambient", &[]).is_err());
        assert!(canonicalize_argument("<not-a-role>/ambient/include", &[]).is_err());
        assert_eq!(
            canonicalize_argument("../mlx-cpp/turbo", &[]).unwrap(),
            "../mlx-cpp/turbo"
        );
        assert_eq!(
            canonicalize_argument("/std:c++20", &[]).unwrap(),
            "/std:c++20"
        );
        assert_eq!(
            canonicalize_argument("/DEFAULTLIB:kernel32.lib", &[]).unwrap(),
            "/DEFAULTLIB:kernel32.lib"
        );
        assert_eq!(
            canonicalize_argument("/debug /INCREMENTAL", &[]).unwrap(),
            "/debug /INCREMENTAL"
        );
        assert!(canonicalize_argument("\"/ambient root/include\"", &[]).is_err());
        assert_eq!(
            canonicalize_argument("$<INSTALL_INTERFACE:include>", &[]).unwrap(),
            "$<INSTALL_INTERFACE:include>"
        );
        assert_eq!(
            canonicalize_argument(
                "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>",
                &[]
            )
            .unwrap(),
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>"
        );
        assert_eq!(
            canonicalize_argument(
                "$<$<COMPILE_LANGUAGE:CXX>:/wd4244;/wd4267>;$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>",
                &[]
            )
            .unwrap(),
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4244;/wd4267>;$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>"
        );
        assert_eq!(
            canonicalize_argument(
                "$<$<COMPILE_LANGUAGE:CXX>:/wd4068;/wd4146>;$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244;-Xcompiler=/wd4267>;$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>",
                &[]
            )
            .unwrap(),
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4068;/wd4146>;$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244;-Xcompiler=/wd4267>;$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
        );
        assert!(canonicalize_argument("$<BUILD_INTERFACE:C:/ambient/include>", &[]).is_err());
        assert!(canonicalize_argument("$<BUILD_INTERFACE:/ambient/include>", &[]).is_err());
    }

    #[test]
    fn path_tokenization_rejects_list_tails_and_root_prefix_lookalikes() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        fs::create_dir(&root).unwrap();
        let mapping = PathToken::new("<workspace>", &root).unwrap();
        let framework_list = format!(
            "-F{}/System/Library/Frameworks;/ambient/include",
            mapping.root.display()
        );
        assert!(canonicalize_argument(&framework_list, &[mapping.clone()]).is_err());
        let plain_list = format!(
            "{}/System/Library/Frameworks;/ambient/include",
            mapping.root.display()
        );
        assert!(canonicalize_argument(&plain_list, &[mapping.clone()]).is_err());
        let expression_list = format!(
            "$<BUILD_INTERFACE:{}/include;/ambient/include>",
            mapping.root.display()
        );
        assert!(canonicalize_argument(&expression_list, &[mapping.clone()]).is_err());
        let lookalike = format!("-I{}-evil/include", mapping.root.display());
        assert!(canonicalize_argument(&lookalike, &[mapping]).is_err());
    }

    #[test]
    fn path_tokenization_rejects_single_component_and_unc_absolute_paths() {
        #[cfg(not(windows))]
        {
            assert!(canonicalize_argument("/tmp", &[]).is_err());
            assert!(canonicalize_argument("$<BUILD_INTERFACE:/tmp>", &[]).is_err());
        }
        assert!(canonicalize_argument("-I//server/share", &[]).is_err());
        assert!(canonicalize_argument("$<BUILD_INTERFACE://server/share>", &[]).is_err());
        assert_eq!(
            canonicalize_argument("-DURL=https://example.invalid/model", &[]).unwrap(),
            "-DURL=https://example.invalid/model"
        );
    }

    #[test]
    fn tool_output_is_bounded_and_line_ending_independent() {
        assert_eq!(
            canonical_tool_output(b" tool 1\r\n\r\nline 2\n".to_vec(), "tool").unwrap(),
            "tool 1 | line 2"
        );
        assert!(canonical_tool_output(b"bad\0value".to_vec(), "tool").is_err());
    }
}
