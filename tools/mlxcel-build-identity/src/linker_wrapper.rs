use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use mlxcel_build_identity::canonical::{self, PathToken};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TokenConfig {
    token: String,
    root: PathBuf,
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("mlxcel-linker-wrapper: {error}");
            std::process::exit(2);
        }
    }
}

fn run() -> Result<i32, String> {
    let real_linker = PathBuf::from(required_os("MLXCEL_REAL_LINKER")?);
    if !real_linker.is_file() {
        return Err(format!(
            "MLXCEL_REAL_LINKER is not a file: {}",
            real_linker.display()
        ));
    }
    let trace_path = PathBuf::from(required_os("MLXCEL_LINK_TRACE")?);
    let artifact = PathBuf::from(required_os("MLXCEL_LINK_ARTIFACT")?);
    let mappings: Vec<TokenConfig> = serde_json::from_str(&required("MLXCEL_LINK_PATH_TOKENS")?)
        .map_err(|err| format!("MLXCEL_LINK_PATH_TOKENS is invalid: {err}"))?;
    let mut tokens = mappings
        .iter()
        .map(|mapping| PathToken::new(&mapping.token, &mapping.root))
        .collect::<Result<Vec<_>, String>>()?;
    let artifact_parent = artifact
        .parent()
        .ok_or_else(|| "link artifact has no parent".to_owned())?;
    tokens.push(PathToken::new("<artifact-dir>", artifact_parent)?);

    let original = env::args_os().skip(1).collect::<Vec<_>>();
    if original.is_empty() {
        return Err("linker wrapper received no arguments".to_owned());
    }
    let mut invoked = original.clone();
    let link_map = env::var_os("MLXCEL_LINK_MAP").map(PathBuf::from);
    if env::var_os("MLXCEL_REQUIRE_LINK_MAP").is_some() && link_map.is_none() {
        return Err("strict link wrapper requires MLXCEL_LINK_MAP".to_owned());
    }
    if let Some(map) = &link_map {
        append_map_argument(&mut invoked, map)?;
    }
    // Expand response files once and pass those exact arguments to the real
    // linker. Reading them only after a successful link leaves a TOCTOU window
    // in which the recorded argv can differ from what the linker consumed.
    let expanded = expand_response_arguments(&invoked, 0)?;

    let status = Command::new(&real_linker)
        .args(&expanded)
        .status()
        .map_err(|err| format!("cannot execute {}: {err}", real_linker.display()))?;
    let code = status.code().unwrap_or(1);
    if !status.success() {
        return Ok(code);
    }
    if !artifact.is_file() {
        return Err(format!(
            "successful linker did not create {}",
            artifact.display()
        ));
    }
    verify_link_output(&expanded, &artifact)?;
    if let Some(map) = &link_map
        && !map.is_file()
    {
        return Err(format!(
            "successful linker did not create map {}",
            map.display()
        ));
    }

    let canonical_arguments = expanded
        .iter()
        .map(|argument| canonical::canonicalize_argument(argument, &tokens))
        .collect::<Result<Vec<_>, String>>()?;
    let archives = capture_archives(&expanded, &canonical_arguments)?;
    let identity = linker_identity(&real_linker)?;
    let output_name = artifact
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "link output name is not strict UTF-8".to_owned())?;
    let trace = serde_json::json!({
        "schema": "mlxcel.final-link-trace.v1",
        "linker_identity": identity,
        "archiver_identity": if archives.is_empty() {
            None
        } else {
            Some(archive_tool_identity()?)
        },
        "output_name": output_name,
        "arguments": canonical_arguments,
        "archives": archives,
    });
    let bytes = serde_json::to_vec_pretty(&trace)
        .map_err(|err| format!("cannot encode link trace: {err}"))?;
    atomic_write(&trace_path, &bytes)?;
    Ok(code)
}

fn expand_response_arguments(arguments: &[OsString], depth: usize) -> Result<Vec<String>, String> {
    if depth > 8 {
        return Err("linker response-file nesting exceeds 8".to_owned());
    }
    let mut expanded = Vec::new();
    for argument in arguments {
        let argument = argument
            .to_str()
            .ok_or_else(|| "linker argument is not strict UTF-8".to_owned())?;
        if let Some(path) = argument.strip_prefix('@') {
            let path = path.trim_matches('"');
            let bytes = fs::read(path)
                .map_err(|err| format!("cannot read linker response file {path}: {err}"))?;
            if bytes.len() > 16 * 1024 * 1024 {
                return Err("linker response file exceeds 16 MiB".to_owned());
            }
            let text = String::from_utf8(bytes)
                .map_err(|_| format!("linker response file {path} is not UTF-8"))?;
            let nested = split_response_file(&text)?
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>();
            expanded.extend(expand_response_arguments(&nested, depth + 1)?);
        } else {
            expanded.push(argument.to_owned());
        }
        if expanded.len() > canonical::MAX_CANONICAL_ITEMS {
            return Err("expanded linker argument list exceeds the item bound".to_owned());
        }
    }
    Ok(expanded)
}

fn split_response_file(text: &str) -> Result<Vec<String>, String> {
    if cfg!(windows) {
        split_windows_response_file(text)
    } else {
        split_unix_response_file(text)
    }
}

fn split_windows_response_file(text: &str) -> Result<Vec<String>, String> {
    let characters = text.chars().collect::<Vec<_>>();
    let mut values = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if character == '\\' {
            let start = index;
            while index < characters.len() && characters[index] == '\\' {
                index += 1;
            }
            let count = index - start;
            if index < characters.len() && characters[index] == '"' {
                current.extend(std::iter::repeat_n('\\', count / 2));
                if count % 2 == 0 {
                    quoted = !quoted;
                } else {
                    current.push('"');
                }
                index += 1;
            } else {
                current.extend(std::iter::repeat_n('\\', count));
            }
            continue;
        }
        if character == '"' {
            quoted = !quoted;
        } else if character.is_whitespace() && !quoted {
            if !current.is_empty() {
                values.push(std::mem::take(&mut current));
            }
        } else {
            current.push(character);
        }
        index += 1;
    }
    if quoted {
        return Err("linker response file ends inside a quote".to_owned());
    }
    if !current.is_empty() {
        values.push(current);
    }
    Ok(values)
}

fn split_unix_response_file(text: &str) -> Result<Vec<String>, String> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escape = false;
    for character in text.chars() {
        if escape {
            current.push(character);
            escape = false;
            continue;
        }
        match quote {
            Some('\'') => {
                if character == '\'' {
                    quote = None;
                } else {
                    current.push(character);
                }
            }
            Some('"') => {
                if character == '"' {
                    quote = None;
                } else if character == '\\' {
                    escape = true;
                } else {
                    current.push(character);
                }
            }
            Some(_) => unreachable!(),
            None if character == '\\' => escape = true,
            None if matches!(character, '\'' | '"') => quote = Some(character),
            None if character.is_whitespace() => {
                if !current.is_empty() {
                    values.push(std::mem::take(&mut current));
                }
            }
            None => current.push(character),
        }
    }
    if quote.is_some() || escape {
        return Err("linker response file ends inside an escape or quote".to_owned());
    }
    if !current.is_empty() {
        values.push(current);
    }
    Ok(values)
}

fn verify_link_output(arguments: &[String], configured: &Path) -> Result<(), String> {
    let mut outputs = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if argument == "-o" {
            let output = arguments
                .get(index + 1)
                .ok_or_else(|| "final linker -o has no output path".to_owned())?;
            outputs.push(output.as_str());
            index += 2;
            continue;
        }
        if argument
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("/out:"))
        {
            outputs.push(argument.get(5..).unwrap_or_default());
        } else if let Some(output) = argument.strip_prefix("-Wl,-o,") {
            outputs.push(output);
        } else if let Some(output) = argument.strip_prefix("-Wl,-o=") {
            outputs.push(output);
        }
        index += 1;
    }
    if outputs.len() != 1 {
        return Err(format!(
            "final linker must select exactly one output, observed {}",
            outputs.len()
        ));
    }
    let output = PathBuf::from(outputs[0].trim_matches('"'));
    let output = fs::canonicalize(&output).map_err(|err| {
        format!(
            "cannot resolve observed link output {}: {err}",
            output.display()
        )
    })?;
    let configured = fs::canonicalize(configured).map_err(|err| {
        format!(
            "cannot resolve configured link artifact {}: {err}",
            configured.display()
        )
    })?;
    if output != configured {
        return Err(format!(
            "observed link output {} does not match configured artifact {}",
            output.display(),
            configured.display()
        ));
    }
    Ok(())
}

fn capture_archives(
    expanded: &[String],
    canonical_arguments: &[String],
) -> Result<Vec<serde_json::Value>, String> {
    let search_directories = archive_search_directories(expanded)?;
    let mut archives = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, argument) in expanded.iter().enumerate() {
        let Some(path) = resolve_archive_argument(argument, &search_directories)? else {
            continue;
        };
        let canonical = fs::canonicalize(&path)
            .map_err(|err| format!("cannot canonicalize archive {}: {err}", path.display()))?;
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let mut members = archive_members(&canonical)?;
        members.sort();
        archives.push(serde_json::json!({
            "argument_index": index,
            "label": canonical_arguments[index],
            "sha256": canonical::hex_digest(canonical::file_sha256(&canonical)?),
            "members_sorted": members,
        }));
    }
    Ok(archives)
}

fn archive_search_directories(arguments: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut raw = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if argument == "-L" {
            raw.push(
                arguments
                    .get(index + 1)
                    .ok_or_else(|| "final linker -L has no directory".to_owned())?
                    .as_str(),
            );
            index += 2;
            continue;
        }
        if let Some(path) = argument.strip_prefix("-L").filter(|path| !path.is_empty()) {
            raw.push(path);
        } else if argument
            .get(..9)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("/libpath:"))
        {
            raw.push(argument.get(9..).unwrap_or_default());
        } else if let Some(path) = argument.strip_prefix("-Wl,-L,") {
            raw.push(path);
        } else if let Some(path) = argument.strip_prefix("-Wl,-L") {
            raw.push(path);
        }
        index += 1;
    }
    let library_environment = env::var_os("LIB")
        .map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .unwrap_or_default();
    let mut directories = Vec::new();
    let mut seen = BTreeSet::new();
    for path in raw
        .into_iter()
        .map(|path| PathBuf::from(path.trim_matches('"')))
        .chain(library_environment)
    {
        if !path.is_dir() {
            continue;
        }
        let path = fs::canonicalize(&path).map_err(|err| {
            format!(
                "cannot canonicalize archive search {}: {err}",
                path.display()
            )
        })?;
        if seen.insert(path.clone()) {
            directories.push(path);
        }
    }
    Ok(directories)
}

fn resolve_archive_argument(
    argument: &str,
    search_directories: &[PathBuf],
) -> Result<Option<PathBuf>, String> {
    let direct = argument.trim_matches('"');
    let candidate = if let Some(path) = direct.strip_prefix("-Wl,-force_load,") {
        Some(path)
    } else if direct
        .get(..14)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("/wholearchive:"))
    {
        direct.get(14..)
    } else if let Some(name) = direct.strip_prefix("-l:") {
        return resolve_searched_archive(name, search_directories);
    } else if direct
        .get(..12)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("/defaultlib:"))
    {
        let name = direct.get(12..).unwrap_or_default();
        return resolve_searched_archive(name.trim_matches('"'), search_directories);
    } else {
        Some(direct)
    };
    let Some(candidate) = candidate else {
        return Ok(None);
    };
    let path = PathBuf::from(candidate.trim_matches('"'));
    if !is_archive_path(&path) {
        return Ok(None);
    }
    if path.is_file() {
        return Ok(Some(path));
    }
    if path.components().count() == 1 {
        return resolve_searched_archive(candidate, search_directories);
    }
    Err(format!(
        "final linker archive input is unavailable: {}",
        path.display()
    ))
}

fn resolve_searched_archive(
    name: &str,
    search_directories: &[PathBuf],
) -> Result<Option<PathBuf>, String> {
    let name = Path::new(name);
    if !is_archive_path(name) {
        return Ok(None);
    }
    for directory in search_directories {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }
    Err(format!(
        "final linker named archive input is absent from its search closure: {}",
        name.display()
    ))
}

fn is_archive_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("a" | "lib" | "rlib")
    )
}

fn archive_members(path: &Path) -> Result<Vec<String>, String> {
    let ar = configured_archive_tool();
    let resolved = resolve_executable(&ar)?;
    let is_msvc_library_manager = resolved
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| {
            name.eq_ignore_ascii_case("lib.exe") || name.eq_ignore_ascii_case("llvm-lib.exe")
        });
    let mut command = Command::new(&resolved);
    if is_msvc_library_manager {
        command.args(["/NOLOGO", "/LIST"]);
    } else {
        command.arg("t");
    }
    let output = command
        .arg(path)
        .output()
        .map_err(|err| format!("cannot inspect archive {}: {err}", path.display()))?;
    if !output.status.success() {
        return Err(format!("archive traversal failed for {}", path.display()));
    }
    if output.stdout.len() > 32 * 1024 * 1024 {
        return Err("archive member listing exceeds 32 MiB".to_owned());
    }
    let members = String::from_utf8(output.stdout)
        .map_err(|_| format!("archive member list is not UTF-8: {}", path.display()))?
        .lines()
        .map(|value| {
            if value.is_empty() || value.len() > canonical::MAX_CANONICAL_TEXT_BYTES {
                Err("archive contains an invalid member name".to_owned())
            } else {
                Ok(value.to_owned())
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    if members.len() > canonical::MAX_CANONICAL_ITEMS {
        return Err("archive member listing exceeds the item bound".to_owned());
    }
    Ok(members)
}

fn linker_identity(path: &Path) -> Result<String, String> {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "real linker name is not strict UTF-8".to_owned())?;
    let is_msvc = file_name.eq_ignore_ascii_case("link.exe");
    let arguments = if is_msvc {
        &["/?"][..]
    } else if cfg!(target_os = "macos") {
        &["-v"][..]
    } else {
        &["--version"][..]
    };
    let output = Command::new(path)
        .args(arguments)
        .output()
        .map_err(|err| format!("cannot execute final linker identity probe: {err}"))?;
    let mut version = output.stdout;
    version.extend(output.stderr);
    let version = canonical::canonical_tool_output(version, "final linker identity")?;
    if (!output.status.success() && !is_msvc && !cfg!(target_os = "macos"))
        || (is_msvc && !version.contains("Incremental Linker Version"))
    {
        return Err("final linker identity probe failed or lacked its version banner".to_owned());
    }
    Ok(format!(
        "{file_name}|sha256={}|{version}",
        canonical::hex_digest(canonical::file_sha256(path)?)
    ))
}

fn archive_tool_identity() -> Result<String, String> {
    let configured = configured_archive_tool();
    let path = resolve_executable(&configured)?;
    let is_msvc_library_manager = path
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| {
            name.eq_ignore_ascii_case("lib.exe") || name.eq_ignore_ascii_case("llvm-lib.exe")
        });
    let mut observed = None;
    let probes: &[&[&str]] = if is_msvc_library_manager {
        &[&["/?"]]
    } else {
        &[&["--version"], &["-V"], &["-h"]]
    };
    for arguments in probes {
        let Ok(output) = Command::new(&path).args(*arguments).output() else {
            continue;
        };
        if !output.status.success() && !is_msvc_library_manager {
            continue;
        }
        let mut bytes = output.stdout;
        bytes.extend(output.stderr);
        if let Ok(value) = canonical::canonical_tool_output(bytes, "archive-tool identity") {
            observed = Some(value);
            break;
        }
    }
    let version = observed.ok_or_else(|| "archive-tool identity probe failed".to_owned())?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "archive-tool name is not strict UTF-8".to_owned())?;
    Ok(format!(
        "{name}|sha256={}|{version}",
        canonical::hex_digest(canonical::file_sha256(&path)?)
    ))
}

fn configured_archive_tool() -> OsString {
    env::var_os("MLXCEL_AR").unwrap_or_else(|| {
        if cfg!(target_env = "msvc") {
            OsString::from("lib.exe")
        } else {
            OsString::from("ar")
        }
    })
}

fn resolve_executable(value: &std::ffi::OsStr) -> Result<PathBuf, String> {
    let candidate = PathBuf::from(value);
    if candidate.is_file() {
        return fs::canonicalize(&candidate)
            .map_err(|err| format!("cannot canonicalize {}: {err}", candidate.display()));
    }
    let search = env::var_os("PATH").ok_or_else(|| "PATH is unavailable".to_owned())?;
    let extensions = if cfg!(windows) {
        env::var_os("PATHEXT")
            .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".into())
            .to_string_lossy()
            .split(';')
            .map(str::to_owned)
            .collect::<Vec<_>>()
    } else {
        vec![String::new()]
    };
    for directory in env::split_paths(&search) {
        for extension in &extensions {
            let mut name = value.to_os_string();
            if !extension.is_empty() && Path::new(value).extension().is_none() {
                name.push(extension);
            }
            let path = directory.join(name);
            if path.is_file() {
                return fs::canonicalize(&path)
                    .map_err(|err| format!("cannot canonicalize {}: {err}", path.display()));
            }
        }
    }
    Err(format!("cannot resolve build tool {value:?}"))
}

fn append_map_argument(arguments: &mut Vec<OsString>, map: &Path) -> Result<(), String> {
    let map = map
        .to_str()
        .ok_or_else(|| "link map path is not strict UTF-8".to_owned())?;
    if cfg!(target_os = "macos") {
        arguments.push(format!("-Wl,-map,{map}").into());
    } else if cfg!(target_env = "msvc") {
        arguments.push(format!("/MAP:{map}").into());
    } else {
        arguments.push(format!("-Wl,-Map,{map}").into());
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "link trace path has no parent".to_owned())?;
    fs::create_dir_all(parent)
        .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, bytes)
        .map_err(|err| format!("cannot write {}: {err}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|err| format!("cannot publish {}: {err}", path.display()))
}

fn required(name: &str) -> Result<String, String> {
    required_os(name)?
        .into_string()
        .map_err(|_| format!("{name} is not strict UTF-8"))
}

fn required_os(name: &str) -> Result<OsString, String> {
    env::var_os(name).ok_or_else(|| format!("required environment {name} is missing"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    #[test]
    fn response_expansion_is_ordered_and_fallible() {
        assert_eq!(
            split_response_file("a \"b c\" d").unwrap(),
            ["a", "b c", "d"]
        );
        assert!(split_response_file("\"unterminated").is_err());
    }

    #[test]
    fn windows_response_parser_preserves_quoted_path_backslashes() {
        assert_eq!(
            split_windows_response_file(r#"/OUT:"C:\Program Files\runner.exe" "C:\libs\mlx.lib""#)
                .unwrap(),
            [r#"/OUT:C:\Program Files\runner.exe"#, r#"C:\libs\mlx.lib"#]
        );
    }

    #[test]
    fn unix_response_parser_preserves_shell_quoted_arguments() {
        assert_eq!(
            split_unix_response_file(r#"-L'/ambient path/lib' escaped\ value "quoted value""#)
                .unwrap(),
            ["-L/ambient path/lib", "escaped value", "quoted value"]
        );
        assert!(split_unix_response_file("'unterminated").is_err());
    }

    #[test]
    fn configured_artifact_must_be_the_observed_link_output() {
        let temp = tempfile::tempdir().unwrap();
        let artifact = temp.path().join("runner.exe");
        let other = temp.path().join("other.exe");
        fs::write(&artifact, b"runner").unwrap();
        fs::write(&other, b"other").unwrap();
        assert!(
            verify_link_output(
                &["-o".to_owned(), artifact.to_string_lossy().into_owned()],
                &artifact
            )
            .is_ok()
        );
        assert!(
            verify_link_output(&[format!("/OUT:{}", other.to_string_lossy())], &artifact).is_err()
        );
        assert!(verify_link_output(&["/NOLOGO".to_owned()], &artifact).is_err());
    }

    #[test]
    fn archive_resolution_covers_rlibs_searches_and_force_load() {
        let temp = tempfile::tempdir().unwrap();
        let rlib = temp.path().join("dependency.rlib");
        let unix = temp.path().join("libnative.a");
        let msvc = temp.path().join("native.lib");
        fs::write(&rlib, b"rlib").unwrap();
        fs::write(&unix, b"archive").unwrap();
        fs::write(&msvc, b"library").unwrap();
        let search = archive_search_directories(&[
            "-L".to_owned(),
            temp.path().to_string_lossy().into_owned(),
        ])
        .unwrap();

        assert_eq!(
            resolve_archive_argument(&rlib.to_string_lossy(), &search).unwrap(),
            Some(rlib)
        );
        assert_eq!(
            resolve_archive_argument("-l:libnative.a", &search).unwrap(),
            Some(fs::canonicalize(unix).unwrap())
        );
        assert_eq!(
            resolve_archive_argument("/DEFAULTLIB:native.lib", &search).unwrap(),
            Some(fs::canonicalize(msvc).unwrap())
        );
        assert_eq!(
            resolve_archive_argument(
                &format!(
                    "-Wl,-force_load,{}",
                    search[0].join("libnative.a").display()
                ),
                &search,
            )
            .unwrap(),
            Some(fs::canonicalize(search[0].join("libnative.a")).unwrap())
        );
    }

    #[test]
    fn link_trace_digest_changes_for_every_argument_mutation() {
        let mut first = sha2::Sha256::new();
        first.update(serde_json::to_vec(&vec!["-o", "a"]).unwrap());
        let mut second = sha2::Sha256::new();
        second.update(serde_json::to_vec(&vec!["-o", "b"]).unwrap());
        assert_ne!(first.finalize(), second.finalize());
    }
}
