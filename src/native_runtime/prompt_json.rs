// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Transport-neutral CPython `json.dumps` compatible `tojson` filter.
//!
//! Native prompt receipts supply the byte and node ceilings. The server facade
//! supplies its established policy ceiling. All output and traversal checks run
//! before the corresponding allocation or append.

use std::fmt::Write as _;

use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Error, ErrorKind};

const MAX_DEPTH: usize = 192;
const _: () = assert!(MAX_DEPTH > 128);
const MAX_INDENT: usize = 64;
pub(crate) const SERVER_PROMPT_JSON_MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
const SERVER_MAX_NODES: usize = super::MAX_NATIVE_PROMPT_CONTEXT_NODES;

#[derive(Debug)]
struct PromptJsonResourceLimit;

impl std::fmt::Display for PromptJsonResourceLimit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("prompt JSON resource limit")
    }
}

impl std::error::Error for PromptJsonResourceLimit {}

fn resource_error() -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        "prompt JSON exceeds its selected byte or node limit",
    )
    .with_source(PromptJsonResourceLimit)
}

fn depth_error() -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("tojson: value nests deeper than {MAX_DEPTH} levels"),
    )
    .with_source(PromptJsonResourceLimit)
}

pub(crate) fn is_prompt_json_resource_limit(error: &Error) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(cause) = current {
        if cause.downcast_ref::<PromptJsonResourceLimit>().is_some() {
            return true;
        }
        current = cause.source();
    }
    false
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DumpOptions {
    ensure_ascii: bool,
    indent: Option<usize>,
    item_separator: String,
    key_separator: String,
    sort_keys: bool,
}

impl DumpOptions {
    fn resolve(
        positional_indent: Option<Value>,
        kwargs: &Kwargs,
        max_output_bytes: usize,
    ) -> Result<Self, Error> {
        let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
        let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
        let indent_arg = match positional_indent {
            Some(value) => Some(value),
            None => kwargs.get::<Option<Value>>("indent")?,
        };
        let indent = match indent_arg {
            None => None,
            Some(value) => match bool::try_from(value.clone()) {
                Ok(true) => Some(2),
                Ok(false) => None,
                Err(_) => Some(usize::try_from(value)?.min(MAX_INDENT)),
            },
        };
        let (item_separator, key_separator) = match kwargs.get::<Option<Value>>("separators")? {
            Some(value) => read_separators(&value, max_output_bytes)?,
            None if indent.is_some() => (",".to_string(), ": ".to_string()),
            None => (", ".to_string(), ": ".to_string()),
        };
        Ok(Self {
            ensure_ascii,
            indent,
            item_separator,
            key_separator,
            sort_keys,
        })
    }
}

fn read_separators(value: &Value, max_output_bytes: usize) -> Result<(String, String), Error> {
    let invalid = || {
        Error::new(
            ErrorKind::InvalidOperation,
            "tojson: `separators` must be a two element tuple of strings, e.g. (\",\", \": \")",
        )
    };
    if value.kind() != ValueKind::Seq {
        return Err(invalid());
    }
    let parts: Vec<Value> = value.try_iter()?.take(3).collect();
    let [item, key] = parts.as_slice() else {
        return Err(invalid());
    };
    let item = item.as_str().ok_or_else(invalid)?;
    let key = key.as_str().ok_or_else(invalid)?;
    if item.len() > max_output_bytes || key.len() > max_output_bytes {
        return Err(resource_error());
    }
    Ok((item.to_string(), key.to_string()))
}

struct BoundedOutput {
    value: String,
    maximum: usize,
}

impl BoundedOutput {
    fn new(maximum: usize) -> Self {
        Self {
            value: String::new(),
            maximum,
        }
    }

    fn ensure_additional(&mut self, additional: usize) -> Result<(), Error> {
        let next = self
            .value
            .len()
            .checked_add(additional)
            .filter(|next| *next <= self.maximum)
            .ok_or_else(resource_error)?;
        debug_assert!(next <= self.maximum);
        self.value
            .try_reserve(additional)
            .map_err(|_| resource_error())
    }

    fn push(&mut self, value: char) -> Result<(), Error> {
        self.ensure_additional(value.len_utf8())?;
        self.value.push(value);
        Ok(())
    }

    fn push_str(&mut self, value: &str) -> Result<(), Error> {
        self.ensure_additional(value.len())?;
        self.value.push_str(value);
        Ok(())
    }

    fn into_string(self) -> String {
        self.value
    }
}

struct JsonState {
    output: BoundedOutput,
    nodes: usize,
    max_nodes: usize,
}

impl JsonState {
    fn new(max_output_bytes: usize, max_nodes: usize) -> Result<Self, Error> {
        if max_output_bytes == 0 || max_nodes == 0 {
            return Err(resource_error());
        }
        Ok(Self {
            output: BoundedOutput::new(max_output_bytes),
            nodes: 0,
            max_nodes,
        })
    }

    fn enter_node(&mut self, level: usize) -> Result<(), Error> {
        if level > MAX_DEPTH {
            return Err(depth_error());
        }
        if self.nodes == self.max_nodes {
            return Err(resource_error());
        }
        self.nodes += 1;
        Ok(())
    }

    fn remaining_nodes(&self) -> usize {
        self.max_nodes.saturating_sub(self.nodes)
    }
}

pub(crate) fn python_tojson_bounded(
    value: Value,
    indent: Option<Value>,
    kwargs: Kwargs,
    max_output_bytes: usize,
    max_nodes: usize,
) -> Result<Value, Error> {
    let options = DumpOptions::resolve(indent, &kwargs, max_output_bytes)?;
    kwargs.assert_all_used()?;
    let mut state = JsonState::new(max_output_bytes, max_nodes)?;
    write_value(&mut state, &value, &options, 0)?;
    Ok(Value::from_safe_string(state.output.into_string()))
}

pub(crate) fn python_tojson_for_server(
    value: Value,
    indent: Option<Value>,
    kwargs: Kwargs,
) -> Result<Value, Error> {
    python_tojson_bounded(
        value,
        indent,
        kwargs,
        SERVER_PROMPT_JSON_MAX_OUTPUT_BYTES,
        SERVER_MAX_NODES,
    )
}

#[cfg(test)]
fn python_tojson(value: Value, indent: Option<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    python_tojson_for_server(value, indent, kwargs)
}

fn write_value(
    state: &mut JsonState,
    value: &Value,
    options: &DumpOptions,
    level: usize,
) -> Result<(), Error> {
    state.enter_node(level)?;
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => state.output.push_str("null")?,
        ValueKind::Bool => state
            .output
            .push_str(if value.is_true() { "true" } else { "false" })?,
        ValueKind::Number => write_number(&mut state.output, value)?,
        ValueKind::String => write_json_string(
            &mut state.output,
            value.as_str().unwrap_or_default(),
            options.ensure_ascii,
        )?,
        ValueKind::Plain => {
            let rendered = value.to_string();
            write_json_string(&mut state.output, &rendered, options.ensure_ascii)?;
        }
        ValueKind::Bytes => {
            let bytes = value.as_bytes().unwrap_or_default();
            write_array(
                state,
                bytes.iter().copied().map(Value::from),
                options,
                level,
            )?;
        }
        ValueKind::Seq | ValueKind::Iterable => {
            write_array(state, value.try_iter()?, options, level)?;
        }
        ValueKind::Map => write_object(state, value, options, level)?,
        other => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("tojson: cannot serialize a {other} value"),
            ));
        }
    }
    Ok(())
}

fn write_container_break(
    output: &mut BoundedOutput,
    options: &DumpOptions,
    level: usize,
) -> Result<(), Error> {
    if let Some(width) = options.indent {
        let fill = width.checked_mul(level).ok_or_else(resource_error)?;
        output.ensure_additional(1usize.checked_add(fill).ok_or_else(resource_error)?)?;
        output.value.push('\n');
        output.value.extend(std::iter::repeat_n(' ', fill));
    }
    Ok(())
}

fn write_array(
    state: &mut JsonState,
    items: impl Iterator<Item = Value>,
    options: &DumpOptions,
    level: usize,
) -> Result<(), Error> {
    state.output.push('[')?;
    let mut count = 0usize;
    for item in items {
        if count > 0 {
            state.output.push_str(&options.item_separator)?;
        }
        write_container_break(&mut state.output, options, level + 1)?;
        write_value(state, &item, options, level + 1)?;
        count = count.checked_add(1).ok_or_else(resource_error)?;
    }
    if count > 0 {
        write_container_break(&mut state.output, options, level)?;
    }
    state.output.push(']')
}

fn write_object(
    state: &mut JsonState,
    value: &Value,
    options: &DumpOptions,
    level: usize,
) -> Result<(), Error> {
    let mut entries: Vec<(String, Value)> = Vec::new();
    for key in value.try_iter()? {
        if entries.len() == state.remaining_nodes() {
            return Err(resource_error());
        }
        let item = value.get_item(&key)?;
        entries.push((object_key(&key, state.output.maximum)?, item));
    }
    if options.sort_keys {
        entries.sort_by(|left, right| left.0.cmp(&right.0));
    }
    if entries.is_empty() {
        return state.output.push_str("{}");
    }
    state.output.push('{')?;
    for (index, (key, item)) in entries.iter().enumerate() {
        if index > 0 {
            state.output.push_str(&options.item_separator)?;
        }
        write_container_break(&mut state.output, options, level + 1)?;
        write_json_string(&mut state.output, key, options.ensure_ascii)?;
        state.output.push_str(&options.key_separator)?;
        write_value(state, item, options, level + 1)?;
    }
    write_container_break(&mut state.output, options, level)?;
    state.output.push('}')
}

fn object_key(key: &Value, maximum: usize) -> Result<String, Error> {
    let rendered = match key.kind() {
        ValueKind::String => {
            let key = key.as_str().unwrap_or_default();
            if key.len() > maximum {
                return Err(resource_error());
            }
            key.to_string()
        }
        ValueKind::Bool => if key.is_true() { "true" } else { "false" }.to_string(),
        ValueKind::Undefined | ValueKind::None => "null".to_string(),
        ValueKind::Number => {
            let mut output = BoundedOutput::new(maximum);
            write_number(&mut output, key)?;
            output.into_string()
        }
        _ => key.to_string(),
    };
    if rendered.len() > maximum {
        return Err(resource_error());
    }
    Ok(rendered)
}

fn write_number(output: &mut BoundedOutput, value: &Value) -> Result<(), Error> {
    if value.is_integer() {
        return output.push_str(&value.to_string());
    }
    output.push_str(&format_python_float(f64::try_from(value.clone())?))
}

fn format_python_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        };
    }
    let scientific = format!("{value:e}");
    let (sign, unsigned) = match scientific.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", scientific.as_str()),
    };
    let Some((mantissa, exponent)) = unsigned.split_once('e') else {
        return scientific;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return scientific;
    };
    let digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let decpt = exponent + 1;
    let mut out = String::with_capacity(digits.len() + 8);
    out.push_str(sign);
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let exponent = decpt - 1;
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let magnitude = exponent.unsigned_abs();
        if magnitude < 10 {
            out.push('0');
        }
        let _ = write!(out, "{magnitude}");
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        out.push_str(&digits);
        for _ in 0..decpt as usize - digits.len() {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        let split = decpt as usize;
        out.push_str(&digits[..split]);
        out.push('.');
        out.push_str(&digits[split..]);
    }
    out
}

fn write_json_string(
    output: &mut BoundedOutput,
    text: &str,
    ensure_ascii: bool,
) -> Result<(), Error> {
    let encoded =
        text.chars().try_fold(2usize, |length, character| {
            let additional =
                match character {
                    '"' | '\\' | '\n' | '\r' | '\t' | '\u{8}' | '\u{c}' => 2,
                    _ if (character as u32) < 0x20 => 6,
                    _ if ensure_ascii && (character as u32) > 0x7e => {
                        if (character as u32) < 0x1_0000 { 6 } else { 12 }
                    }
                    _ => character.len_utf8(),
                };
            length.checked_add(additional)
        });
    output.ensure_additional(encoded.ok_or_else(resource_error)?)?;
    output.value.push('"');
    for character in text.chars() {
        match character {
            '"' => output.value.push_str("\\\""),
            '\\' => output.value.push_str("\\\\"),
            '\n' => output.value.push_str("\\n"),
            '\r' => output.value.push_str("\\r"),
            '\t' => output.value.push_str("\\t"),
            '\u{8}' => output.value.push_str("\\b"),
            '\u{c}' => output.value.push_str("\\f"),
            _ if (character as u32) < 0x20 => {
                let _ = write!(output.value, "\\u{:04x}", character as u32);
            }
            _ if ensure_ascii && (character as u32) > 0x7e => {
                let code = character as u32;
                if code < 0x1_0000 {
                    let _ = write!(output.value, "\\u{code:04x}");
                } else {
                    let offset = code - 0x1_0000;
                    let high = 0xd800 | (offset >> 10);
                    let low = 0xdc00 | (offset & 0x3ff);
                    let _ = write!(output.value, "\\u{high:04x}\\u{low:04x}");
                }
            }
            _ => output.value.push(character),
        }
    }
    output.value.push('"');
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_output_limit_accepts_exact_and_rejects_plus_one() {
        let empty_kwargs = || std::iter::empty::<(&str, Value)>().collect::<Kwargs>();
        let exact =
            python_tojson_bounded(Value::from("123456"), None, empty_kwargs(), 8, 1).unwrap();
        assert_eq!(exact.to_string(), "\"123456\"");

        let error =
            python_tojson_bounded(Value::from("1234567"), None, empty_kwargs(), 8, 1).unwrap_err();
        assert!(is_prompt_json_resource_limit(&error));

        let sequence = Value::from(vec![Value::from(1), Value::from(2)]);
        assert!(
            python_tojson_bounded(sequence.clone(), None, empty_kwargs(), 8, 3).is_ok(),
            "sequence root plus two items is the exact node edge"
        );
        let error = python_tojson_bounded(sequence, None, empty_kwargs(), 8, 2).unwrap_err();
        assert!(is_prompt_json_resource_limit(&error));
    }

    /// Render `value` through the filter with the given kwargs.
    fn dump(value: Value, kwargs: &[(&str, Value)]) -> String {
        let kwargs: Kwargs = kwargs.iter().map(|(k, v)| (*k, v.clone())).collect();
        python_tojson(value, None, kwargs)
            .expect("filter must succeed")
            .to_string()
    }

    fn float_str(value: f64) -> String {
        format_python_float(value)
    }

    #[test]
    fn python_float_repr_matches_cpython_for_the_threshold_cases() {
        // Every expected value here is `repr(x)` in CPython 3. The three columns
        // Rust's `{}` gets wrong are the `.0` suffix, the exponent thresholds,
        // and the two-digit signed exponent.
        assert_eq!(float_str(1.0), "1.0");
        assert_eq!(float_str(2.5), "2.5");
        assert_eq!(float_str(-0.0), "-0.0");
        assert_eq!(float_str(0.0), "0.0");
        assert_eq!(float_str(-2.5), "-2.5");

        // Lower exponent threshold: switch at decpt <= -4, i.e. below 1e-4.
        assert_eq!(float_str(0.0001), "0.0001");
        assert_eq!(float_str(0.00001), "1e-05");
        assert_eq!(float_str(0.000123), "0.000123");
        assert_eq!(float_str(1.5e-5), "1.5e-05");
        assert_eq!(float_str(1e-323), "1e-323");

        // Upper exponent threshold: switch at decpt > 16, i.e. above 1e15.
        assert_eq!(float_str(1e15), "1000000000000000.0");
        assert_eq!(float_str(1e16), "1e+16");
        assert_eq!(float_str(1.0000000000000002e16), "1.0000000000000002e+16");
        assert_eq!(float_str(1e20), "1e+20");
        assert_eq!(float_str(1.7976931348623157e308), "1.7976931348623157e+308");

        // Shortest round-trip digits, not a fixed precision.
        assert_eq!(float_str(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(float_str(1.0 / 3.0), "0.3333333333333333");

        // `json.dumps` emits these bare, which is not valid JSON but is what the
        // reference renderer produces.
        assert_eq!(float_str(f64::NAN), "NaN");
        assert_eq!(float_str(f64::INFINITY), "Infinity");
        assert_eq!(float_str(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn every_finite_float_round_trips_through_the_repr() {
        // The formatter reassembles shortest round-trip digits by hand, so the
        // property that actually matters is that the text still parses back to
        // the same bits.
        for value in [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            2.5,
            1e-5,
            1e-4,
            1e15,
            1e16,
            1e300,
            1e-300,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            std::f64::consts::PI,
            0.1 + 0.2,
        ] {
            let text = format_python_float(value);
            let parsed: f64 = text.parse().expect("repr must parse back");
            assert_eq!(
                parsed.to_bits(),
                value.to_bits(),
                "{text} did not round-trip for {value:?}"
            );
        }
    }

    #[test]
    fn integers_and_floats_are_kept_apart() {
        assert_eq!(dump(Value::from(2), &[]), "2");
        assert_eq!(dump(Value::from(2.0), &[]), "2.0");
        assert_eq!(dump(Value::from(-7i64), &[]), "-7");
        assert_eq!(dump(Value::from(u64::MAX), &[]), "18446744073709551615");
    }

    #[test]
    fn default_separators_have_the_python_spaces() {
        let value = Value::from_serialize(serde_json::json!({"b": [1, 2], "a": "x"}));
        assert_eq!(dump(value, &[]), r#"{"b": [1, 2], "a": "x"}"#);
    }

    #[test]
    fn compact_separators_and_sort_keys_apply_recursively() {
        let value = Value::from_serialize(serde_json::json!({
            "b": {"z": 1, "y": 2},
            "a": [{"n": 1, "m": 2}]
        }));
        let rendered = dump(
            value,
            &[
                ("sort_keys", Value::from(true)),
                (
                    "separators",
                    Value::from(vec![Value::from(","), Value::from(":")]),
                ),
            ],
        );
        assert_eq!(rendered, r#"{"a":[{"m":2,"n":1}],"b":{"y":2,"z":1}}"#);
    }

    #[test]
    fn ensure_ascii_escapes_non_ascii_and_surrogate_pairs() {
        let value = Value::from("é 😀 \u{7f}");
        assert_eq!(
            dump(value.clone(), &[("ensure_ascii", Value::from(true))]),
            r#""\u00e9 \ud83d\ude00 \u007f""#
        );
        // The default leaves UTF-8 alone, U+007F included.
        assert_eq!(dump(value, &[]), "\"é 😀 \u{7f}\"");
    }

    #[test]
    fn control_characters_use_the_python_escape_table() {
        let value = Value::from("a\nb\tc\rd\u{8}e\u{c}f\u{1}g\"h\\i");
        assert_eq!(dump(value, &[]), r#""a\nb\tc\rd\be\ff\u0001g\"h\\i""#);
    }

    #[test]
    fn html_characters_are_not_escaped() {
        // minijinja's builtin turns these four into \\u003c, \\u003e, \\u0026
        // and \\u0027 so its output can be inlined into an HTML document.
        // `transformers` does not, and a JSON Schema `pattern` full of `<`
        // is a different prompt than the provider tokenized.
        let value = Value::from("a < b && c > d, it's fine");
        assert_eq!(dump(value, &[]), r#""a < b && c > d, it's fine""#);
    }

    #[test]
    fn indent_switches_to_python_item_separators() {
        let value = Value::from_serialize(serde_json::json!({"a": 1, "b": [1, 2]}));
        let rendered = dump(value, &[("indent", Value::from(2))]);
        assert_eq!(
            rendered,
            "{\n  \"a\": 1,\n  \"b\": [\n    1,\n    2\n  ]\n}"
        );
    }

    #[test]
    fn indent_true_means_two_spaces() {
        let value = Value::from_serialize(serde_json::json!({"a": 1}));
        assert_eq!(
            dump(value, &[("indent", Value::from(true))]),
            "{\n  \"a\": 1\n}"
        );
    }

    #[test]
    fn empty_containers_stay_on_one_line_under_indent() {
        let value = Value::from_serialize(serde_json::json!({"a": {}, "b": []}));
        assert_eq!(
            dump(value, &[("indent", Value::from(2))]),
            "{\n  \"a\": {},\n  \"b\": []\n}"
        );
    }

    #[test]
    fn explicit_separators_win_over_the_indent_default() {
        let value = Value::from_serialize(serde_json::json!({"a": 1, "b": 2}));
        let rendered = dump(
            value,
            &[
                ("indent", Value::from(2)),
                (
                    "separators",
                    Value::from(vec![Value::from(","), Value::from(" = ")]),
                ),
            ],
        );
        assert_eq!(rendered, "{\n  \"a\" = 1,\n  \"b\" = 2\n}");
    }

    #[test]
    fn null_and_bool_scalars_match_python() {
        assert_eq!(dump(Value::from(()), &[]), "null");
        assert_eq!(dump(Value::from(true), &[]), "true");
        assert_eq!(dump(Value::from(false), &[]), "false");
        assert_eq!(dump(Value::UNDEFINED, &[]), "null");
    }

    #[test]
    fn non_string_object_keys_are_coerced_like_python() {
        let mut map = std::collections::BTreeMap::new();
        map.insert(Value::from(1), Value::from("a"));
        map.insert(Value::from(true), Value::from("b"));
        let rendered = dump(Value::from(map), &[("sort_keys", Value::from(true))]);
        assert_eq!(rendered, r#"{"1": "a", "true": "b"}"#);
    }

    #[test]
    fn unknown_keyword_arguments_are_still_rejected() {
        let kwargs: Kwargs = [("cls", Value::from("Encoder"))].into_iter().collect();
        let err = python_tojson(Value::from(1), None, kwargs).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::TooManyArguments);
    }

    #[test]
    fn malformed_separators_are_rejected() {
        let kwargs: Kwargs = [("separators", Value::from(vec![Value::from(",")]))]
            .into_iter()
            .collect();
        let err = python_tojson(Value::from(1), None, kwargs).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    }

    #[test]
    fn deeply_nested_values_are_refused_rather_than_overflowing_the_stack() {
        let mut value = Value::from(1);
        for _ in 0..MAX_DEPTH + 5 {
            value = Value::from(vec![value]);
        }
        let err = python_tojson(
            value,
            None,
            Kwargs::from_iter(Vec::<(String, Value)>::new()),
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    }

    #[test]
    fn map_insertion_order_is_preserved_by_default() {
        // The `preserve_order` guard: a client-sent JSON Schema must reach the
        // prompt in wire order, not alphabetized.
        let value: serde_json::Value =
            serde_json::from_str(r#"{"pattern":1,"include":2,"a":3}"#).unwrap();
        assert_eq!(
            dump(Value::from_serialize(value), &[]),
            r#"{"pattern": 1, "include": 2, "a": 3}"#
        );
    }

    /// Render `value` and return the error message, expecting failure.
    fn dump_err(value: Value, kwargs: &[(&str, Value)]) -> String {
        let kwargs: Kwargs = kwargs.iter().map(|(k, v)| (*k, v.clone())).collect();
        python_tojson(value, None, kwargs)
            .expect_err("filter must reject this input")
            .to_string()
    }

    #[test]
    fn indent_width_is_clamped_instead_of_emitting_unbounded_padding() {
        // `indent` is template-supplied. Without the clamp, `width * level` both
        // overflows `usize` for a large enough width and emits gigabytes of
        // spaces below that threshold. Clamping keeps the render alive, which
        // matters because a failed render costs the template its tools block.
        let value: serde_json::Value = serde_json::from_str(r#"[1]"#).unwrap();
        let huge = dump(
            Value::from_serialize(&value),
            &[("indent", Value::from(u64::MAX))],
        );
        let clamped = dump(
            Value::from_serialize(&value),
            &[("indent", Value::from(MAX_INDENT as u64))],
        );
        assert_eq!(huge, clamped);
        assert_eq!(huge, format!("[\n{}1\n]", " ".repeat(MAX_INDENT)));
    }

    #[test]
    fn separators_arity_is_checked_before_the_sequence_is_materialized() {
        // A three-element sequence is still rejected, but the check must not
        // require collecting the whole thing first: `separators` is template
        // supplied and could be an enormous lazy range.
        let value: serde_json::Value = serde_json::from_str(r#"[1]"#).unwrap();
        let message = dump_err(
            Value::from_serialize(&value),
            &[(
                "separators",
                Value::from(vec![
                    Value::from(","),
                    Value::from(":"),
                    Value::from("extra"),
                ]),
            )],
        );
        assert!(
            message.contains("two element tuple"),
            "expected an arity rejection, got: {message}"
        );
    }

    #[test]
    fn depth_limit_leaves_headroom_over_the_request_parsers_own_ceiling() {
        // `chat_request` splices a separately-parsed `function.arguments` into
        // the message tree, so an assembled value can exceed serde_json's own
        // 128-level limit. The cap has to sit above that or the filter errors on
        // a legitimate request and the render falls back to the generic prompt.
        // The headroom itself is pinned by a `const` assertion at the definition
        // site, so this test covers the runtime behavior at the cap.
        //
        // Past the cap the filter still errors rather than recursing without
        // bound. Build the nesting iteratively so the test itself cannot blow
        // the stack constructing its input.
        let mut nested = serde_json::Value::from(1u64);
        for _ in 0..(MAX_DEPTH + 2) {
            nested = serde_json::Value::Array(vec![nested]);
        }
        let message = dump_err(Value::from_serialize(&nested), &[]);
        assert!(
            message.contains("nests deeper than"),
            "expected a depth rejection, got: {message}"
        );
    }
}
