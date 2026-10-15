//! File-name templates, in the style Substance Painter presets use:
//!
//!   `$mesh_$textureSet_BaseColor(_$colorSpace)(.$udim)`
//!
//! `$name` is a variable. A parenthesised group is optional: it is dropped entirely when
//! any variable inside it expands to nothing. Values and the finished name are sanitised
//! so a template can never produce a path outside the export folder.

/// Values for the variables we know. Variables with an empty value drop their optional group.
#[derive(Clone, Debug, Default)]
pub struct Vars<'a> {
    pub project: &'a str,
    pub texture_set: &'a str,
    pub mesh: &'a str,
}

/// Variable names a template may use. Anything else is reported at import time.
pub const KNOWN_VARS: [&str; 5] = ["project", "textureSet", "mesh", "colorSpace", "udim"];

fn value<'a>(name: &str, v: &Vars<'a>) -> &'a str {
    match name {
        "project" => v.project,
        "textureSet" => v.texture_set,
        "mesh" => v.mesh,
        // We export a single texture set without UDIM tiles and do not tag the color space.
        _ => "",
    }
}

/// Make a substituted value safe inside a file name: only plain characters survive, runs of
/// dots collapse to one underscore (so `..` can never appear), and a leading dot is removed.
/// A single dot inside a name (`my.project v1.2`) is kept.
fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut dots = 0;
    for c in s.chars() {
        if c == '.' {
            dots += 1;
            continue;
        }
        if dots == 1 && !out.is_empty() {
            out.push('.');
        } else if dots > 1 {
            out.push('_');
        }
        dots = 0;
        out.push(if c.is_ascii_alphanumeric() || "-_ ".contains(c) { c } else { '_' });
    }
    if dots == 1 && !out.is_empty() {
        out.push('.');
    } else if dots > 1 {
        out.push('_');
    }
    out
}

/// Names of variables used in `template`, in order, without duplicates.
pub fn variables(template: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '$' {
            let mut name = String::new();
            while let Some(&n) = chars.peek() {
                if n.is_ascii_alphabetic() {
                    name.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            if !name.is_empty() && !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

/// Reject templates that cannot be expanded sensibly (unbalanced or nested groups).
pub fn check_template(template: &str) -> Result<(), String> {
    let mut depth = 0i32;
    for c in template.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }
        if !(0..=1).contains(&depth) {
            return Err("unbalanced or nested parentheses in file name".into());
        }
    }
    if depth != 0 {
        return Err("unbalanced parentheses in file name".into());
    }
    Ok(())
}

fn expand_segment(seg: &str, vars: &Vars<'_>) -> (String, bool) {
    let mut out = String::new();
    let mut all_present = true;
    let mut chars = seg.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '$' {
            let mut name = String::new();
            while let Some(&n) = chars.peek() {
                if n.is_ascii_alphabetic() {
                    name.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            let val = value(&name, vars);
            if val.is_empty() {
                all_present = false;
            }
            out.push_str(&sanitize(val));
        } else {
            out.push(c);
        }
    }
    (out, all_present)
}

/// Expand `template`. Returns an error for malformed templates or an empty result.
pub fn expand_template(template: &str, vars: &Vars<'_>) -> Result<String, String> {
    check_template(template)?;
    let mut out = String::new();
    let mut rest = template;
    while !rest.is_empty() {
        match rest.find('(') {
            None => {
                out.push_str(&expand_segment(rest, vars).0);
                break;
            }
            Some(open) => {
                out.push_str(&expand_segment(&rest[..open], vars).0);
                let close = rest[open..].find(')').map(|i| open + i).ok_or("unbalanced parentheses")?;
                let (text, present) = expand_segment(&rest[open + 1..close], vars);
                if present {
                    out.push_str(&text);
                }
                rest = &rest[close + 1..];
            }
        }
    }
    let name = out.trim().to_string();
    if name.is_empty() || name.starts_with('.') || name.contains("..") {
        return Err(format!("template '{template}' expands to an unusable file name '{name}'"));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars() -> Vars<'static> {
        Vars { project: "proj", texture_set: "body", mesh: "hero" }
    }

    #[test]
    fn variables_expand_and_optional_groups_with_empty_values_vanish() {
        assert_eq!(
            expand_template("$mesh_$textureSet_BaseColor(_$colorSpace)(.$udim)", &vars()).unwrap(),
            "hero_body_BaseColor"
        );
        assert_eq!(expand_template("t_$textureSet_Normal(.$udim)", &vars()).unwrap(), "t_body_Normal");
    }

    #[test]
    fn optional_group_with_present_values_is_kept_without_its_parentheses() {
        assert_eq!(expand_template("a(_$mesh)b", &vars()).unwrap(), "a_herob");
    }

    #[test]
    fn values_cannot_inject_path_components() {
        let evil = Vars { project: "../../x", texture_set: "a/b\\c", mesh: "m" };
        let name = expand_template("$project_$textureSet", &evil).unwrap();
        assert!(!name.contains('/') && !name.contains('\\'), "{name}");
        assert!(!name.contains(".."), "{name}");
    }

    #[test]
    fn sanitising_keeps_single_dots_and_removes_dangerous_ones() {
        assert_eq!(sanitize("my.project v1.2"), "my.project v1.2");
        assert_eq!(sanitize("../../evil"), "____evil");
        for hostile in ["..", "a..b", "...", ".hidden", "a/../b", "x\\..\\y", "tail."] {
            let s = sanitize(hostile);
            assert!(!s.contains("..") && !s.contains('/') && !s.contains('\\') && !s.starts_with('.'), "{hostile:?} -> {s:?}");
        }
    }

    #[test]
    fn malformed_templates_are_rejected() {
        assert!(expand_template("a(b", &vars()).is_err());
        assert!(expand_template("a)b(", &vars()).is_err());
        assert!(expand_template("a((b))", &vars()).is_err(), "nesting");
        assert!(expand_template("$udim", &vars()).is_err(), "expands to nothing");
        assert!(expand_template(".hidden", &vars()).is_err());
    }

    #[test]
    fn variables_are_listed_without_duplicates() {
        assert_eq!(variables("$mesh_$textureSet_$mesh(.$udim)"), ["mesh", "textureSet", "udim"]);
        assert!(variables("plain").is_empty());
    }
}
