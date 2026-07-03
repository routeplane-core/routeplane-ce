//! Minimal JSON-Schema subset validator for the `json_schema` check.
//!
//! HONEST SCOPE (the parity bar is "response is parseable JSON matching a
//! provided structural schema", not Draft 2020-12 conformance — a full
//! validator crate would pull a heavy dependency tree onto the hot path):
//!
//! Supported keywords: `type` (string or array; object/array/string/number/
//! integer/boolean/null), `properties`, `required`, `items` (single-schema
//! form), `enum`, `minLength`/`maxLength` (chars), `minimum`/`maximum`,
//! `minItems`/`maxItems`, `additionalProperties` (boolean form only).
//!
//! Load-bearing keywords we do NOT implement are REJECTED at parse time
//! (silently ignoring `$ref` or `oneOf` would un-enforce the user's intent):
//! see [`UNSUPPORTED_KEYWORDS`]. Other unknown keywords are ignored, matching
//! standard JSON-Schema behavior. Recursion depth is capped at parse time.

use serde_json::Value;

const MAX_SCHEMA_DEPTH: usize = 16;

/// Keywords whose silent omission would change validation semantics badly —
/// rejected at parse time, fail loud.
const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "$ref",
    "$dynamicRef",
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "pattern",
    "patternProperties",
    "propertyNames",
    "prefixItems",
    "contains",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaType {
    Object,
    Array,
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

impl SchemaType {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "object" => SchemaType::Object,
            "array" => SchemaType::Array,
            "string" => SchemaType::String,
            "number" => SchemaType::Number,
            "integer" => SchemaType::Integer,
            "boolean" => SchemaType::Boolean,
            "null" => SchemaType::Null,
            other => return Err(format!("unsupported schema type '{other}'")),
        })
    }

    fn matches(self, v: &Value) -> bool {
        match self {
            SchemaType::Object => v.is_object(),
            SchemaType::Array => v.is_array(),
            SchemaType::String => v.is_string(),
            SchemaType::Boolean => v.is_boolean(),
            SchemaType::Null => v.is_null(),
            SchemaType::Number => v.is_number(),
            SchemaType::Integer => match v.as_f64() {
                Some(f) => f.is_finite() && f.fract() == 0.0,
                None => false,
            },
        }
    }

    fn name(self) -> &'static str {
        match self {
            SchemaType::Object => "object",
            SchemaType::Array => "array",
            SchemaType::String => "string",
            SchemaType::Number => "number",
            SchemaType::Integer => "integer",
            SchemaType::Boolean => "boolean",
            SchemaType::Null => "null",
        }
    }
}

/// A parsed schema node. Immutable, allocation happens at parse time only.
#[derive(Debug, Clone, Default)]
pub struct SchemaNode {
    types: Option<Vec<SchemaType>>,
    properties: Vec<(String, SchemaNode)>,
    required: Vec<String>,
    items: Option<Box<SchemaNode>>,
    enum_values: Option<Vec<Value>>,
    min_length: Option<u64>,
    max_length: Option<u64>,
    minimum: Option<f64>,
    maximum: Option<f64>,
    min_items: Option<u64>,
    max_items: Option<u64>,
    /// `Some(false)` => properties not named in `properties` are violations.
    additional_properties: Option<bool>,
}

impl SchemaNode {
    pub fn parse(value: &Value) -> Result<Self, String> {
        Self::parse_at(value, 0)
    }

    fn parse_at(value: &Value, depth: usize) -> Result<Self, String> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(format!("schema nesting exceeds {MAX_SCHEMA_DEPTH} levels"));
        }
        let obj = value
            .as_object()
            .ok_or_else(|| "schema must be a JSON object".to_string())?;

        for key in obj.keys() {
            if UNSUPPORTED_KEYWORDS.contains(&key.as_str()) {
                return Err(format!(
                    "unsupported schema keyword '{key}' (the Routeplane json_schema check \
                     supports a documented structural subset; rejecting rather than silently \
                     not enforcing)"
                ));
            }
        }

        let types = match obj.get("type") {
            None => None,
            Some(Value::String(s)) => Some(vec![SchemaType::parse(s)?]),
            Some(Value::Array(list)) => {
                let mut out = Vec::with_capacity(list.len());
                for t in list {
                    let s = t
                        .as_str()
                        .ok_or_else(|| "`type` array entries must be strings".to_string())?;
                    out.push(SchemaType::parse(s)?);
                }
                Some(out)
            }
            Some(_) => return Err("`type` must be a string or array of strings".to_string()),
        };

        let mut properties = Vec::new();
        if let Some(p) = obj.get("properties") {
            let map = p
                .as_object()
                .ok_or_else(|| "`properties` must be an object".to_string())?;
            for (name, sub) in map {
                properties.push((name.clone(), SchemaNode::parse_at(sub, depth + 1)?));
            }
        }

        let mut required = Vec::new();
        if let Some(r) = obj.get("required") {
            let list = r
                .as_array()
                .ok_or_else(|| "`required` must be an array".to_string())?;
            for entry in list {
                required.push(
                    entry
                        .as_str()
                        .ok_or_else(|| "`required` entries must be strings".to_string())?
                        .to_string(),
                );
            }
        }

        let items = match obj.get("items") {
            None => None,
            Some(i) => Some(Box::new(SchemaNode::parse_at(i, depth + 1)?)),
        };

        let enum_values = match obj.get("enum") {
            None => None,
            Some(e) => Some(
                e.as_array()
                    .ok_or_else(|| "`enum` must be an array".to_string())?
                    .clone(),
            ),
        };

        let get_u64 = |key: &str| -> Result<Option<u64>, String> {
            match obj.get(key) {
                None => Ok(None),
                Some(v) => v
                    .as_u64()
                    .map(Some)
                    .ok_or_else(|| format!("`{key}` must be a non-negative integer")),
            }
        };
        let get_f64 = |key: &str| -> Result<Option<f64>, String> {
            match obj.get(key) {
                None => Ok(None),
                Some(v) => v
                    .as_f64()
                    .map(Some)
                    .ok_or_else(|| format!("`{key}` must be a number")),
            }
        };

        let additional_properties = match obj.get("additionalProperties") {
            None => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(_) => {
                return Err(
                    "`additionalProperties` supports only the boolean form in this subset"
                        .to_string(),
                )
            }
        };

        Ok(Self {
            types,
            properties,
            required,
            items,
            enum_values,
            min_length: get_u64("minLength")?,
            max_length: get_u64("maxLength")?,
            minimum: get_f64("minimum")?,
            maximum: get_f64("maximum")?,
            min_items: get_u64("minItems")?,
            max_items: get_u64("maxItems")?,
            additional_properties,
        })
    }

    /// Validate `value`, returning the first violation as a `$`-rooted path
    /// message. Recursion is bounded by the parse-time depth cap.
    pub fn validate(&self, value: &Value) -> Result<(), String> {
        self.validate_at(value, "$")
    }

    fn validate_at(&self, value: &Value, path: &str) -> Result<(), String> {
        if let Some(types) = &self.types {
            if !types.iter().any(|t| t.matches(value)) {
                let expected: Vec<&str> = types.iter().map(|t| t.name()).collect();
                return Err(format!("{path}: expected type {}", expected.join("|")));
            }
        }
        if let Some(allowed) = &self.enum_values {
            if !allowed.iter().any(|a| a == value) {
                return Err(format!("{path}: value not in enum"));
            }
        }
        if let Some(s) = value.as_str() {
            let len = s.chars().count() as u64;
            if self.min_length.is_some_and(|m| len < m) {
                return Err(format!("{path}: string shorter than minLength"));
            }
            if self.max_length.is_some_and(|m| len > m) {
                return Err(format!("{path}: string longer than maxLength"));
            }
        }
        if let Some(n) = value.as_f64() {
            if self.minimum.is_some_and(|m| n < m) {
                return Err(format!("{path}: number below minimum"));
            }
            if self.maximum.is_some_and(|m| n > m) {
                return Err(format!("{path}: number above maximum"));
            }
        }
        if let Some(obj) = value.as_object() {
            for req in &self.required {
                if !obj.contains_key(req) {
                    return Err(format!("{path}: missing required property '{req}'"));
                }
            }
            for (name, sub) in &self.properties {
                if let Some(v) = obj.get(name) {
                    sub.validate_at(v, &format!("{path}.{name}"))?;
                }
            }
            if self.additional_properties == Some(false) {
                for key in obj.keys() {
                    if !self.properties.iter().any(|(n, _)| n == key) {
                        // The key name is INPUT-derived — do not echo it into
                        // the outcome detail (no PII reflection).
                        return Err(format!("{path}: unexpected additional property"));
                    }
                }
            }
        }
        if let Some(arr) = value.as_array() {
            let len = arr.len() as u64;
            if self.min_items.is_some_and(|m| len < m) {
                return Err(format!("{path}: fewer items than minItems"));
            }
            if self.max_items.is_some_and(|m| len > m) {
                return Err(format!("{path}: more items than maxItems"));
            }
            if let Some(items) = &self.items {
                for (i, v) in arr.iter().enumerate() {
                    items.validate_at(v, &format!("{path}[{i}]"))?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema(v: serde_json::Value) -> SchemaNode {
        SchemaNode::parse(&v).expect("schema should parse")
    }

    #[test]
    fn type_checks_each_primitive() {
        assert!(schema(json!({"type":"string"}))
            .validate(&json!("x"))
            .is_ok());
        assert!(schema(json!({"type":"string"}))
            .validate(&json!(1))
            .is_err());
        assert!(schema(json!({"type":"integer"}))
            .validate(&json!(3))
            .is_ok());
        assert!(schema(json!({"type":"integer"}))
            .validate(&json!(3.0))
            .is_ok()); // zero-fraction
        assert!(schema(json!({"type":"integer"}))
            .validate(&json!(3.5))
            .is_err());
        assert!(schema(json!({"type":["string","null"]}))
            .validate(&json!(null))
            .is_ok());
    }

    #[test]
    fn required_and_nested_properties() {
        let s = schema(json!({
            "type":"object",
            "required":["a"],
            "properties":{"a":{"type":"object","required":["b"],"properties":{"b":{"type":"number"}}}}
        }));
        assert!(s.validate(&json!({"a":{"b":1}})).is_ok());
        let err = s.validate(&json!({"a":{}})).unwrap_err();
        assert!(err.contains("$.a"), "path in: {err}");
        assert!(s.validate(&json!({})).is_err());
    }

    #[test]
    fn items_enum_and_array_bounds() {
        let s =
            schema(json!({"type":"array","minItems":1,"maxItems":2,"items":{"enum":["a","b"]}}));
        assert!(s.validate(&json!(["a"])).is_ok());
        assert!(s.validate(&json!([])).is_err());
        assert!(s.validate(&json!(["a", "b", "a"])).is_err());
        let err = s.validate(&json!(["a", "z"])).unwrap_err();
        assert!(err.contains("[1]"), "index in: {err}");
    }

    #[test]
    fn string_and_number_bounds() {
        let s = schema(json!({"type":"string","minLength":2,"maxLength":3}));
        assert!(s.validate(&json!("ab")).is_ok());
        assert!(s.validate(&json!("a")).is_err());
        let s = schema(json!({"type":"number","minimum":0,"maximum":1}));
        assert!(s.validate(&json!(0.5)).is_ok());
        assert!(s.validate(&json!(-1)).is_err());
    }

    #[test]
    fn additional_properties_false() {
        let s = schema(json!({"type":"object","properties":{"a":{}},"additionalProperties":false}));
        assert!(s.validate(&json!({"a":1})).is_ok());
        assert!(s.validate(&json!({"a":1,"b":2})).is_err());
    }

    #[test]
    fn unsupported_keywords_fail_parse_not_silently_ignore() {
        for kw in ["$ref", "oneOf", "pattern", "allOf", "not"] {
            let v = json!({ kw: {} });
            assert!(SchemaNode::parse(&v).is_err(), "{kw} must be rejected");
        }
    }

    #[test]
    fn depth_limit_is_enforced() {
        // Build a schema nested past the cap.
        let mut v = json!({"type":"string"});
        for _ in 0..(MAX_SCHEMA_DEPTH + 2) {
            v = json!({"type":"object","properties":{"x": v}});
        }
        assert!(SchemaNode::parse(&v).is_err());
    }
}
