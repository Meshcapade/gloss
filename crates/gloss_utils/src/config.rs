// use config_manager::{config, ConfigInit};
// use config::{File, FileStoredFormat, Format, Map, Value, ValueKind};
use config::{ConfigError, FileStoredFormat, Format, Map, Value, ValueKind};
// use serde::de::Unexpected;
use std::error::Error;
// use std::io::{Error, ErrorKind};

#[derive(Debug, Clone)]
pub struct MyTomlFile;

impl Format for MyTomlFile {
    fn parse(&self, uri: Option<&String>, text: &str) -> Result<Map<String, Value>, Box<dyn Error + Send + Sync>> {
        let value = from_toml_value(uri, &toml::from_str(text)?);

        // Have a proper error fire if the root of a file is ever not a Table
        // https://github.com/mehcode/config-rs/blob/master/src/format.rs#L28
        match value.kind {
            ValueKind::Table(map) => Ok(map),
            _ => Err(ConfigError::Message("The config is not a table".to_string())),
        }
        .map_err(|err| Box::new(err) as Box<dyn Error + Send + Sync>)
    }
}
// A slice of extensions associated to this format, when an extension
// is omitted from a file source, these will be tried implicitly:
impl FileStoredFormat for MyTomlFile {
    fn file_extensions(&self) -> &'static [&'static str] {
        &["toml"]
    }
}

//pretty much all from https://github.com/mehcode/config-rs/blob/master/src/file/format/toml.rs
//the adition is that the word "auto" is set to nil so that we know to
// automatically set that value later
fn from_toml_value(uri: Option<&String>, value: &toml::Value) -> Value {
    match *value {
        toml::Value::String(ref value) => {
            if value.to_string().to_lowercase() == "auto" {
                Value::new(uri, ValueKind::Nil)
            } else {
                Value::new(uri, value.to_string())
            }
        }
        toml::Value::Float(value) => Value::new(uri, value),
        toml::Value::Integer(value) => Value::new(uri, value),
        toml::Value::Boolean(value) => Value::new(uri, value),

        toml::Value::Table(ref table) => {
            let mut m = Map::new();

            for (key, value) in table {
                m.insert(key.clone(), from_toml_value(uri, value));
            }

            Value::new(uri, m)
        }

        toml::Value::Array(ref array) => {
            let mut l = Vec::new();

            for value in array {
                l.push(from_toml_value(uri, value));
            }

            Value::new(uri, l)
        }

        toml::Value::Datetime(ref datetime) => Value::new(uri, datetime.to_string()),
    }
}
