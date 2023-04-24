use std::fs;
use pdf_extract;
use regex::Regex;
use std::path::Path;
use std::time::Duration;

use reqwest;
use select::document::Document;
use select::predicate::Name;

use tokio;
use screenshots::Screen;
use leptess::{leptonica, tesseract};

const TMP_SC_PATH: &str = "/tmp/erica_sc.png";
const TMP_TEXT_PATH: &str = "/tmp/erica_text.txt";
const TMP_PREVIOUS_FILE_PARSED: &str = "/tmp/erica_previous_file_parsed.txt";

fn write_file(file_path: &str, content: &str) -> Result<(), std::io::Error> {
    fs::write(file_path, content)?;
    Ok(())
}

fn read_file(file_path: &str) -> Result<String, std::io::Error> {
    let content = fs::read_to_string(file_path)?;
    Ok(content)
}

fn take_sc() -> Result<(), Box<dyn std::error::Error>> {
    let screens = Screen::all().unwrap();
    let sc = screens[0].capture().unwrap();
    let buffer = sc.buffer();
    fs::write(TMP_SC_PATH, buffer).unwrap();
    Ok(())
}

fn extract_text_from_webpage(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let html = reqwest::blocking::get(reqwest::Url::parse(url)?)?.text()?;
    let text = Document::from(html.as_str())
        .find(Name("p"))
        .map(|p| p.text())
        .collect::<Vec<_>>()
        .join("\n");

    Ok(text)
}

fn extract_text_from_pdf(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let out = pdf_extract::extract_text(file_path);
    match out {
        Ok(out) => Ok(out),
        Err(e) => Err(Box::new(e)),
    }
}

fn extract_text_from_image(img_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut api = tesseract::TessApi::new(None, "eng").unwrap();
    let pix = leptonica::pix_read(Path::new(img_path)).unwrap();
    api.set_image(&pix);
    let text = api.get_utf8_text().unwrap();
    Ok(text)
}


fn does_contain_url(s: &str) -> (bool, String) {
    let re = Regex::new(r#"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)
                           (?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))
                           (?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"#).unwrap();

    if let Some(cap) = re.captures(s) {
        return (true, cap[0].to_string());
    }

    (false, "".to_string())
}

#[tokio::main]
async fn main() {
    loop {
        take_sc().unwrap();
        let context = extract_text_from_image(TMP_SC_PATH).unwrap();
        let (contains_url, url) = does_contain_url(&context);
        if contains_url && url != read_file(TMP_PREVIOUS_FILE_PARSED).unwrap() {
            // check if ends with .pdf
            if url.ends_with(".pdf") {
                let text = extract_text_from_pdf(&url).unwrap();
                write_file(TMP_TEXT_PATH, &text).unwrap();
                write_file(TMP_PREVIOUS_FILE_PARSED, &url).unwrap();
                println!("Context: {}", text);
            } else {
                let text = extract_text_from_webpage(&url).unwrap();
                write_file(TMP_TEXT_PATH, &text).unwrap();
                write_file(TMP_PREVIOUS_FILE_PARSED, &url).unwrap();
                println!("Context: {}", text);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            continue;
        }
        write_file(TMP_TEXT_PATH, &context).unwrap();
        println!("Context: {}", context);
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
