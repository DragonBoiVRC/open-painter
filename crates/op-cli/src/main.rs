//! Agent control for open-painter.
//!
//!   op-cli call <command> ['{"json":"params"}']   run one command on the running app
//!   op-cli mcp                                    MCP server on stdio for Claude and other agents
//!
//! Both talk to `op-app --control <port>` (default 7878); if nothing is listening they start it.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_PORT: u16 = 7878;

struct App {
    port: u16,
    conn: Option<(TcpStream, BufReader<TcpStream>)>,
    next_id: u64,
}

impl App {
    fn connect(&mut self) -> Result<(), String> {
        if self.conn.is_some() {
            return Ok(());
        }
        let try_connect = |port| TcpStream::connect(("127.0.0.1", port)).ok();
        let stream = match try_connect(self.port) {
            Some(s) => s,
            None => {
                // op-app sits next to this binary when built or installed together; otherwise it is on PATH.
                let sibling = std::env::current_exe().ok().and_then(|e| e.with_file_name("op-app").canonicalize().ok());
                Command::new(sibling.unwrap_or_else(|| "op-app".into()))
                    .args(["--control", &self.port.to_string()])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .spawn()
                    .map_err(|e| format!("could not start op-app: {e}"))?;
                let start = Instant::now();
                loop {
                    if let Some(s) = try_connect(self.port) {
                        break s;
                    }
                    if start.elapsed() > Duration::from_secs(60) {
                        return Err(format!("op-app did not open its control port {}", self.port));
                    }
                    std::thread::sleep(Duration::from_millis(300));
                }
            }
        };
        self.conn = Some((stream.try_clone().map_err(|e| e.to_string())?, BufReader::new(stream)));
        Ok(())
    }

    fn call(&mut self, cmd: &str, params: Value) -> Result<Value, String> {
        self.connect()?;
        self.next_id += 1;
        let (w, r) = self.conn.as_mut().expect("connected above");
        let sent = writeln!(w, "{}", json!({"id": self.next_id, "cmd": cmd, "params": params})).and_then(|_| w.flush());
        let mut line = String::new();
        if sent.is_err() || r.read_line(&mut line).unwrap_or(0) == 0 {
            self.conn = None; // the app closed; the next call reconnects (or restarts it)
            return Err("lost the connection to op-app".into());
        }
        let reply: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if reply["ok"] == true {
            Ok(reply["result"].clone())
        } else {
            Err(reply["error"].as_str().unwrap_or("failed").to_string())
        }
    }
}

fn mcp(app: &mut App) {
    let out = std::io::stdout();
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
        let Some(id) = msg.get("id").cloned() else { continue }; // notifications need no answer
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let result: Result<Value, (i32, String)> = match msg["method"].as_str().unwrap_or_default() {
            "initialize" => Ok(json!({
                "protocolVersion": params.get("protocolVersion").cloned().unwrap_or(json!("2025-03-26")),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "open-painter", "version": env!("CARGO_PKG_VERSION")},
                "instructions": "Drives a running open-painter. Start with `info` (layers, texture sets, export presets), then add_layer / paint_uv / bake / export. Use `screenshot` to look at the model.",
            })),
            "ping" => Ok(json!({})),
            "tools/list" => app.call("commands", Value::Null).map_err(|e| (-32000, e)).map(|list| {
                let tools: Vec<Value> = list
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|c| json!({"name": c["name"], "description": c["help"], "inputSchema": {"type": "object", "additionalProperties": true}}))
                    .collect();
                json!({"tools": tools})
            }),
            "tools/call" => {
                let name = params["name"].as_str().unwrap_or_default().to_string();
                let args = params.get("arguments").cloned().unwrap_or(Value::Null);
                Ok(match app.call(&name, args) {
                    Ok(v) => json!({"content": [{"type": "text", "text": serde_json::to_string_pretty(&v).unwrap_or_default()}]}),
                    Err(e) => json!({"content": [{"type": "text", "text": e}], "isError": true}),
                })
            }
            other => Err((-32601, format!("method not found: {other}"))),
        };
        let reply = match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err((code, message)) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
        };
        let mut out = out.lock();
        let _ = writeln!(out, "{reply}").and_then(|_| out.flush());
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut port = DEFAULT_PORT;
    if let Some(i) = args.iter().position(|a| a == "--port") {
        port = args.get(i + 1).and_then(|p| p.parse().ok()).unwrap_or_else(|| usage());
        args.drain(i..=i + 1);
    }
    let mut app = App { port, conn: None, next_id: 0 };
    match args.first().map(String::as_str) {
        Some("mcp") => mcp(&mut app),
        Some("call") if args.len() >= 2 => {
            let params = args.get(2).map_or(Ok(Value::Null), |s| serde_json::from_str(s)).unwrap_or_else(|e| {
                eprintln!("params are not JSON: {e}");
                std::process::exit(2)
            });
            match app.call(&args[1], params) {
                Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default()),
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        }
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage: op-cli [--port N] call <command> [json-params]\n       op-cli [--port N] mcp\n(run `op-cli call commands` for the command list)");
    std::process::exit(2)
}
