use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::signal;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::timeout;

const DEFAULT_SOCKET: &str = "/tmp/lv-mcp-bridge.sock";
const PROTOCOL_VERSION: &str = "2024-11-05";

// ---------- Wire types ----------

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<u64>,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

// ---------- Client protocol types ----------

#[derive(Debug, Deserialize)]
struct AddServerRequest {
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default = "default_timeout")]
    timeout: u64,
    #[serde(default = "default_init_timeout")]
    init_timeout: u64,
}

#[derive(Debug, Deserialize)]
struct CallToolRequest {
    name: String,
    tool: String,
    #[serde(default)]
    arguments: Value,
    #[serde(default = "default_timeout")]
    timeout: u64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd")]
enum ClientCommand {
    #[serde(rename = "add_server")]
    AddServer(AddServerRequest),
    #[serde(rename = "list_tools")]
    ListTools { name: String },
    #[serde(rename = "call_tool")]
    CallTool(CallToolRequest),
    #[serde(rename = "remove_server")]
    RemoveServer { name: String },
    #[serde(rename = "status")]
    Status,
}

fn default_timeout() -> u64 {
    60
}
fn default_init_timeout() -> u64 {
    120
}

// ---------- Server state ----------

#[derive(Debug, Clone, Serialize)]
struct ToolInfo {
    name: String,
    description: Option<String>,
    parameters: Option<Value>,
}

struct ServerConnection {
    name: String,
    #[allow(dead_code)]
    child: Child,
    writer_tx: mpsc::UnboundedSender<String>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>,
    next_id: AtomicU64,
    tools: Mutex<Vec<ToolInfo>>,
    error: Mutex<Option<String>>,
}

impl ServerConnection {
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    async fn call(&self, method: &str, params: Option<Value>) -> Result<Value, String> {
        let id = self.next_id();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: method.to_string(),
            params,
        };
        let line = serde_json::to_string(&req).map_err(|e| e.to_string())?;

        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().await;
            pending.insert(id, tx);
        }

        if self.writer_tx.send(line).is_err() {
            self.pending.lock().await.remove(&id);
            return Err("server writer closed".to_string());
        }

        match rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(e),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err("response channel dropped".to_string())
            }
        }
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), String> {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: method.to_string(),
            params,
        };
        let line = serde_json::to_string(&req).map_err(|e| e.to_string())?;
        self.writer_tx
            .send(line)
            .map_err(|_| "server writer closed".to_string())
    }

    async fn set_error(&self, err: String) {
        let mut pending = self.pending.lock().await;
        for (_, tx) in pending.drain() {
            let _ = tx.send(Err(err.clone()));
        }
        *self.error.lock().await = Some(err);
    }
}

// ---------- Bridge state ----------

struct BridgeState {
    servers: HashMap<String, Arc<ServerConnection>>,
}

impl BridgeState {
    fn new() -> Self {
        Self {
            servers: HashMap::new(),
        }
    }
}

type SharedState = Arc<Mutex<BridgeState>>;

// ---------- Response helper ----------

#[derive(Debug, Serialize)]
struct ClientResponse {
    success: bool,
    output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
}

fn ok(output: impl Into<String>, metadata: Option<Value>) -> ClientResponse {
    ClientResponse {
        success: true,
        output: output.into(),
        error: None,
        metadata,
    }
}

fn err(error: impl Into<String>) -> ClientResponse {
    ClientResponse {
        success: false,
        output: String::new(),
        error: Some(error.into()),
        metadata: None,
    }
}

async fn write_response<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    resp: ClientResponse,
) -> tokio::io::Result<()> {
    let line = serde_json::to_string(&resp).unwrap();
    writer.write_all(line.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await
}

// ---------- Server lifecycle ----------

async fn add_server(state: SharedState, req: AddServerRequest) -> ClientResponse {
    let mut cmd = Command::new(&req.command);
    cmd.args(&req.args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());

    for (k, v) in &req.env {
        cmd.env(k, v);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return err(format!("failed to spawn '{}': {}", req.command, e)),
    };

    let stdin = child.stdin.take().expect("stdin was piped");
    let stdout = child.stdout.take().expect("stdout was piped");

    let (writer_tx, mut writer_rx) = mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        let mut writer = BufWriter::new(stdin);
        while let Some(line) = writer_rx.recv().await {
            if writer.write_all(line.as_bytes()).await.is_err() {
                break;
            }
            if writer.write_all(b"\n").await.is_err() {
                break;
            }
            if writer.flush().await.is_err() {
                break;
            }
        }
    });

    let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let server = Arc::new(ServerConnection {
        name: req.name.clone(),
        child,
        writer_tx,
        pending: pending.clone(),
        next_id: AtomicU64::new(3),
        tools: Mutex::new(Vec::new()),
        error: Mutex::new(None),
    });

    tokio::spawn(read_server_stdout(stdout, server.clone()));

    // Handshake.
    let init_params = json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": { "name": "lv-mcp-bridge", "version": "0.1.0" }
    });
    let init_result = match timeout(
        Duration::from_secs(req.init_timeout),
        server.call("initialize", Some(init_params)),
    )
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return err(format!("initialize failed: {}", e)),
        Err(_) => return err("initialize timed out".to_string()),
    };

    let protocol = init_result
        .get("protocolVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    if let Err(e) = server.notify("notifications/initialized", None).await {
        return err(format!("initialized notification failed: {}", e));
    }

    let tools_result = match timeout(
        Duration::from_secs(req.init_timeout),
        server.call("tools/list", Some(json!({}))),
    )
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return err(format!("tools/list failed: {}", e)),
        Err(_) => return err("tools/list timed out".to_string()),
    };

    let tools: Vec<ToolInfo> = tools_result
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|t| ToolInfo {
                    name: t
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string(),
                    description: t.get("description").and_then(|d| d.as_str()).map(String::from),
                    parameters: t.get("inputSchema").cloned().or_else(|| t.get("parameters").cloned()),
                })
                .collect()
        })
        .unwrap_or_default();

    *server.tools.lock().await = tools.clone();

    {
        let mut st = state.lock().await;
        st.servers.insert(req.name.clone(), server.clone());
    }

    ok(
        format!(
            "server '{}' ready (protocol {}), {} tools",
            req.name,
            protocol,
            tools.len()
        ),
        Some(json!({
            "name": req.name,
            "tools": tools,
            "protocolVersion": protocol,
        })),
    )
}

async fn read_server_stdout(stdout: tokio::process::ChildStdout, server: Arc<ServerConnection>) {
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                let parsed: Result<JsonRpcResponse, _> = serde_json::from_str(&line);
                match parsed {
                    Ok(resp) => {
                        if let Some(id) = resp.id {
                            let tx = server.pending.lock().await.remove(&id);
                            if let Some(tx) = tx {
                                if let Some(result) = resp.result {
                                    let _ = tx.send(Ok(result));
                                } else if let Some(error) = resp.error {
                                    let _ = tx.send(Err(format!(
                                        "{} (code {})",
                                        error.message, error.code
                                    )));
                                } else {
                                    let _ = tx.send(Err("empty response".to_string()));
                                }
                            }
                        }
                        // Server-to-client requests/notifications are ignored for stdio MCP.
                    }
                    Err(e) => {
                        eprintln!("bridge: invalid JSON from {}: {} | {}", server.name, e, line);
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("bridge: read error from {}: {}", server.name, e);
                break;
            }
        }
    }
    server.set_error("server process closed stdout".to_string()).await;
}

async fn list_tools(state: SharedState, name: String) -> ClientResponse {
    let st = state.lock().await;
    let server = match st.servers.get(&name) {
        Some(s) => s.clone(),
        None => return err(format!("server '{}' not found", name)),
    };
    drop(st);

    let tools = server.tools.lock().await.clone();
    ok(
        format!("{} tools", tools.len()),
        Some(json!({ "tools": tools })),
    )
}

async fn call_tool(state: SharedState, req: CallToolRequest) -> ClientResponse {
    let st = state.lock().await;
    let server = match st.servers.get(&req.name) {
        Some(s) => s.clone(),
        None => return err(format!("server '{}' not found", req.name)),
    };
    drop(st);

    if let Some(error) = server.error.lock().await.as_ref() {
        return err(format!("server '{}' is down: {}", req.name, error));
    }

    let params = json!({
        "name": req.tool,
        "arguments": req.arguments,
    });

    let result = match timeout(
        Duration::from_secs(req.timeout),
        server.call("tools/call", Some(params)),
    )
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return err(format!("tool call failed: {}", e)),
        Err(_) => return err("tool call timed out".to_string()),
    };

    // MCP tool result content -> string.
    let output = if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
        content
            .iter()
            .map(|item| {
                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                    text.to_string()
                } else {
                    item.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("")
    } else {
        result.to_string()
    };

    ok(output, Some(result))
}

async fn remove_server(state: SharedState, name: String) -> ClientResponse {
    let mut st = state.lock().await;
    let server = match st.servers.remove(&name) {
        Some(s) => s,
        None => return err(format!("server '{}' not found", name)),
    };
    drop(st);

    server.set_error("removed by client".to_string()).await;
    // Dropping the child terminates the server process.
    ok(format!("server '{}' removed", name), None)
}

async fn status(state: SharedState) -> ClientResponse {
    let st = state.lock().await;
    let mut servers = Vec::new();
    for (name, server) in &st.servers {
        let tool_count = server.tools.lock().await.len();
        let error = server.error.lock().await.clone();
        servers.push(json!({
            "name": name,
            "tools": tool_count,
            "error": error,
        }));
    }
    ok(
        format!("{} active server(s)", servers.len()),
        Some(json!({ "servers": servers })),
    )
}

// ---------- Client handling ----------

async fn handle_client(mut stream: UnixStream, state: SharedState) {
    let (reader, mut writer) = stream.split();
    let mut lines = BufReader::new(reader).lines();

    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                let cmd: Result<ClientCommand, _> = serde_json::from_str(&line);
                let resp = match cmd {
                    Ok(ClientCommand::AddServer(req)) => add_server(state.clone(), req).await,
                    Ok(ClientCommand::ListTools { name }) => list_tools(state.clone(), name).await,
                    Ok(ClientCommand::CallTool(req)) => call_tool(state.clone(), req).await,
                    Ok(ClientCommand::RemoveServer { name }) => remove_server(state.clone(), name).await,
                    Ok(ClientCommand::Status) => status(state.clone()).await,
                    Err(e) => err(format!("invalid command: {}", e)),
                };

                if write_response(&mut writer, resp).await.is_err() {
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
}

// ---------- Main ----------

#[tokio::main]
async fn main() -> tokio::io::Result<()> {
    let socket = std::env::args()
        .position(|a| a == "--socket")
        .and_then(|i| std::env::args().nth(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET));

    // Remove stale socket.
    if socket.exists() {
        let _ = tokio::fs::remove_file(&socket).await;
    }

    let listener = UnixListener::bind(&socket)?;
    eprintln!("lv-mcp-bridge listening on {}", socket.display());

    let state: SharedState = Arc::new(Mutex::new(BridgeState::new()));

    let accept_loop = {
        let state = state.clone();
        async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        let state = state.clone();
                        tokio::spawn(handle_client(stream, state));
                    }
                    Err(e) => {
                        eprintln!("bridge: accept error: {}", e);
                    }
                }
            }
        }
    };

    let shutdown = async move {
        let _ = signal::ctrl_c().await;
        eprintln!("lv-mcp-bridge shutting down");
        let st = state.lock().await;
        for (_name, server) in &st.servers {
            server.set_error("bridge shutting down".to_string()).await;
        }
    };

    tokio::select! {
        _ = accept_loop => {},
        _ = shutdown => {},
    }

    // Best-effort cleanup.
    let _ = tokio::fs::remove_file(&socket).await;
    Ok(())
}
