//! OpenAI Chat Completions 协议的 mock 上游。
//!
//! - POST /v1/chat/completions 支持非流式与 SSE 流式响应
//! - 相同请求体经由 moka 缓存命中，保证响应确定性；usage 中的数字在首次生成时随机
//! - GET /health 用于健康检查

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use moka::future::Cache;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// 每个响应最大生成 token 数（安全防护，忽略 max_tokens 超过该值的请求）。
const MAX_GEN_TOKENS: u64 = 4096;

/// OpenAI Chat Completions 协议的 mock 上游，支持流式/非流式响应与请求级缓存。
#[derive(Parser, Debug, Clone)]
#[command(name = "mock-stream", version, about)]
struct Cli {
    /// 监听地址（host:port）
    #[arg(short, long, env = "MOCK_STREAM_ADDR", default_value = "0.0.0.0:8080")]
    addr: String,

    /// 缓存的最大请求体条数，超出后按 TinyLFU 淘汰
    #[arg(long, env = "MOCK_STREAM_CACHE_CAPACITY", default_value_t = 50_000)]
    cache_capacity: u64,

    /// 缓存条目的过期时间（秒）
    #[arg(long, env = "MOCK_STREAM_CACHE_TTL_SECS", default_value_t = 1_800)]
    cache_ttl_secs: u64,

    /// 流式响应每个 token 之间的发送间隔（毫秒），0 表示尽快推送
    #[arg(long, env = "MOCK_STREAM_TOKEN_DELAY_MS", default_value_t = 8)]
    token_delay_ms: u64,

    /// 收到 SIGTERM/SIGINT 后等待在途请求完成的最长时间（秒），超时强制退出
    #[arg(long, env = "MOCK_STREAM_DRAIN_TIMEOUT_SECS", default_value_t = 30)]
    drain_timeout_secs: u64,
}

/// moka 的 try_get_with 要求错误实现 Display，本服务不会失败，用空结构体占位。
#[derive(Debug)]
struct Never;

impl std::fmt::Display for Never {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("unreachable")
    }
}

type AppResult<T> = Result<T, AppError>;

/// 协议格式的错误响应（与 OpenAI 错误结构一致）。
struct AppError {
    status: StatusCode,
    message: String,
    err_type: &'static str,
}

impl AppError {
    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            err_type: "invalid_request_error",
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            err_type: "server_error",
        }
    }
}

impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.message,
                "type": self.err_type,
                "param": Value::Null,
                "code": Value::Null,
            }
        });
        (self.status, Json(body)).into_response()
    }
}

#[derive(Clone)]
struct AppState {
    /// key 为请求体原文，value 为该请求确定下来的响应模板（首字节时间敏感字段除外）。
    cache: Cache<String, Arc<ResponseBlueprint>>,
    /// 进程内单调递增的响应 id 计数。
    id_counter: Arc<AtomicU64>,
    /// 流式响应每个 token 之间的发送间隔。
    token_delay: Duration,
    /// 累计 token 消费统计，按 model 分桶。
    stats: Arc<Mutex<Stats>>,
}

/// 进程启动以来的累计统计；仅进程内有效，重启清零。
#[derive(Default)]
struct Stats {
    requests_total: u64,
    stream_requests: u64,
    cache_hits: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    /// 按 model 分桶的计数。
    models: BTreeMap<String, ModelStats>,
}

#[derive(Clone, Default, Serialize)]
struct ModelStats {
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

impl Stats {
    fn record(&mut self, bp: &ResponseBlueprint, stream_mode: bool, cache_hit: bool) {
        self.requests_total += 1;
        self.stream_requests += u64::from(stream_mode);
        self.cache_hits += u64::from(cache_hit);
        self.prompt_tokens += bp.usage_prompt;
        self.completion_tokens += bp.usage_completion;
        self.total_tokens += bp.usage_total;
        let m = self.models.entry(bp.model.clone()).or_default();
        m.requests += 1;
        m.prompt_tokens += bp.usage_prompt;
        m.completion_tokens += bp.usage_completion;
        m.total_tokens += bp.usage_total;
    }

    fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            requests_total: self.requests_total,
            stream_requests: self.stream_requests,
            non_stream_requests: self.requests_total - self.stream_requests,
            cache_hits: self.cache_hits,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            models: self
                .models
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }
}

#[derive(Serialize)]
struct StatsSnapshot {
    requests_total: u64,
    stream_requests: u64,
    non_stream_requests: u64,
    cache_hits: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    models: BTreeMap<String, ModelStats>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Option<Vec<Value>>,
    stream: Option<bool>,
    /// OpenAI 新字段，completion tokens 上限。
    max_completion_tokens: Option<u64>,
    /// 旧字段，与 max_completion_tokens 取其一。
    max_tokens: Option<u64>,
    /// 客户端若声明 include_usage: false，则流式响应中不发送 usage 帧。
    stream_options: Option<StreamOptions>,
    // 其余协议字段（temperature 等）由 serde 默认忽略。
}

#[derive(Debug, Deserialize)]
struct StreamOptions {
    include_usage: Option<bool>,
}

#[derive(Clone)]
struct ResponseBlueprint {
    model: String,
    fingerprint: String,
    /// 逐 token 的文本，个数等于 usage_completion；流式按此切帧，非流式直接拼接。
    tokens: Vec<String>,
    finish_reason: &'static str,
    usage_prompt: u64,
    usage_completion: u64,
    usage_total: u64,
}

struct Chunk {
    delta_role: bool,
    delta_content: Option<String>,
    finish_reason: Option<&'static str>,
    usage: Option<Value>,
}

impl Chunk {
    /// 协议规定 usage 帧的 choices 必须为空数组，而非含 delta 的单元素数组。
    fn is_usage_frame(&self) -> bool {
        self.usage.is_some()
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mock_stream=info,tower_http=info".into()),
        )
        .init();

    let state = AppState {
        cache: Cache::builder()
            .max_capacity(cli.cache_capacity)
            .time_to_live(Duration::from_secs(cli.cache_ttl_secs))
            .build(),
        id_counter: Arc::new(AtomicU64::new(1)),
        token_delay: Duration::from_millis(cli.token_delay_ms),
        stats: Arc::new(Mutex::new(Stats::default())),
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/health", get(health))
        .route("/mock/stats", get(get_stats))
        .route("/mock/stats/reset", post(reset_stats))
        .with_state(state);

    let addr = cli.addr;
    let drain_timeout = Duration::from_secs(cli.drain_timeout_secs);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%e, %addr, "failed to bind");
            std::process::exit(1);
        }
    };
    tracing::info!(%addr, ?drain_timeout, "mock chat completions upstream listening");

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(drain_timeout))
        .await
    {
        tracing::error!(%e, "server exited with error");
        std::process::exit(1);
    }
    tracing::info!("shutdown complete");
}

/// 监听 SIGINT 与 SIGTERM；信号到达后停止接收新连接，等待在途请求完成，
/// 超过 drain_timeout 仍未排空则记录日志并强制退出进程。
async fn shutdown_signal(drain_timeout: Duration) {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(%e, "failed to install SIGTERM handler, only SIGINT is handled");
            tokio::signal::ctrl_c().await.ok();
            return;
        }
    };

    tokio::select! {
        r = tokio::signal::ctrl_c() => {
            if let Err(e) = r {
                tracing::warn!(%e, "failed to listen for SIGINT");
            }
            tracing::info!("received SIGINT, draining in-flight requests");
        }
        _ = sigterm.recv() => {
            tracing::info!("received SIGTERM, draining in-flight requests");
        }
    }

    // 优雅关停本身没有超时：超时的兜底由这里负责，避免挂住的 SSE 连接让进程永远不退出。
    tokio::spawn(async move {
        tokio::time::sleep(drain_timeout).await;
        tracing::error!(
            timeout_secs = drain_timeout.as_secs(),
            "drain timeout exceeded, forcing exit"
        );
        std::process::exit(1);
    });
}

async fn health() -> &'static str {
    "ok"
}

/// GET /mock/stats —— 返回进程启动以来累计的 token 消费统计（按 model 分桶）。
async fn get_stats(State(state): State<AppState>) -> Response {
    // entry_count 是异步维护的，run_pending_tasks 后才是精确值。
    state.cache.run_pending_tasks().await;
    let cache_entries = state.cache.entry_count();
    match state.stats.lock() {
        Ok(s) => {
            let mut body = serde_json::to_value(s.snapshot())
                .unwrap_or_else(|_| json!({"error": "serialize failed"}));
            body["cache_entries"] = Value::from(cache_entries);
            Json(body).into_response()
        }
        Err(_) => AppError::internal("stats lock poisoned").into_response(),
    }
}

/// POST /mock/stats/reset —— 清零统计，返回清空后的快照。
async fn reset_stats(State(state): State<AppState>) -> Response {
    match state.stats.lock() {
        Ok(mut s) => {
            *s = Stats::default();
            Json(json!({"reset": true})).into_response()
        }
        Err(_) => AppError::internal("stats lock poisoned").into_response(),
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> AppResult<Response> {
    let body: Cow<'_, str> = match String::from_utf8(body.to_vec()) {
        Ok(s) => Cow::Owned(s),
        Err(_) => {
            return Err(AppError::invalid_request(
                "request body must be valid UTF-8 JSON",
            ));
        }
    };

    let req: ChatCompletionRequest = serde_json::from_str(&body)
        .map_err(|e| AppError::invalid_request(format!("invalid JSON body: {e}")))?;

    let stream_mode = req.stream.unwrap_or(false);
    // 协议约定：仅当 stream_options.include_usage == true 时流式响应才附带 usage 帧。
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);
    let request_id = next_id(&state);

    let cache_hit = state.cache.contains_key(&*body);
    let blueprint = state
        .cache
        .try_get_with(body.into_owned(), {
            let state = state.clone();
            async move { build_blueprint(&state, &req).map_err(Arc::new) }
        })
        .await
        .map_err(|e| {
            // try_get_with 的错误为 Arc<Never>，仅可能是 init future 返回的 Never（不会发生）。
            AppError::internal(format!("cache init failed: {e}"))
        })?;

    if let Ok(mut stats) = state.stats.lock() {
        stats.record(&blueprint, stream_mode, cache_hit);
    }

    let created = unix_now();
    if stream_mode {
        Ok(stream_response(
            &state,
            blueprint,
            &request_id,
            created,
            include_usage,
        ))
    } else {
        Ok(unary_response(blueprint, &request_id, created))
    }
}

fn unary_response(bp: Arc<ResponseBlueprint>, request_id: &str, created: u64) -> Response {
    let body = json!({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": bp.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": bp.tokens.concat(),
            },
            "logprobs": Value::Null,
            "finish_reason": bp.finish_reason,
        }],
        "usage": usage_json(&bp),
        "system_fingerprint": bp.fingerprint,
    });
    (StatusCode::OK, Json(body)).into_response()
}

fn stream_response(
    state: &AppState,
    bp: Arc<ResponseBlueprint>,
    request_id: &str,
    created: u64,
    include_usage: bool,
) -> Response {
    let id = request_id.to_string();
    let token_delay = state.token_delay;

    let stream = stream! {
        let chunks = plan_chunks(&bp, include_usage);
        for (idx, chunk) in chunks.into_iter().enumerate() {
            if token_delay > Duration::ZERO && idx > 0 {
                tokio::time::sleep(token_delay).await;
            }
            let payload = chunk_payload(&id, created, &bp, chunk, include_usage);
            yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).into_response()
}

/// 把完整内容切成若干流式 chunk：role 首帧 + 内容帧 + finish 帧（+ 可选 usage 帧）。
fn plan_chunks(bp: &ResponseBlueprint, include_usage: bool) -> Vec<Chunk> {
    let mut chunks = Vec::with_capacity(bp.tokens.len() + 3);

    chunks.push(Chunk {
        delta_role: true,
        delta_content: None,
        finish_reason: None,
        usage: None,
    });
    for tok in &bp.tokens {
        chunks.push(Chunk {
            delta_role: false,
            delta_content: Some(tok.clone()),
            finish_reason: None,
            usage: None,
        });
    }
    chunks.push(Chunk {
        delta_role: false,
        delta_content: None,
        finish_reason: Some(bp.finish_reason),
        usage: None,
    });
    if include_usage {
        chunks.push(Chunk {
            delta_role: false,
            delta_content: None,
            finish_reason: None,
            usage: Some(usage_json(bp)),
        });
    }
    chunks
}

fn chunk_payload(
    id: &str,
    created: u64,
    bp: &ResponseBlueprint,
    chunk: Chunk,
    include_usage: bool,
) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("id".into(), Value::from(id));
    obj.insert("object".into(), Value::from("chat.completion.chunk"));
    obj.insert("created".into(), Value::from(created));
    obj.insert("model".into(), Value::from(bp.model.clone()));

    if chunk.is_usage_frame() {
        obj.insert("choices".into(), Value::Array(vec![]));
        if let Some(usage) = chunk.usage {
            obj.insert("usage".into(), usage);
        }
        return Value::Object(obj);
    }

    let mut delta = serde_json::Map::new();
    if chunk.delta_role {
        delta.insert("role".into(), Value::from("assistant"));
    }
    if let Some(content) = chunk.delta_content {
        delta.insert("content".into(), Value::from(content));
    }

    let mut choice = serde_json::Map::new();
    choice.insert("index".into(), Value::from(0));
    choice.insert("delta".into(), Value::Object(delta));
    choice.insert("logprobs".into(), Value::Null);
    match chunk.finish_reason {
        Some(r) => {
            choice.insert("finish_reason".into(), Value::from(r));
        }
        None => {
            choice.insert("finish_reason".into(), Value::Null);
        }
    }

    obj.insert("choices".into(), Value::Array(vec![Value::Object(choice)]));
    obj.insert(
        "system_fingerprint".into(),
        Value::from(bp.fingerprint.clone()),
    );
    // 协议约定：开启 include_usage 时，除最后的 usage 帧外其余帧也带 usage 字段，值为 null。
    if include_usage {
        obj.insert("usage".into(), Value::Null);
    }
    Value::Object(obj)
}

fn usage_json(bp: &ResponseBlueprint) -> Value {
    json!({
        "prompt_tokens": bp.usage_prompt,
        "completion_tokens": bp.usage_completion,
        "total_tokens": bp.usage_total,
    })
}

fn build_blueprint(state: &AppState, req: &ChatCompletionRequest) -> Result<Arc<ResponseBlueprint>, Never> {
    let model = req
        .model
        .clone()
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "mock-stream-1".into());

    let prompt_chars: usize = req
        .messages
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|m| m.to_string().len())
        .sum();

    let mut rng = rand::rng();
    // prompt_tokens：消息字符数粗算 + 随机抖动，保证各请求之间 usage 有区分度。
    let usage_prompt = (prompt_chars as u64 / 4) + rng.random_range(1..=64);

    // 指定了上限时恰好生成上限个 token（超过 MAX_GEN_TOKENS 时按该值截断），
    // 生成因触达上限而停止，按协议 finish_reason 为 length；未指定时随机长度自然结束。
    let requested = req
        .max_completion_tokens
        .or(req.max_tokens)
        .filter(|&n| n > 0)
        .map(|n| n.min(MAX_GEN_TOKENS));
    let (usage_completion, finish_reason) = match requested {
        Some(n) => (n, "length"),
        None => (rng.random_range(32..=512), "stop"),
    };

    let fingerprint = fingerprint_for(state);
    let tokens = build_tokens(&model, usage_completion, &fingerprint, &mut rng);

    Ok(Arc::new(ResponseBlueprint {
        model,
        fingerprint,
        tokens,
        finish_reason,
        usage_prompt,
        usage_completion,
        usage_total: usage_prompt + usage_completion,
    }))
}

/// 生成正好 completion_tokens 个 token：前缀 `[mock model fp]` 整体算第一个，
/// 其余为字母数字混排的随机词，自带前导空格，拼接即为完整内容。
fn build_tokens(
    model: &str,
    completion_tokens: u64,
    fingerprint: &str,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec<String> {
    let mut tokens = Vec::with_capacity(completion_tokens as usize);
    tokens.push(format!("[mock {model} {fingerprint}]"));
    for _ in 1..completion_tokens {
        let len = rng.random_range(2..=8usize);
        let mut tok = String::with_capacity(len + 1);
        tok.push(' ');
        for _ in 0..len {
            let c = rng.random_range(0..36u8);
            tok.push(match c {
                0..=25 => (b'a' + c) as char,
                _ => (b'0' + (c - 26)) as char,
            });
        }
        tokens.push(tok);
    }
    tokens
}

fn fingerprint_for(state: &AppState) -> String {
    let n = state.id_counter.fetch_add(1, Ordering::Relaxed);
    let mut h = DefaultHasher::new();
    n.hash(&mut h);
    unix_now().hash(&mut h);
    format!("fp_{:016x}", h.finish())
}

fn next_id(state: &AppState) -> String {
    let n = state.id_counter.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-mock-{n:016x}")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
