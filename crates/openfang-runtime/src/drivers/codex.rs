//! OpenAI Codex (ChatGPT subscription) driver.
//!
//! Uses the Codex Responses API at `chatgpt.com/backend-api/codex/responses`
//! which is completely different from the standard OpenAI chat/completions API.
//! Codex OAuth tokens (from ChatGPT Plus/Team/Pro subscriptions) only work with
//! this endpoint, not `api.openai.com/v1`.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use async_trait::async_trait;
use futures::StreamExt;
use openfang_types::message::{ContentBlock, MessageContent, Role, StopReason, TokenUsage};
use openfang_types::tool::ToolCall;
use serde::Serialize;
use std::sync::Mutex;
use tracing::{debug, warn};
use zeroize::Zeroizing;

/// Default base URL for the Codex backend API.
const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";

/// OAuth token endpoint for refreshing tokens.
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// OAuth client ID (same as Codex CLI).
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// JWT claim path for extracting account ID.
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

/// Refresh buffer — refresh token this many seconds before expiry.
const REFRESH_BUFFER_SECS: u64 = 300;

/// Codex Responses API driver.
pub struct CodexDriver {
    /// Current access token (refreshable).
    access_token: Mutex<Zeroizing<String>>,
    /// Refresh token for obtaining new access tokens.
    refresh_token: Mutex<Zeroizing<String>>,
    /// ChatGPT account ID (extracted from JWT or auth.json).
    account_id: String,
    /// Token expiry (Unix timestamp in seconds).
    token_expires_at: Mutex<u64>,
    /// Base URL (default: https://chatgpt.com/backend-api).
    base_url: String,
    client: reqwest::Client,
}

impl CodexDriver {
    /// Create a new Codex driver from auth credentials.
    pub fn new(
        access_token: String,
        refresh_token: String,
        account_id: String,
        token_expires_at: u64,
        base_url: Option<String>,
    ) -> Self {
        Self {
            access_token: Mutex::new(Zeroizing::new(access_token)),
            refresh_token: Mutex::new(Zeroizing::new(refresh_token)),
            account_id,
            token_expires_at: Mutex::new(token_expires_at),
            base_url: base_url.unwrap_or_else(|| DEFAULT_CODEX_BASE_URL.to_string()),
            client: reqwest::Client::builder()
                .user_agent(crate::USER_AGENT)
                .build()
                .unwrap_or_default(),
        }
    }

    /// Create from `~/.codex/auth.json`.
    pub fn from_codex_auth() -> Result<Self, LlmError> {
        let auth = read_codex_auth_json().ok_or_else(|| {
            LlmError::MissingApiKey(
                "No Codex CLI credentials found. Run `codex login` first.".to_string(),
            )
        })?;
        Ok(Self::new(
            auth.access_token,
            auth.refresh_token,
            auth.account_id,
            auth.expires_at,
            None,
        ))
    }

    /// Get a valid access token, refreshing if needed.
    async fn get_token(&self) -> Result<String, LlmError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let expires_at = *self.token_expires_at.lock().unwrap_or_else(|e| e.into_inner());

        if now + REFRESH_BUFFER_SECS < expires_at {
            let token = self.access_token.lock().unwrap_or_else(|e| e.into_inner());
            return Ok(token.as_str().to_string());
        }

        debug!("Codex access token expired or expiring soon, refreshing");
        self.refresh().await
    }

    /// Refresh the access token using the refresh token.
    async fn refresh(&self) -> Result<String, LlmError> {
        let refresh_tok = {
            let lock = self.refresh_token.lock().unwrap_or_else(|e| e.into_inner());
            lock.as_str().to_string()
        };

        let form_body = format!(
            "grant_type=refresh_token&refresh_token={}&client_id={}",
            urlencod(&refresh_tok),
            CLIENT_ID
        );

        let resp = self
            .client
            .post(TOKEN_URL)
            .header("content-type", "application/x-www-form-urlencoded")
            .body(form_body)
            .send()
            .await
            .map_err(|e| LlmError::Http(format!("Token refresh failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::AuthenticationFailed(format!(
                "Token refresh returned {status}: {body}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Parse(format!("Failed to parse refresh response: {e}")))?;

        let new_access = json["access_token"]
            .as_str()
            .ok_or_else(|| LlmError::Parse("No access_token in refresh response".to_string()))?
            .to_string();
        let new_refresh = json["refresh_token"]
            .as_str()
            .ok_or_else(|| LlmError::Parse("No refresh_token in refresh response".to_string()))?
            .to_string();
        let expires_in = json["expires_in"].as_u64().unwrap_or(864000);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let new_expires_at = now + expires_in;

        // Update stored tokens
        {
            let mut lock = self.access_token.lock().unwrap_or_else(|e| e.into_inner());
            *lock = Zeroizing::new(new_access.clone());
        }
        {
            let mut lock = self.refresh_token.lock().unwrap_or_else(|e| e.into_inner());
            *lock = Zeroizing::new(new_refresh);
        }
        {
            let mut lock = self.token_expires_at.lock().unwrap_or_else(|e| e.into_inner());
            *lock = new_expires_at;
        }

        debug!(expires_in, "Codex token refreshed successfully");
        Ok(new_access)
    }

    /// Build the Codex Responses API URL.
    fn responses_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/codex/responses") {
            base.to_string()
        } else if base.ends_with("/codex") {
            format!("{base}/responses")
        } else {
            format!("{base}/codex/responses")
        }
    }
}

// -- Responses API request types --

#[derive(Debug, Serialize)]
struct CodexRequest {
    model: String,
    store: bool,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    input: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    include: Vec<String>,
}

#[async_trait]
impl LlmDriver for CodexDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let token = self.get_token().await?;
        let url = self.responses_url();
        // Codex API requires stream=true; we collect the SSE events into a full response
        let body = build_codex_request(&request, true);

        let max_retries = 3;
        for attempt in 0..=max_retries {
            debug!(url = %url, attempt, model = %request.model, "Sending Codex Responses API request (streamed-collect)");

            let resp = self
                .client
                .post(&url)
                .header("authorization", format!("Bearer {token}"))
                .header("chatgpt-account-id", &self.account_id)
                .header("openai-beta", "responses=experimental")
                .header("originator", "openfang")
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .json(&body)
                .send()
                .await
                .map_err(|e| LlmError::Http(e.to_string()))?;

            let status = resp.status().as_u16();
            if status == 429 {
                if attempt < max_retries {
                    let retry_ms = (attempt + 1) as u64 * 2000;
                    warn!(status, retry_ms, "Codex rate limited, retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(retry_ms)).await;
                    continue;
                }
                return Err(LlmError::RateLimited { retry_after_ms: 5000 });
            }

            if status == 401 || status == 403 {
                if attempt == 0 {
                    warn!("Codex auth failed, attempting token refresh");
                    let _ = self.refresh().await;
                    continue;
                }
                let body = resp.text().await.unwrap_or_default();
                return Err(LlmError::AuthenticationFailed(body));
            }

            if !resp.status().is_success() {
                let body = resp.text().await.unwrap_or_default();
                if attempt < max_retries && (status >= 500 || status == 502 || status == 503) {
                    let retry_ms = (attempt + 1) as u64 * 1500;
                    warn!(status, retry_ms, "Codex server error, retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(retry_ms)).await;
                    continue;
                }
                return Err(LlmError::Api { status, message: body });
            }

            // Collect SSE stream into full response
            let sse_body = resp.text().await.map_err(|e| LlmError::Http(e.to_string()))?;
            // Find the response.completed event which has the full response JSON
            for line in sse_body.lines() {
                let data = match line.strip_prefix("data:") {
                    Some(d) => d.trim(),
                    None => match line.strip_prefix("data: ") {
                        Some(d) => d.trim(),
                        None => continue,
                    },
                };
                if data == "[DONE]" {
                    break;
                }
                if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                    if event.get("type").and_then(|t| t.as_str()) == Some("response.completed") {
                        if let Some(resp_obj) = event.get("response") {
                            return parse_codex_response(resp_obj);
                        }
                    }
                }
            }

            return Err(LlmError::Parse("No response.completed event in Codex SSE stream".to_string()));
        }

        Err(LlmError::Api {
            status: 0,
            message: "Max retries exceeded".to_string(),
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let token = self.get_token().await?;
        let url = self.responses_url();
        let body = build_codex_request(&request, true);

        let max_retries = 3;
        for attempt in 0..=max_retries {
            debug!(url = %url, attempt, model = %request.model, "Sending Codex streaming request");

            let resp = self
                .client
                .post(&url)
                .header("authorization", format!("Bearer {token}"))
                .header("chatgpt-account-id", &self.account_id)
                .header("openai-beta", "responses=experimental")
                .header("originator", "openfang")
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .json(&body)
                .send()
                .await
                .map_err(|e| LlmError::Http(e.to_string()))?;

            let status = resp.status().as_u16();
            if status == 429 {
                if attempt < max_retries {
                    let retry_ms = (attempt + 1) as u64 * 2000;
                    warn!(status, retry_ms, "Codex rate limited (stream), retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(retry_ms)).await;
                    continue;
                }
                return Err(LlmError::RateLimited { retry_after_ms: 5000 });
            }

            if status == 401 || status == 403 {
                if attempt == 0 {
                    warn!("Codex auth failed (stream), attempting token refresh");
                    let _ = self.refresh().await;
                    continue;
                }
                let body = resp.text().await.unwrap_or_default();
                return Err(LlmError::AuthenticationFailed(body));
            }

            if !resp.status().is_success() {
                let err_body = resp.text().await.unwrap_or_default();
                if attempt < max_retries && status >= 500 {
                    let retry_ms = (attempt + 1) as u64 * 1500;
                    warn!(status, retry_ms, "Codex server error (stream), retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(retry_ms)).await;
                    continue;
                }
                return Err(LlmError::Api { status, message: err_body });
            }

            // Parse SSE stream
            let mut buffer = String::new();
            let mut text_content = String::new();
            let mut thinking_content = String::new();
            // Track current items in the streaming response
            let mut _current_item_type: Option<String> = None;
            let mut tool_accum: Vec<(String, String, String, String)> = Vec::new(); // (call_id, item_id, name, arguments)
            let mut usage = TokenUsage::default();
            let mut response_status: Option<String> = None;

            let mut byte_stream = resp.bytes_stream();
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = chunk_result.map_err(|e| LlmError::Http(e.to_string()))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim_end().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    let data = match line.strip_prefix("data:") {
                        Some(d) => d.trim_start(),
                        None => continue,
                    };

                    if data == "[DONE]" {
                        continue;
                    }

                    let event: serde_json::Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let event_type = event["type"].as_str().unwrap_or("");

                    match event_type {
                        "response.output_item.added" => {
                            let item = &event["item"];
                            let item_type = item["type"].as_str().unwrap_or("");
                            _current_item_type = Some(item_type.to_string()); // track for future use

                            if item_type == "function_call" {
                                let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                                let item_id = item["id"].as_str().unwrap_or("").to_string();
                                let name = item["name"].as_str().unwrap_or("").to_string();
                                tool_accum.push((call_id.clone(), item_id, name.clone(), String::new()));
                                let _ = tx
                                    .send(StreamEvent::ToolUseStart {
                                        id: call_id,
                                        name,
                                    })
                                    .await;
                            }
                        }

                        "response.output_text.delta" => {
                            if let Some(delta) = event["delta"].as_str() {
                                if !delta.is_empty() {
                                    text_content.push_str(delta);
                                    let _ = tx
                                        .send(StreamEvent::TextDelta {
                                            text: delta.to_string(),
                                        })
                                        .await;
                                }
                            }
                        }

                        "response.reasoning_summary_text.delta" => {
                            if let Some(delta) = event["delta"].as_str() {
                                if !delta.is_empty() {
                                    thinking_content.push_str(delta);
                                    let _ = tx
                                        .send(StreamEvent::ThinkingDelta {
                                            text: delta.to_string(),
                                        })
                                        .await;
                                }
                            }
                        }

                        "response.function_call_arguments.delta" => {
                            if let Some(delta) = event["delta"].as_str() {
                                if let Some(tool) = tool_accum.last_mut() {
                                    tool.3.push_str(delta);
                                    let _ = tx
                                        .send(StreamEvent::ToolInputDelta {
                                            text: delta.to_string(),
                                        })
                                        .await;
                                }
                            }
                        }

                        "response.output_item.done" => {
                            let item = &event["item"];
                            let item_type = item["type"].as_str().unwrap_or("");

                            if item_type == "function_call" {
                                if let Some(tool) = tool_accum.last() {
                                    let _ = tx
                                        .send(StreamEvent::ToolUseEnd {
                                            id: tool.0.clone(),
                                            name: tool.2.clone(),
                                            input: serde_json::from_str(&tool.3)
                                                .unwrap_or_default(),
                                        })
                                        .await;
                                }
                            }
                            _current_item_type = None; // reset for next item
                        }

                        "response.completed" | "response.done" => {
                            if let Some(resp) = event.get("response") {
                                response_status = resp["status"].as_str().map(|s| s.to_string());

                                if let Some(u) = resp.get("usage") {
                                    let cached = u["input_tokens_details"]["cached_tokens"]
                                        .as_u64()
                                        .unwrap_or(0);
                                    usage.input_tokens =
                                        u["input_tokens"].as_u64().unwrap_or(0) - cached;
                                    usage.output_tokens =
                                        u["output_tokens"].as_u64().unwrap_or(0);
                                }
                            }
                        }

                        "response.failed" => {
                            let msg = event
                                .pointer("/response/error/message")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Codex response failed");
                            return Err(LlmError::Api {
                                status: 0,
                                message: msg.to_string(),
                            });
                        }

                        "error" => {
                            let msg = event["message"]
                                .as_str()
                                .unwrap_or("Unknown Codex error");
                            return Err(LlmError::Api {
                                status: 0,
                                message: msg.to_string(),
                            });
                        }

                        _ => {}
                    }
                }
            }

            // Build final response
            let mut content = Vec::new();
            let mut tool_calls = Vec::new();

            if !thinking_content.is_empty() {
                content.push(ContentBlock::Thinking {
                    thinking: thinking_content,
                });
            }

            if !text_content.is_empty() {
                content.push(ContentBlock::Text {
                    text: text_content,
                    provider_metadata: None,
                });
            }

            for (call_id, _item_id, name, arguments) in &tool_accum {
                let input: serde_json::Value =
                    serde_json::from_str(arguments).unwrap_or_default();
                content.push(ContentBlock::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    provider_metadata: None,
                });
                tool_calls.push(ToolCall {
                    id: call_id.clone(),
                    name: name.clone(),
                    input,
                });
            }

            let stop_reason = match response_status.as_deref() {
                Some("completed") => {
                    if !tool_calls.is_empty() {
                        StopReason::ToolUse
                    } else {
                        StopReason::EndTurn
                    }
                }
                Some("incomplete") => StopReason::MaxTokens,
                _ => {
                    if !tool_calls.is_empty() {
                        StopReason::ToolUse
                    } else {
                        StopReason::EndTurn
                    }
                }
            };

            if !content.is_empty() && usage.input_tokens == 0 && usage.output_tokens == 0 {
                usage.output_tokens = 1;
            }

            let _ = tx
                .send(StreamEvent::ContentComplete { stop_reason, usage })
                .await;

            return Ok(CompletionResponse {
                content,
                stop_reason,
                tool_calls,
                usage,
            });
        }

        Err(LlmError::Api {
            status: 0,
            message: "Max retries exceeded".to_string(),
        })
    }
}

// -- Request building --

fn build_codex_request(request: &CompletionRequest, stream: bool) -> CodexRequest {
    let mut input: Vec<serde_json::Value> = Vec::new();

    // Convert messages to Responses API format
    for msg in &request.messages {
        match (&msg.role, &msg.content) {
            (Role::System, _) => {
                // System messages go into `instructions`, not `input`
            }
            (Role::User, MessageContent::Text(text)) => {
                input.push(serde_json::json!({
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}]
                }));
            }
            (Role::User, MessageContent::Blocks(blocks)) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text, .. } => {
                            input.push(serde_json::json!({
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}]
                            }));
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            input.push(serde_json::json!({
                                "type": "function_call_output",
                                "call_id": tool_use_id,
                                "output": if content.is_empty() { "(empty)" } else { content }
                            }));
                        }
                        ContentBlock::Image { media_type, data } => {
                            input.push(serde_json::json!({
                                "role": "user",
                                "content": [{
                                    "type": "input_image",
                                    "detail": "auto",
                                    "image_url": format!("data:{media_type};base64,{data}")
                                }]
                            }));
                        }
                        _ => {}
                    }
                }
            }
            (Role::Assistant, MessageContent::Text(text)) => {
                input.push(serde_json::json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                    "status": "completed"
                }));
            }
            (Role::Assistant, MessageContent::Blocks(blocks)) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text, .. } => {
                            input.push(serde_json::json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text, "annotations": []}],
                                "status": "completed"
                            }));
                        }
                        ContentBlock::ToolUse { id, name, input: tool_input, .. } => {
                            input.push(serde_json::json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": serde_json::to_string(tool_input).unwrap_or_default()
                            }));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let tools: Vec<serde_json::Value> = request
        .tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": openfang_types::tool::normalize_schema_for_provider(
                    &t.input_schema, "openai"
                ),
                "strict": false
            })
        })
        .collect();

    let tool_choice = if tools.is_empty() {
        None
    } else {
        Some("auto".to_string())
    };

    CodexRequest {
        model: request.model.clone(),
        store: false,
        stream,
        instructions: request.system.clone(),
        input,
        tools,
        tool_choice,
        include: vec!["reasoning.encrypted_content".to_string()],
    }
}

// -- Response parsing (non-streaming) --

fn parse_codex_response(json: &serde_json::Value) -> Result<CompletionResponse, LlmError> {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();

    if let Some(output) = json["output"].as_array() {
        for item in output {
            let item_type = item["type"].as_str().unwrap_or("");
            match item_type {
                "message" => {
                    if let Some(parts) = item["content"].as_array() {
                        for part in parts {
                            if let Some(text) = part["text"].as_str() {
                                if !text.is_empty() {
                                    content.push(ContentBlock::Text {
                                        text: text.to_string(),
                                        provider_metadata: None,
                                    });
                                }
                            }
                        }
                    }
                }
                "reasoning" => {
                    if let Some(summary) = item["summary"].as_array() {
                        let thinking: String = summary
                            .iter()
                            .filter_map(|s| s["text"].as_str())
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        if !thinking.is_empty() {
                            content.push(ContentBlock::Thinking { thinking });
                        }
                    }
                }
                "function_call" => {
                    let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                    let name = item["name"].as_str().unwrap_or("").to_string();
                    let args_str = item["arguments"].as_str().unwrap_or("{}");
                    let input: serde_json::Value =
                        serde_json::from_str(args_str).unwrap_or_default();

                    content.push(ContentBlock::ToolUse {
                        id: call_id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                        provider_metadata: None,
                    });
                    tool_calls.push(ToolCall {
                        id: call_id,
                        name,
                        input,
                    });
                }
                _ => {}
            }
        }
    }

    let response_status = json["status"].as_str().unwrap_or("completed");
    let stop_reason = match response_status {
        "completed" => {
            if !tool_calls.is_empty() {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }
        }
        "incomplete" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    };

    let mut usage = TokenUsage::default();
    if let Some(u) = json.get("usage") {
        let cached = u["input_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0);
        usage.input_tokens = u["input_tokens"].as_u64().unwrap_or(0) - cached;
        usage.output_tokens = u["output_tokens"].as_u64().unwrap_or(0);
    }

    if !content.is_empty() && usage.input_tokens == 0 && usage.output_tokens == 0 {
        usage.output_tokens = 1;
    }

    Ok(CompletionResponse {
        content,
        stop_reason,
        tool_calls,
        usage,
    })
}

// -- Auth helpers --

/// Parsed Codex CLI auth credentials.
pub struct CodexAuth {
    pub access_token: String,
    pub refresh_token: String,
    pub account_id: String,
    pub expires_at: u64,
}

/// Read credentials from `~/.codex/auth.json`.
pub fn read_codex_auth_json() -> Option<CodexAuth> {
    let codex_home = std::env::var("CODEX_HOME")
        .map(std::path::PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| std::path::PathBuf::from(h).join(".codex"))
        })?;

    let auth_path = codex_home.join("auth.json");
    let content = std::fs::read_to_string(&auth_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Codex CLI stores tokens nested under "tokens"
    let tokens = parsed.get("tokens")?;
    let access_token = tokens.get("access_token")?.as_str()?.to_string();
    let refresh_token = tokens.get("refresh_token")?.as_str()?.to_string();

    // Account ID: from auth.json tokens, or extract from JWT
    let account_id = tokens
        .get("account_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| extract_account_id_from_jwt(&access_token))
        .unwrap_or_default();

    if account_id.is_empty() {
        return None;
    }

    // Compute expiry from JWT exp claim
    let expires_at = extract_jwt_exp(&access_token).unwrap_or(0);

    Some(CodexAuth {
        access_token,
        refresh_token,
        account_id,
        expires_at,
    })
}

/// Extract account ID from a JWT access token.
fn extract_account_id_from_jwt(token: &str) -> Option<String> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let payload = base64_decode_jwt(parts[1])?;
    let json: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    json.pointer(&format!("/{}/chatgpt_account_id", JWT_CLAIM_PATH.replace('.', "/")))
        .or_else(|| json.get(JWT_CLAIM_PATH).and_then(|v| v.get("chatgpt_account_id")))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extract expiry from a JWT access token.
fn extract_jwt_exp(token: &str) -> Option<u64> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let payload = base64_decode_jwt(parts[1])?;
    let json: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    json["exp"].as_u64()
}

/// Decode base64url JWT payload.
fn base64_decode_jwt(input: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    // JWT uses base64url encoding without padding
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    engine.decode(input).ok().or_else(|| {
        // Try with standard padding
        let padded = match input.len() % 4 {
            2 => format!("{input}=="),
            3 => format!("{input}="),
            _ => input.to_string(),
        };
        base64::engine::general_purpose::URL_SAFE.decode(&padded).ok()
    })
}

/// Percent-encode a string for application/x-www-form-urlencoded.
fn urlencod(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(b as char);
            }
            b' ' => result.push('+'),
            _ => {
                result.push('%');
                result.push(char::from(HEX[(b >> 4) as usize]));
                result.push(char::from(HEX[(b & 0xf) as usize]));
            }
        }
    }
    result
}

const HEX: [u8; 16] = *b"0123456789ABCDEF";
