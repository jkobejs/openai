#![deny(clippy::all)]
#![deny(clippy::self_named_module_files)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
use async_openai::{
    self,
    error::OpenAIError,
    types::{
        ChatCompletionRequestMessageArgs, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, Role,
    },
};
use async_trait::async_trait;
use std::{env, error::Error, fmt::Display, time::Duration};
use tiktoken_rs::async_openai::get_chat_completion_max_tokens;

#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait Client: Send + Sync {
    /// Asks `OpenAI's` chat API a question and returns the response as a string
    ///
    /// # Arguments
    ///
    /// * `system` - A string representing the system being used
    /// * `question` - A string representing the question being asked
    /// * `model` - A string representing the model being used
    /// * `temperature` - A float representing the temperature of the response
    ///
    /// # Returns
    ///
    /// A `String` representing the response from `OpenAI's` chat API or an `Error` if the request failed
    ///
    /// # Errors
    ///
    /// Returns an error if the request failed.
    async fn ask_question(
        &self,
        system: &str,
        question: &str,
        model: &str,
        temperature: f32,
    ) -> Result<String, ChatError>;
}

#[derive(Debug)]
pub struct ChatError {
    pub model: String,
    pub kind: ChatErrorKind,
}

#[derive(Debug)]
pub enum ChatErrorKind {
    Request(OpenAIError),
    BuildRequest(String),
}

impl Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ChatErrorKind::BuildRequest(err) => write!(f, "error building chat request {err}"),
            ChatErrorKind::Request(_) => write!(f, "error asking chat model {}", self.model),
        }
    }
}

impl Error for ChatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            ChatErrorKind::Request(err) => Some(err),
            ChatErrorKind::BuildRequest(_) => None,
        }
    }
}

struct TimeoutClient {
    client: async_openai::Client,
}

/// Creates a new `OpenAI` client with timeout functionality
/// # Arguments
/// * `timeout` - A `Duration` representing the timeout for the client
/// # Returns
/// A `Client` implementation with timeout functionality
/// # Errors
/// Returns an error if the client could not be created.
pub fn live(timeout: Duration) -> Result<impl Client, CreateClientError> {
    TimeoutClient::new(timeout)
}

#[derive(Debug)]
pub struct CreateClientError {
    timeout: Duration,
    kind: CreateClientErrorKind,
}

#[derive(Debug)]
pub enum CreateClientErrorKind {
    OpenAIKey(std::env::VarError),
    BuildClient(reqwest::Error),
}

impl Display for CreateClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "error creating openai client with timeout {:?}",
            self.timeout,
        )
    }
}

impl Error for CreateClientError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            CreateClientErrorKind::OpenAIKey(err) => Some(err),
            CreateClientErrorKind::BuildClient(err) => Some(err),
        }
    }
}

impl TimeoutClient {
    /// Creates a new `OpenAI` client with timeout functionality
    ///
    /// # Arguments
    ///
    /// * `timeout` - A `Duration` representing the timeout for the client
    ///
    /// # Returns
    ///
    /// A `RetriableClient` object or an `Error` if the client could not be created
    ///
    /// # Errors
    ///
    /// Returns an error if the client could not be created or if the `OPENAI_API_KEY` environment variable is not set.
    fn new(timeout: Duration) -> Result<Self, CreateClientError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|err| CreateClientError {
            timeout,
            kind: CreateClientErrorKind::OpenAIKey(err),
        })?;
        let http_client = reqwest::ClientBuilder::new()
            // Limiting request duration
            .timeout(timeout)
            .build()
            .map_err(|err| CreateClientError {
                timeout,
                kind: CreateClientErrorKind::BuildClient(err),
            })?;
        let client = async_openai::Client::new()
            .with_api_key(api_key)
            .with_backoff(
                // It retries on on rate limit error
                backoff::ExponentialBackoffBuilder::default()
                    .with_initial_interval(Duration::from_secs(4))
                    .with_multiplier(2.0)
                    .with_max_interval(Duration::from_secs(20))
                    .with_max_elapsed_time(Some(timeout))
                    .build(),
            )
            .with_http_client(http_client);
        Ok(Self { client })
    }
}

#[async_trait]
impl Client for TimeoutClient {
    async fn ask_question(
        &self,
        system: &str,
        question: &str,
        model: &str,
        temperature: f32,
    ) -> Result<String, ChatError> {
        let request = build_request(system, question, model, temperature)?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|err| ChatError {
                model: model.to_string(),
                kind: ChatErrorKind::Request(err),
            })?;
        let mut response_mesasge = String::new();

        for chat_choice in response.choices {
            response_mesasge.push_str(&chat_choice.message.content);
        }

        Ok(response_mesasge)
    }
}

fn build_request(
    system: &str,
    question: &str,
    model: &str,
    temeperature: f32,
) -> Result<CreateChatCompletionRequest, ChatError> {
    let system_message = ChatCompletionRequestMessageArgs::default()
        .content(system)
        .role(Role::System)
        .build()
        .map_err(|err| ChatError {
            model: model.to_string(),
            kind: ChatErrorKind::BuildRequest(err.to_string()),
        })?;

    let user_message = ChatCompletionRequestMessageArgs::default()
        .content(question)
        .role(Role::User)
        .build()
        .map_err(|err| ChatError {
            model: model.to_string(),
            kind: ChatErrorKind::BuildRequest(err.to_string()),
        })?;

    let messages = [system_message, user_message];

    let max_tokens = u16::try_from(get_chat_completion_max_tokens(model, &messages).map_err(
        |err| ChatError {
            model: model.to_string(),
            kind: ChatErrorKind::BuildRequest(err.to_string()),
        },
    )?)
    .map_err(|_| ChatError {
        model: model.to_string(),
        kind: ChatErrorKind::BuildRequest("max tokens out of range".to_string()),
    })?;

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .max_tokens(max_tokens)
        .temperature(temeperature)
        .messages(messages)
        .build()
        .map_err(|err| ChatError {
            model: model.to_string(),
            kind: ChatErrorKind::BuildRequest(err.to_string()),
        })?;

    Ok(request)
}
