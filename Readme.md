# Openai: Simple async-openai Abstraction

Openai is a simple async-openai abstraction library for Rust, which provides an easy way create requests to openai chat api.

## Usage

Add the following to your `Cargo.toml` file:

```toml
[dependencies]
openai = { git = "https://github.com/jkobejs/openai.git", branch = "master" }
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"
```

## Examples

Reading a file:

```rust
use std::time::Duration;

use openai::{live, Client};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = live(Duration::from_secs(60))?;

    let questions = vec![
       "Identify the date or day mentioned in the given text and provide it as the output Q: Dinner with Alice next Tuesday at Taco Bell A:".to_string(),
       "Identify the date or day mentioned in the given text and provide it as the output Q: CorpConf on 11/4 A:".to_string(),
       "Identify the date or day mentioned in the given text and provide it as the output Q: tomorrow at 10 am A:".to_string(),
    ];

    for q in questions {
        let a = client.ask_question("", &q, "gpt-3.5-turbo", 0.5).await?;

        println!("Q: {}", q);
        println!("A: {}", a);
    }

    Ok(())
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.