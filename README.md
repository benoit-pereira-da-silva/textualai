# textualai

**textualai** is a set of high-performance, streaming-first building blocks for working with Large Language Model (LLM) providers.

It is designed for developers who need **fine-grained control over streaming**, **strong typing**, and **composable processing pipelines**, with **performance and reliability as first-class concerns**.

---

## Why Go (Golang)?

**textualai is built in Go**, which is a deliberate and strategic choice:

- **High-performance concurrency** via goroutines and channels
- **Predictable latency and low memory overhead**
- **Excellent async I/O and networking support**
- **Strong static typing** without sacrificing developer productivity
- **Easy deployment** with single, self-contained binaries

These properties make Go particularly well-suited for **long-lived streaming pipelines**, **LLM token streams**, and **low-level orchestration** of model interactions.

---

## Providers

Currently focused on **OpenAI-compatible APIs**:

- [OpenAI Platform](https://platform.openai.com/docs/overview)
- [Ollama](https://docs.ollama.com)
- *More providers coming soon…*

---

## Core Capabilities

- **Streaming-first architecture** (default mode)
- **Thinking / reasoning streaming**
- **Structured outputs**
- **Function calling (tools)**

### Coming Soon 🚀
- Embeddings
- Vision models
- Web search
- Conversation persistence

---

## Architecture & Design

Built on top of **[Textual](https://github.com/benoit-pereira-da-silva/textual)** to deliver **low-level, high-performance streaming processing**, including:

- Processor chaining
- Predicate-based routing
- Async I/O
- Strongly-typed transcoding
- Extensible streaming pipelines

![Textual logo](https://github.com/benoit-pereira-da-silva/textual/blob/main/assets/logo.png)

---

## License

Licensed under the **Apache License, Version 2.0**.  
See the `LICENSE` file for details.
