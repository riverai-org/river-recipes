# River Recipes

Open-source recipes from [river.ai](https://river.ai) showing how to train LoRAs with the River API.

## The River API

River lets you fine-tune and sample from large hosted models through a single Python client — bring your data, train LoRA adapters on open-weight base models, and sample from the result.

- [Console](https://console.river.ai/) — create API keys and manage your account
- [Docs](https://docs.river.ai/) — guides and full API reference

## Getting started

```bash
pip install river-client
export RIVER_API_KEY="rv_..."   # create a key in the console
```

## Examples

Each example is self-contained — data preparation, training, and evaluation — and runs end to end with your own API key.

| Example | Description |
| ------- | ----------- |
| [text-to-sql](text-to-sql/) | Multi-turn text-to-SQL with reinforcement learning: the model explores a SQLite database with a SQL tool, then commits one query. Trains on BIRD-Platinum and evaluates on Arcwise-Plat-SQL. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — in particular the setup step that installs the leak-prevention hooks.

## License

[Apache License 2.0](LICENSE) — free to use, modify, and build on, including commercially.
