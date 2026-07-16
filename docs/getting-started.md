# Getting Started

## Install

The macOS app is the simplest route: download the DMG from
[mtplx.com](https://mtplx.com/download), drag it to Applications, and let the
app select and install a model appropriate for the Mac.

For the CLI:

```bash
brew install youssofal/mtplx/mtplx
```

Pip is also supported: `python3 -m pip install mtplx`.

## Start

```bash
mtplx start
```

Onboarding selects a model, mode, and chat surface. Re-run it with `mtplx start
--fresh`. Commands that inspect configuration and metadata do not need MLX or a
loaded model.

## Save one setting

```bash
mtplx settings user set runtime.profile=sustained
mtplx settings explain runtime.profile
```

For a one-run change:

```bash
mtplx start --set generation.temperature=0.7
```

See [Settings](settings.md) for bundles, precedence, live scope, and secrets.

## Start the API server

```bash
mtplx quickstart --port 8000
```

The default bind exposes compatible OpenAI and Anthropic endpoints on
`127.0.0.1:8000`. Use `mtplx status` to inspect it and `mtplx stop` to stop it.

## Connect a client

```bash
mtplx connect openwebui
mtplx start opencode --port 18083
mtplx start pi --port 8000
```

The app also launches supported clients against the same server, avoiding a
second model load.
