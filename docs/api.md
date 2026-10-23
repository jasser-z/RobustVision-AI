# API

- `GET /health` returns service and explicit device metadata.
- `GET /models` returns model identifiers and parameter counts.
- `GET /checkpoints` returns locally saved, allow-listed checkpoint identifiers.
- `GET /corruptions` returns supported severity mappings.
- `POST /predict/{checkpoint_id}` accepts an image multipart field named `file`; the identifier must name a saved local `.pt` or `.pth` checkpoint.
- `POST /preview-corruption/{name}?severity=1..5&seed=...` returns a PNG.

Uploads are size-limited, decoded via Pillow, and invalid data produces a safe HTTP error. The API does not accept model paths or user-supplied checkpoints.
