# prediction_server (Axum integration)

Minimal HTTP inference service using `libsvm-rs`.

- `GET /health`
- `POST /predict`

On first start, if `MODEL_PATH` does not exist, the server trains a small model from `data/heart_scale` and saves it.

## Run

```bash
bash examples/integrations/prediction_server/run.sh
```

## Test

Health:

```bash
curl -s http://127.0.0.1:3000/health | jq
```

Prediction:

```bash
curl -s -X POST http://127.0.0.1:3000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[{"index":1,"value":0.7},{"index":2,"value":-1.1},{"index":3,"value":0.2}]}' | jq
```

## Environment variables

- `PORT` (default: `3000`)
- `MODEL_PATH` (default: `data/heart_scale.server.model`)
