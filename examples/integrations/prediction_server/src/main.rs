use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use libsvm_rs::io::{load_model, load_problem, save_model};
use libsvm_rs::predict::predict_probability;
use libsvm_rs::train::svm_train;
use libsvm_rs::{set_quiet, KernelType, SvmNode, SvmParameter, SvmType};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    model: Arc<libsvm_rs::SvmModel>,
}

#[derive(Debug, Deserialize)]
struct Feature {
    index: i32,
    value: f64,
}

#[derive(Debug, Deserialize)]
struct PredictRequest {
    features: Vec<Feature>,
}

#[derive(Debug, Serialize)]
struct PredictResponse {
    label: f64,
    probabilities: Option<Vec<ClassProbability>>,
}

#[derive(Debug, Serialize)]
struct ClassProbability {
    class: i32,
    probability: f64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    model_path: String,
}

#[tokio::main]
async fn main() {
    set_quiet(true);

    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "data/heart_scale.server.model".to_string());
    let model_path_buf = PathBuf::from(model_path.clone());

    let model = ensure_model(&model_path_buf).unwrap_or_else(|e| {
        panic!(
            "failed to prepare model '{}': {e}",
            model_path_buf.display()
        )
    });

    let state = AppState {
        model: Arc::new(model),
    };

    let app = Router::new()
        .route(
            "/health",
            get({
                let model_path = model_path.clone();
                move || async move {
                    Json(HealthResponse {
                        status: "ok",
                        model_path,
                    })
                }
            }),
        )
        .route("/predict", post(predict_handler))
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(3000);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    println!("prediction_server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind listener");
    axum::serve(listener, app).await.expect("server failed");
}

fn ensure_model(model_path: &Path) -> Result<libsvm_rs::SvmModel, String> {
    if model_path.exists() {
        return load_model(model_path).map_err(|e| format!("load_model failed: {e}"));
    }

    let train_path = Path::new("data/heart_scale");
    let problem = load_problem(train_path).map_err(|e| format!("load_problem failed: {e}"))?;

    let param = SvmParameter {
        svm_type: SvmType::CSvc,
        kernel_type: KernelType::Rbf,
        gamma: 1.0 / 13.0,
        c: 1.0,
        probability: true,
        ..Default::default()
    };

    let model = svm_train(&problem, &param);
    save_model(model_path, &model).map_err(|e| format!("save_model failed: {e}"))?;
    Ok(model)
}

async fn predict_handler(
    State(state): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.features.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "features must not be empty".to_string(),
            }),
        ));
    }

    let mut nodes: Vec<SvmNode> = req
        .features
        .into_iter()
        .map(|f| SvmNode {
            index: f.index,
            value: f.value,
        })
        .collect();
    nodes.sort_by_key(|n| n.index);

    let (label, probabilities) =
        if let Some((pred, probs)) = predict_probability(&state.model, &nodes) {
            let pairs = state
                .model
                .label
                .iter()
                .zip(probs.iter())
                .map(|(class, prob)| ClassProbability {
                    class: *class,
                    probability: *prob,
                })
                .collect::<Vec<_>>();
            (pred, Some(pairs))
        } else {
            let pred = libsvm_rs::predict::predict(&state.model, &nodes);
            (pred, None)
        };

    Ok(Json(PredictResponse {
        label,
        probabilities,
    }))
}
