//! Lightweight scoring helpers for CLI outputs and test checks.

/// Compute mean squared error and squared correlation coefficient.
pub fn regression_metrics(predictions: &[f64], labels: &[f64]) -> (f64, f64) {
    let n = predictions.len();
    if n == 0 || n != labels.len() {
        return (0.0, 0.0);
    }

    let mut sumv = 0.0;
    let mut sumy = 0.0;
    let mut sumvv = 0.0;
    let mut sumyy = 0.0;
    let mut sumvy = 0.0;
    let mut total_error = 0.0;

    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        total_error += (pred - label) * (pred - label);
        sumv += pred;
        sumy += label;
        sumvv += pred * pred;
        sumyy += label * label;
        sumvy += pred * label;
    }

    let n_f = n as f64;
    let mse = total_error / n_f;
    let pred_var_term = n_f * sumvv - sumv * sumv;
    let label_var_term = n_f * sumyy - sumy * sumy;
    let r2 = if pred_var_term == 0.0 || label_var_term == 0.0 {
        0.0
    } else {
        let numerator = n_f * sumvy - sumv * sumy;
        (numerator * numerator) / (pred_var_term * label_var_term)
    };

    (mse, r2)
}

/// Compute classification accuracy in `[0, 100]` as percent.
pub fn accuracy_percentage(predictions: &[f64], labels: &[f64]) -> f64 {
    if predictions.is_empty() || predictions.len() != labels.len() {
        return 0.0;
    }

    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(&pred, &label)| pred == label)
        .count();

    100.0 * correct as f64 / labels.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_metrics_zero_error_is_zero_and_unit_r2() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let (mse, r2) = regression_metrics(&preds, &labels);
        assert!((mse - 0.0).abs() < 1e-12);
        assert!((r2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn regression_metrics_constant_labels_return_zero_r2() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![5.0, 5.0, 5.0];
        let (mse, r2) = regression_metrics(&preds, &labels);
        assert!((mse - 9.666666666666666).abs() < 1e-12);
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn accuracy_percentage_matches_simple_case() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 0.0, 3.0];
        assert_eq!(accuracy_percentage(&preds, &labels), 66.66666666666667);
    }

    #[test]
    fn accuracy_percentage_zero_when_no_predictions_match() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![4.0, 5.0, 6.0];
        assert_eq!(accuracy_percentage(&preds, &labels), 0.0);
    }

    #[test]
    fn regression_metrics_misaligned_lengths_returns_zero() {
        let (mse, r2) = regression_metrics(&[1.0, 2.0], &[1.0]);
        assert_eq!(mse, 0.0);
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn accuracy_percentage_misaligned_lengths_returns_zero() {
        let percent = accuracy_percentage(&[1.0, 2.0], &[1.0]);
        assert_eq!(percent, 0.0);
    }
}
