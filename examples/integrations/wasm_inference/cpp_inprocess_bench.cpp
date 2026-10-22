#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "svm.h"

namespace {

void print_null(const char *) {}

struct ParsedProblem {
  svm_problem prob{};
  std::vector<double> labels;
  std::vector<svm_node *> rows;
  std::vector<svm_node> x_space;
};

struct BenchResult {
  int train_rows = 0;
  int test_rows = 0;
  std::vector<double> train_samples_ms;
  std::vector<double> predict_samples_ms;
  std::vector<double> predictions;
  double accuracy = 0.0;
};

std::vector<std::string> split_ws(const std::string &line) {
  std::istringstream iss(line);
  std::vector<std::string> out;
  std::string tok;
  while (iss >> tok) {
    out.push_back(tok);
  }
  return out;
}

ParsedProblem load_problem(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open file: " + path);
  }

  std::vector<double> labels;
  std::vector<std::vector<svm_node>> rows_nodes;
  std::string line;
  int line_no = 0;

  while (std::getline(in, line)) {
    line_no++;
    if (line.empty()) {
      continue;
    }
    auto toks = split_ws(line);
    if (toks.empty()) {
      continue;
    }

    char *endp = nullptr;
    double y = std::strtod(toks[0].c_str(), &endp);
    if (endp == toks[0].c_str() || *endp != '\0') {
      throw std::runtime_error("invalid label at line " + std::to_string(line_no));
    }
    labels.push_back(y);

    std::vector<svm_node> nodes;
    int prev_index = 0;
    for (size_t i = 1; i < toks.size(); i++) {
      const auto &tok = toks[i];
      auto pos = tok.find(':');
      if (pos == std::string::npos) {
        throw std::runtime_error("invalid feature token at line " + std::to_string(line_no));
      }
      std::string idx_s = tok.substr(0, pos);
      std::string val_s = tok.substr(pos + 1);

      endp = nullptr;
      long idx_l = std::strtol(idx_s.c_str(), &endp, 10);
      if (endp == idx_s.c_str() || *endp != '\0' || idx_l <= prev_index || idx_l <= 0 || idx_l > INT32_MAX) {
        throw std::runtime_error("invalid feature index at line " + std::to_string(line_no));
      }

      endp = nullptr;
      double val = std::strtod(val_s.c_str(), &endp);
      if (endp == val_s.c_str() || *endp != '\0') {
        throw std::runtime_error("invalid feature value at line " + std::to_string(line_no));
      }

      svm_node node{};
      node.index = static_cast<int>(idx_l);
      node.value = val;
      nodes.push_back(node);
      prev_index = node.index;
    }

    svm_node end{};
    end.index = -1;
    end.value = 0.0;
    nodes.push_back(end);
    rows_nodes.push_back(std::move(nodes));
  }

  ParsedProblem parsed;
  parsed.labels = std::move(labels);
  parsed.prob.l = static_cast<int>(parsed.labels.size());
  parsed.rows.resize(parsed.prob.l, nullptr);

  size_t total_nodes = 0;
  for (const auto &row : rows_nodes) {
    total_nodes += row.size();
  }
  parsed.x_space.resize(total_nodes);

  size_t cursor = 0;
  for (int i = 0; i < parsed.prob.l; i++) {
    parsed.rows[i] = parsed.x_space.data() + cursor;
    const auto &src = rows_nodes[static_cast<size_t>(i)];
    for (size_t j = 0; j < src.size(); j++) {
      parsed.x_space[cursor + j] = src[j];
    }
    cursor += src.size();
  }

  parsed.prob.y = parsed.labels.data();
  parsed.prob.x = parsed.rows.data();
  return parsed;
}

svm_parameter default_param() {
  svm_parameter p{};
  p.svm_type = C_SVC;
  p.kernel_type = RBF;
  p.degree = 3;
  p.gamma = 1.0 / 13.0;
  p.coef0 = 0.0;
  p.cache_size = 100.0;
  p.eps = 1e-3;
  p.C = 1.0;
  p.nr_weight = 0;
  p.weight_label = nullptr;
  p.weight = nullptr;
  p.nu = 0.5;
  p.p = 0.1;
  p.shrinking = 1;
  p.probability = 0;
  return p;
}

std::vector<double> predict_all(const svm_model *model, const ParsedProblem &test) {
  std::vector<double> preds;
  preds.reserve(static_cast<size_t>(test.prob.l));
  for (int i = 0; i < test.prob.l; i++) {
    preds.push_back(svm_predict(model, test.prob.x[i]));
  }
  return preds;
}

double accuracy(const std::vector<double> &labels, const std::vector<double> &preds) {
  if (labels.empty() || labels.size() != preds.size()) {
    return 0.0;
  }
  size_t ok = 0;
  for (size_t i = 0; i < labels.size(); i++) {
    if (std::abs(labels[i] - preds[i]) < 1e-12) {
      ok++;
    }
  }
  return static_cast<double>(ok) / static_cast<double>(labels.size());
}

double to_ms(const std::chrono::steady_clock::duration &dur) {
  return std::chrono::duration<double, std::milli>(dur).count();
}

BenchResult run_bench(const ParsedProblem &train, const ParsedProblem &test, int warmup, int runs) {
  svm_set_print_string_function(print_null);
  svm_parameter param = default_param();
  const char *err = svm_check_parameter(&train.prob, &param);
  if (err != nullptr) {
    throw std::runtime_error(std::string("svm_check_parameter failed: ") + err);
  }

  for (int i = 0; i < warmup; i++) {
    svm_model *m = svm_train(&train.prob, &param);
    auto preds = predict_all(m, test);
    (void)preds;
    svm_free_and_destroy_model(&m);
  }

  int run_count = runs > 0 ? runs : 1;
  BenchResult out;
  out.train_rows = train.prob.l;
  out.test_rows = test.prob.l;
  out.train_samples_ms.reserve(static_cast<size_t>(run_count));
  out.predict_samples_ms.reserve(static_cast<size_t>(run_count));

  for (int i = 0; i < run_count; i++) {
    const auto t0 = std::chrono::steady_clock::now();
    svm_model *m = svm_train(&train.prob, &param);
    const auto t1 = std::chrono::steady_clock::now();
    out.train_samples_ms.push_back(to_ms(t1 - t0));

    const auto p0 = std::chrono::steady_clock::now();
    auto preds = predict_all(m, test);
    const auto p1 = std::chrono::steady_clock::now();
    out.predict_samples_ms.push_back(to_ms(p1 - p0));

    if (i == run_count - 1) {
      out.predictions = std::move(preds);
      out.accuracy = accuracy(test.labels, out.predictions);
    }
    svm_free_and_destroy_model(&m);
  }

  svm_destroy_param(&param);
  return out;
}

void write_json(const BenchResult &r, std::ostream &os) {
  os << std::setprecision(17);
  os << "{\n";
  os << "  \"train_rows\": " << r.train_rows << ",\n";
  os << "  \"test_rows\": " << r.test_rows << ",\n";
  os << "  \"train_samples_ms\": [";
  for (size_t i = 0; i < r.train_samples_ms.size(); i++) {
    if (i != 0) os << ", ";
    os << r.train_samples_ms[i];
  }
  os << "],\n";
  os << "  \"predict_samples_ms\": [";
  for (size_t i = 0; i < r.predict_samples_ms.size(); i++) {
    if (i != 0) os << ", ";
    os << r.predict_samples_ms[i];
  }
  os << "],\n";
  os << "  \"accuracy\": " << r.accuracy << ",\n";
  os << "  \"predictions\": [";
  for (size_t i = 0; i < r.predictions.size(); i++) {
    if (i != 0) os << ", ";
    os << r.predictions[i];
  }
  os << "]\n";
  os << "}\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: cpp_inprocess_bench <train_file> <test_file> <warmup> <runs>\n";
    return 2;
  }

  try {
    const std::string train_file = argv[1];
    const std::string test_file = argv[2];
    const int warmup = std::stoi(argv[3]);
    const int runs = std::stoi(argv[4]);

    if (warmup < 0 || runs <= 0) {
      throw std::runtime_error("warmup must be >= 0 and runs must be > 0");
    }

    ParsedProblem train = load_problem(train_file);
    ParsedProblem test = load_problem(test_file);
    BenchResult res = run_bench(train, test, warmup, runs);
    write_json(res, std::cout);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
