#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");

function usageAndExit() {
  console.error(
    "Usage: node benchmark_node.cjs <pkg_js> <train_file> <test_file> <warmup> <runs>"
  );
  process.exit(2);
}

const args = process.argv.slice(2);
if (args.length < 5) {
  usageAndExit();
}

const [pkgJs, trainFile, testFile, warmupRaw, runsRaw] = args;
const pkgPath = path.resolve(pkgJs);
const trainText = fs.readFileSync(path.resolve(trainFile), "utf8");
const testText = fs.readFileSync(path.resolve(testFile), "utf8");
const warmup = Number.parseInt(warmupRaw, 10);
const runs = Number.parseInt(runsRaw, 10);

if (!Number.isFinite(warmup) || !Number.isFinite(runs)) {
  usageAndExit();
}

const wasm = require(pkgPath);
const payload = wasm.benchmark_from_libsvm_text(trainText, testText, warmup, runs);
console.log(payload);
