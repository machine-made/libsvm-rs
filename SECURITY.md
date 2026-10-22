# Security Policy

## Supported versions

Security fixes go into the latest release on crates.io (currently 0.9.x).

## Reporting a vulnerability

GitHub issues are disabled for this repository. Report a vulnerability
privately through GitHub instead:
[Security → Report a vulnerability](https://github.com/ricardofrantz/libsvm-rs/security/advisories/new).

Please include:

- the `libsvm-rs` version or commit,
- the model file, problem file, or serde payload that triggers the problem,
- the command or API call you ran, and what happened.

## Scope

The loaders treat model files, problem files, and serde payloads as untrusted
input. [`.oss-scanner/threat_model.md`](.oss-scanner/threat_model.md) lists the
attack surface and how severity is rated.
[`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) lists earlier findings and their fixes.
