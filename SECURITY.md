# Security Policy

## Scope and Support

`string2fingerprint` is a Linux-only Rust library that scores untrusted text
for malformed content and escape attempts using SIMD fingerprinting. The
[design document](docs/design.md) states the contracts that findings are
assessed against. `main` is the development branch for triage and fixes.
Release support and backport commitments will be established before publication.

Review repository code, documentation, dependencies, and build or CI
configuration when they affect safety or integrity in real use.

## Threat Model and Requirements

- Treat caller-controlled strings as untrusted. Define their accepted forms,
  interpretation, and failure behavior before stabilizing the interface.
- Preserve memory safety and bound input-driven work and allocation. Handle
  short, malformed, and boundary-length inputs through the documented interface.
- Execute CPU-specific instructions only when all required features and safety
  preconditions are established. Paths sharing a fingerprint or detection
  contract must agree on their specified results.
- Keep secrets and sensitive caller data out of diagnostics, CI logs, and
  public evidence. Preserve the integrity of dependencies, builds, and CI.

Assess findings using demonstrated reachability, attacker control, required
platform or configuration, and impact. Relevant impacts include memory safety
violations, input-triggered process failure or resource exhaustion, and
behavior that violates an established detection contract. Current checks
exercise the scaffold. Evaluate each runtime boundary when it is implemented.

## Reporting a Vulnerability

Use the [GitHub private vulnerability report form](https://github.com/OxiBelt/string2fingerprint/security/advisories/new).
If the form is unavailable, hold the details until a private route is
available. Include the affected revision, platform and CPU features, relevant
configuration, prerequisites, and observed impact. Share sanitized evidence
through the private report.

## Response and Disclosure

Maintainers validate reports and coordinate fixes privately. Use one private
report or repository advisory for each validated, report-worthy finding. Keep
it private until a supported fix or actionable mitigation is verified and
disclosure is authorized. Obtain explicit approval for the exact payload before
any advisory submission or publication. Check the repository, affected
revision, duplicate status, identity, permissions, and visibility before
sending, then verify the result. Request a CVE only when explicitly authorized.
