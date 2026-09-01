# Third-party notices

This directory contains attribution and license references for third-party model
materials included in Prism container images.

- `QWEN3-MODEL-NOTICE.txt` covers Qwen3 model metadata used by the Gateway
  profiles.
- `APACHE-2.0.txt` is the Apache License 2.0 text that applies to those Qwen3
  model materials.
- Prism-serve source code is licensed under [Apache-2.0](../LICENSE).

The Docker image copies these files to `/opt/prism/licenses/`, so attribution and
license terms remain available without a source checkout. Add a separate notice
for each additional third-party model or component; keep third-party terms
separate from the project `LICENSE`.
