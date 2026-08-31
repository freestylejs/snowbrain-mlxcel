# Native runtime backend fixtures

`sentencepiece-toy.model.b64` is the base64 encoding of `testdata/toy.model`
from the `sentencepiece` Rust crate version 0.13.2. The source archive is
available from <https://crates.io/api/v1/crates/sentencepiece/0.13.2/download>.

- Decoded size: 253,270 bytes
- Decoded SHA-256: `68e82cf4c9c3ef0be1f25a7cae2c836fcafa7d3c83821fb986411e56779ad1e2`
- Upstream license: MIT OR Apache-2.0; this copy is used under Apache-2.0.

The fixture is stored as text because repository edits in this execution lane
must be made through the patch mechanism. Tests verify the exact size and hash
before writing it into a private temporary directory and loading it through
the real SentencePiece backend.
