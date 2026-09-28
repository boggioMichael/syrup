# shared-types

The contract between the pieces of the lipreader example: `schema/*.schema.json`
(JSON Schema 2020-12) is the source of truth; `src/index.ts` mirrors it for
the Chrome extension and web code, `example/lipreader/lipreader/schema.py`
for the Python pipeline and API, `example/ios/Sources/LipReaderCore/Schema.swift`
for the iOS app. The Python tests validate real results against the schema.

`tsc -p tsconfig.json` writes `dist/index.js` and `dist/index.d.ts`; the
extension imports the declarations only.
