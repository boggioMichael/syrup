# lipreader — web page

One HTML file that uses the inference API from a browser: pick a video,
choose the mode, language and speakers, and get the people boxed and
subtitled over the player, a timeline lane per person (speaking spans and
segments; click to seek), the transcript with confidence per word, the four
exports, and a delete button that removes the result from the service.

```sh
python3 -m lipreader_api                 # in example/inference-api
python3 -m http.server 8792 --bind 127.0.0.1   # in example/web, then open http://127.0.0.1:8792/
```

The page never sends the video anywhere but the service address you give it
and says where that service processed it. `NODE_PATH=$(npm root -g) node
test/e2e.mjs` runs it in Chromium against the real service (upload,
transcript, seek, export, delete: 7 checks, passing in the build environment).
