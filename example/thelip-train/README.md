# thelip-train

Training the lip-reading models thelip-server serves, from video whose
licence allows it, under a budget of hours and money. English is a
fine-tuning of the best published open model on video that looks like a
phone in front of a face; Hebrew — for which no visual speech model or
corpus exists anywhere public — is the first one, its decoder learned from
scratch on the encoder that English transferred.

Everything runs on one rented Linux GPU machine. One command:

```sh
curl -sSL https://raw.githubusercontent.com/boggioMichael/syrup/claude/intents-2/example/thelip-train/bootstrap.sh \
  | bash -s -- --languages en,he --hours 40 --hourly-cost 3.5 --video-hours en=150,he=80
```

`--hours` is the wall-clock budget: the run stops starting stages when it
is spent, and says where it got to; run again and it continues from
`state.json`. `--hourly-cost` is what the machine costs, so that every
message says what has been spent. Both are the two limits that were set for
this work: at most 48 hours, at most $10 per hour.

## What was and was not run

Honest labelling: the pieces of this pipeline that can run without a GPU,
a download or a model have been run, on GRID's sample clips, by
`test_train.py` (in CI): text rules, error rates, caption parsing,
utterance cutting, the split, the manifest format, the patch to Auto-AVSR's
trainer, and the mouth crops (syrup's own face and eye cascades for the
points, Chaplin's alignment for the crop). Fetching (yt-dlp), transcription
(faster-whisper), training and evaluation (Auto-AVSR on CUDA) have **not**
been run yet: they need the GPU machine, whose account and bill are the
owner's. The numbers below marked *expected* are expectations, not results;
`exp/<lang>/results.json` and the model card in `models/` hold the measured
ones once a run has happened.

## Data, and its licences

Only material that may be used for this: works of the US federal
government (public domain), uploads whose owner chose a Creative Commons
BY / BY-SA / CC0 licence (YouTube's own licence field decides; Wikimedia
Commons' per-file licence decides), and MIT OpenCourseWare (CC BY-NC-SA
4.0, which the non-commercial models this trains are compatible with).
Films, broadcasts, and anything under an ordinary copyright are not
touched, whatever their availability. Every downloaded item keeps its
licence in its `info.json`, and each training clip carries it in the
manifests.

| source | language | licence | what it is |
|---|---|---|---|
| `nasa` | en | public domain (NASA) | briefings and interviews from images.nasa.gov, often with .srt captions |
| `whitehouse`, `statedept` | en | public domain (US government) | press briefings and remarks, a speaker at a podium |
| `mit-ocw` | en | CC BY-NC-SA 4.0 | lectures; wide shots are dropped by the face-size test |
| `youtube-cc-*` | en, he, es, fr, de, ar | CC BY (YouTube's licence field) | interviews, talks, vlogs to camera, found with YouTube's Creative Commons filter |
| `wikimedia-*` | en, he | per file | interviews, speeches and talks on Wikimedia Commons |
| `phone` | any | consented users of thelip.ai | mouth crops thelip-server kept when the *improve* switch was on, with the reader's correction |

The phone samples are the ones that matter most for a phone: they are
test material first (so that the number reported is the number a user
sees), and training material once there are enough of them.

## Stages

`run_all.py` runs them in order and skips the ones `state.json` says are done.

1. **tools** — Auto-AVSR (pinned `182b628`) and Chaplin (pinned `7aee1f8`),
   the base checkpoint `vsr_trlrs2lrs3vox2avsp_base.pth` (3,448 hours of
   training video, 20.3% WER on LRS3; the authors' Google Drive), the face
   detector.
2. **fetch** — video per source, to the hours asked for (`--video-hours`),
   with the licence check above, through `yt-dlp` (YouTube) or plain
   downloads (NASA, Wikimedia). Candidates of 1–120 minutes.
3. **label** — a transcript with word times per video: the source's human
   captions when it has them (YouTube's, NASA's), else Whisper through
   faster-whisper with word timestamps — `large-v3` for most languages,
   [ivrit.ai](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2)'s
   Hebrew-tuned Whisper for Hebrew. A Whisper transcript is not a human
   one: some of its words are wrong, and the model learns those too; the
   per-word probabilities are kept in the labels for a later, stricter cut.
4. **segment** — the face has to be there and big enough (checked at
   intervals over the video; lectures shot wide fail here and are dropped),
   the transcript is cut into utterances of 0.8–12 seconds at sentence ends
   and pauses over 0.6 s, and each utterance becomes one clip: 96×96 grey mouth
   crops at 25 fps, aligned the way Auto-AVSR's data was (Chaplin's
   `VideoProcess`), written as an mp4 next to its text.
5. **phone** — thelip-server's `data/` (given with `--phone-data`): the
   kept crops with their corrected texts become test clips.
6. **manifest** — a SentencePiece unigram tokenizer for the language
   (Auto-AVSR's own 5000-piece English one is kept for English; Hebrew
   gets its own, without niqqud), then `train.csv` / `val.csv` / `test.csv`
   in Auto-AVSR's format. The test set is a whole held-out source plus
   every phone sample, so that no speaker or video straddles train and
   test.
7. **baseline** — the base model's error rate on the test set, English
   only (a new language has no base to measure).
8. **train** — Auto-AVSR's own `train.py`, unchanged but for three lines
   (CSV logging instead of wandb, no SLURM, a wall-clock limit), in a copy
   of the repository per language with the language's tokenizer in place.
   English adapts the whole model from the base checkpoint at a low
   learning rate; another language transfers the front end and the
   Conformer encoder and learns a new decoder and CTC head
   (`--transfer-encoder`). The wall clock is capped to what the budget
   leaves, and the last five epochs are averaged into the model to serve.
9. **evaluate** — the fine-tuned model on the same test set, per source.
10. **export** — `models/<lang>-thelip-v1/` with `model.pth`, the
    tokenizer, `info.json` (base, hours by source with licence, results)
    and a model card with the measured numbers; `--push
    boggioMichael/thelip` also publishes it to Hugging Face.

Copy `models/<lang>-thelip-v1/` into thelip-server's `models/` (on Windows,
`C:\Users\Public\thelip-server\models\`) and start the server: it serves a
trained model for a language that has no published one (Hebrew), and for
English only when the model's card shows it beat the published model on
the phone samples.

## Budget

At $3.50 per hour for an H100 (the usual RunPod price; the limit is $10),
40 hours is about $140. A rough split, for `en=150,he=80` hours of video:

| stage | hours (expected) |
|---|---|
| fetch 230 h of video | 6–10 (network-bound) |
| label with Whisper | 3–5 |
| segment | 4–6 |
| train English (8 epochs) | 6–8 |
| train Hebrew (30 epochs) | 8–10 |
| evaluate, export | 1 |

The run checks the hours left before every stage and before training, and
splits what is left between the languages still to train.

## What to expect

*Expected, not measured.* Lip reading is probabilistic: many sounds look
the same on the lips, and a model's number is a measurement of how often it
guesses right, never a promise.

- **English**: the published model reads TED talks at 19.1% word error
  rate. On phone clips of an untrained speaker it does markedly worse (the
  baseline stage measures how much). Fine-tuning on 150 hours of talking
  faces plus the consented phone samples should bring the phone number
  down; the model is served only if it does.
- **Hebrew**: 80 hours is a small corpus for a new language, and mixed
  video from the web is harder than the studio newsreaders behind the
  Mandarin model's 8% character error rate. The first Hebrew model will
  make many mistakes; its value is that it exists and improves with every
  corrected reading on thelip.ai. Its error rate goes on its model card.
- **Spanish, French, Portuguese, Mandarin**: published models exist
  (languages.py) and are served already; this pipeline can adapt them
  later the same way as English.
- **Arabic, German**: sources are listed; training them works like Hebrew
  (new decoder on the transferred encoder) once video hours are given.

## Running it by hand

```sh
pip install -r requirements.txt            # CUDA PyTorch from PyPI, Lightning, faster-whisper, yt-dlp, mediapipe
export THELIP_TRAIN_HOME=/workspace/thelip/work
python run_all.py --languages he --hours 20 --hourly-cost 3.5 --video-hours he=80 --phone-data /path/to/thelip-server/data
python run_all.py --only fetch --languages he      # one stage
```

Tests, without a GPU (Chaplin and Auto-AVSR checkouts, ffmpeg, and
LipNet's GRID sample clips):

```sh
THELIP_CHAPLIN=/path/to/chaplin THELIP_AUTO_AVSR=/path/to/auto_avsr python test_train.py
```

## Credits

Auto-AVSR (Ma, Petridis & Pantic, ICASSP 2023; Apache-2.0 code, model
weights for non-commercial use), Chaplin (Amanvir Parhar, MIT), the
Imperial College preprocessing (Apache-2.0), faster-whisper (MIT) and
OpenAI's Whisper weights (MIT), ivrit.ai's Hebrew Whisper (Apache-2.0),
yt-dlp (Unlicense), SentencePiece (Apache-2.0). Data: the sources above,
each under its own licence, and the people who left the improve switch on.
