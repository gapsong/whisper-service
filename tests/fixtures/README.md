# Test speech

German sentences synthesized with [Piper](https://github.com/rhasspy/piper) using the "Thorsten" voice ([`de_DE-thorsten-medium`](https://huggingface.co/rhasspy/piper-voices/tree/main/de/de_DE/thorsten/medium)), then converted to 16 kHz mono 16-bit WAV.
The voice model is MIT licensed; the [Thorsten-Voice](https://github.com/thorstenMueller/Thorsten-Voice) dataset it was trained on is CC0.

| File | Text |
| --- | --- |
| `speech_message.wav` | Ich schreibe gerade eine Nachricht an Max. |
| `speech_meeting.wav` | Wir treffen uns morgen um zehn. |
| `speech_ja.wav` | Ja. |

Regenerate one (needs `uv`, the voice files from the link above, and macOS `afconvert`):

```sh
echo "Ja." | uvx --from piper-tts piper -m de_DE-thorsten-medium.onnx -f raw.wav
afconvert -f WAVE -d LEI16@16000 -c 1 raw.wav speech_ja.wav
```
