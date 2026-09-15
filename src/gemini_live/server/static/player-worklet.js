// Plays model audio from a ring buffer.
// `flush` is wired to the Live API's `interrupted` signal so barge-in silences
// the previous response immediately instead of letting it drain.
class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._queue = [];
    this._offset = 0;
    this._playing = false;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (data === 'flush') {
        this._queue = [];
        this._offset = 0;
        return;
      }
      this._queue.push(new Float32Array(data));
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    if (!output) return true;

    let written = 0;
    while (written < output.length && this._queue.length > 0) {
      const head = this._queue[0];
      const available = head.length - this._offset;
      const need = output.length - written;
      const take = Math.min(available, need);

      output.set(head.subarray(this._offset, this._offset + take), written);
      written += take;
      this._offset += take;

      if (this._offset >= head.length) {
        this._queue.shift();
        this._offset = 0;
      }
    }
    if (written < output.length) output.fill(0, written);

    const nowPlaying = this._queue.length > 0;
    if (nowPlaying !== this._playing) {
      this._playing = nowPlaying;
      this.port.postMessage({ playing: nowPlaying });
    }
    return true;
  }
}

registerProcessor('player-processor', PlayerProcessor);
