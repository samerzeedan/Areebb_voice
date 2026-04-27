// Minimal AudioWorklet that drains a Float32 ring buffer at the device sample
// rate. The main thread feeds frames via `port.postMessage(float32Array)`.

class PcmRingPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.cursor = 0;
    this.port.onmessage = (event) => {
      if (!event.data) return;
      if (event.data === 'flush') {
        this.queue = [];
        this.cursor = 0;
        return;
      }
      this.queue.push(event.data);
    };
  }

  process(_inputs, outputs) {
    const channel = outputs[0][0];
    if (!channel) return true;
    let written = 0;
    while (written < channel.length) {
      if (this.queue.length === 0) {
        channel.fill(0, written);
        return true;
      }
      const head = this.queue[0];
      const remaining = head.length - this.cursor;
      const need = channel.length - written;
      const take = Math.min(remaining, need);
      channel.set(head.subarray(this.cursor, this.cursor + take), written);
      this.cursor += take;
      written += take;
      if (this.cursor >= head.length) {
        this.queue.shift();
        this.cursor = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-ring-player', PcmRingPlayer);
