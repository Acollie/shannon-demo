// GLM Pocket: the on-device chat model (WebLLM 0.2.85, WebGPU) runs here, off the page's main thread.
// Only the WebLLM script (jsDelivr) and the model files (Hugging Face + the MLC model-lib wasm on GitHub) are downloaded;
// no prompt or reply leaves this device.
import { WebWorkerMLCEngineHandler } from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/+esm";
const handler = new WebWorkerMLCEngineHandler();
self.onmessage = msg => handler.onmessage(msg);
