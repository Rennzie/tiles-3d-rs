import "./css/normalize.css";
import "./css/styles.css";
import { run } from "tiles-3d-rs";

console.log("📦 - Webpack started");
const canvas = document.createElement("canvas");
canvas.setAttribute("id", "tiles-3d-rs");
document.body.appendChild(canvas);

// Initialise the wasm program
await run();
