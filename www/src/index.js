import { run } from "tiles-3d-rs";
console.log("📦 - Webpack started");
const canvas = document.createElement("canvas");
canvas.setAttribute("id", "tiles-3d-rs");
document.body.appendChild(canvas);

// add a canvas to the screen
// add

// Initialise the wasm program
await run();

window.addEventListener("click", () => console.log("clickety click"));
