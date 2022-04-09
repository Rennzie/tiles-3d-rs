import { run } from "tiles-3d-rs";
console.log("📦 - Webpack started");
const root = document.createElement("div");
root.setAttribute("id", "tiles-3d-rs");
document.body.appendChild(root);

// Initialise the wasm program
await run();

window.addEventListener("click", () => console.log("clickety click"));
