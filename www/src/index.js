import { run } from "tiles-3d-rs";
console.log("its running fine");
const root = document.createElement("div");
root.setAttribute("id", "tiles-3d-rs");
document.body.appendChild(root);
run("tiles-3d-rs");
console.log("🦀 up and running");

window.addEventListener("click", () => console.log("clickety click"));
