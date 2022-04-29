# Tiles-3D-RS
A 3D tiles render built with Rust and `wgpu` for the web.

## Pre-requisites
Install all the things:

- rustup
- cargo
- wasm-pack
- yarn


## Developing

The best workflow

- Build with `wasm-pack build`
- Link `pkg` and `www/package.json` with `yarn link`
- Start dev server with `yarn start`

Starts an http server in watch mode with the example. You'll find it a `localhost:8080`.

Make changes to Rust code then re-build with `wasm-pack build`. If all works well webpack will re-bundle the changes. Go to the webpage and hit refresh to see the new things.

### Serving static assets

WASM projects need static assets served by a webserver. Webpack can do this for us. Put any assets in the `www/public/assets` directory. The `resources.rs` module is configured to fetch directly from there. Use the file name directly when making a request. i.e `/cube.obj`.
