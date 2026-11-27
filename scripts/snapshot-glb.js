#!/usr/bin/env node
/**
 * Simple GLB snapshot helper using Playwright + Three.js.
 * Requires: npm install playwright (or npx playwright install-deps && npx playwright install chromium)
 *
 * Usage:
 *   node scripts/snapshot-glb.js path/to/model.glb out.png
 */

const { existsSync, readFileSync } = require("fs");
const path = require("path");

async function main() {
  const glbPath = process.argv[2];
  const outPath = process.argv[3] || "snapshot.png";
  if (!glbPath || !existsSync(glbPath)) {
    console.error("Usage: node scripts/snapshot-glb.js <model.glb> [out.png]");
    process.exit(1);
  }

  let chromium, firefox;
  try {
    ({ chromium, firefox } = require("playwright"));
  } catch (err) {
    console.error("Playwright is required. Install with: npm install playwright");
    process.exit(1);
  }

  const glbBase64 = readFileSync(glbPath).toString("base64");

  const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>GLB Snapshot</title>
  <style>html,body{margin:0;padding:0;overflow:hidden;background:#111;}canvas{width:100vw;height:100vh;display:block;}</style>
</head>
<body>
<canvas id="c"></canvas>
<script type="module">
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('c'), antialias: true, preserveDrawingBuffer: true });
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(6, 6, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0);
controls.update();

const dirLight = new THREE.DirectionalLight(0xffffff, 1.1);
dirLight.position.set(8, 10, 6);
scene.add(dirLight);
scene.add(new THREE.AmbientLight(0xffffff, 0.35));

const grid = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
scene.add(grid);

const loader = new GLTFLoader();

const base64 = "${glbBase64}";
const binary = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
const blob = new Blob([binary], { type: 'model/gltf-binary' });
const url = URL.createObjectURL(blob);

loader.load(url, (gltf) => {
  scene.add(gltf.scene);
  window.__loaded = true;
  URL.revokeObjectURL(url);
}, undefined, (err) => {
  console.error(err);
  window.__loaded = 'error';
});

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>`;

  let browser;
  const attempts = [
    { name: "webkit", launcher: require("playwright").webkit, opts: { headless: true } },
    { name: "firefox", launcher: firefox, opts: { headless: true } },
    {
      name: "chromium",
      launcher: chromium,
      opts: {
        headless: true,
        chromiumSandbox: false,
        args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
      },
    },
  ];

  const errors = [];
  for (const a of attempts) {
    try {
      browser = await a.launcher.launch(a.opts);
      break;
    } catch (err) {
      errors.push(`${a.name}: ${err.message}`);
    }
  }

  if (!browser) {
    console.error("Failed to launch any Playwright browser:\n" + errors.join("\n"));
    process.exit(1);
  }

  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  await page.setContent(html, { waitUntil: "networkidle" });
  await page.waitForFunction("window.__loaded === true", { timeout: 30000 }).catch(() => {});
  await page.screenshot({ path: outPath, fullPage: true });
  await browser.close();
  console.log(`Snapshot saved to ${outPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
