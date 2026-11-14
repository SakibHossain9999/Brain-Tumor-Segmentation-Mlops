import * as THREE from "three";

/**
 * Ghost brain: almost invisible where the surface faces the camera, glowing at the silhouette
 * (the Fresnel effect). Optional proximity glow: the brain also lights up near the tumour
 * ("focus + context": the eye is drawn to where the tumour sits).
 */
export function ghostMaterial(color = new THREE.Color(0x8a8f99)) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: color },
      uBase: { value: 0.04 }, // opacity where the surface faces the camera
      uEdge: { value: 0.55 }, // extra opacity at the silhouette
      uPower: { value: 2.5 }, // how quickly the glow fades from the edge toward the middle
      uFocus: { value: new THREE.Vector3() }, // tumour centre, in the brain mesh's own coordinates (mm)
      uFocusRadius: { value: 40.0 }, // how far the proximity glow reaches (mm)
      uFocusStrength: { value: 0.0 }, // 0 = off
      uFocusColor: { value: new THREE.Color(0xe0623a) },
      uOpacity: { value: 1.0 }, // overall brain opacity (multiplies everything above)
    },
    vertexShader: /* glsl */ `
      varying vec3 vNormal;
      varying vec3 vViewDir;
      varying vec3 vPos;
      #include <clipping_planes_pars_vertex>
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vec4 mvPosition = mv;
        #include <clipping_planes_vertex>
        vNormal = normalize(normalMatrix * normal);  // surface direction, in camera space
        vViewDir = normalize(-mv.xyz);               // direction from the surface to the camera
        vPos = position;                             // position in millimetres, for the proximity glow
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor, uFocus, uFocusColor;
      uniform float uBase, uEdge, uPower, uFocusRadius, uFocusStrength, uOpacity;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      varying vec3 vPos;
      #include <clipping_planes_pars_fragment>
      void main() {
        #include <clipping_planes_fragment>
        // 1 = surface faces the camera, 0 = surface seen edge-on (the silhouette)
        float facing = abs(dot(normalize(vNormal), normalize(vViewDir)));
        float rim = pow(1.0 - facing, uPower);
        // proximity: 1 at the tumour centre, fading smoothly to 0 beyond uFocusRadius
        float d = distance(vPos, uFocus);
        float near = uFocusStrength * exp(-(d * d) / (uFocusRadius * uFocusRadius));
        float alpha = uBase + uEdge * rim + near * (0.25 + 0.75 * rim);
        gl_FragColor = vec4(mix(uColor, uFocusColor, clamp(near, 0.0, 1.0)), alpha * uOpacity);
      }`,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    clipping: true,
  });
}

/**
 * Ink silhouette: only pen-like lines where the surface turns away from the viewer
 * (the outline and the deep folds); everything else is fully transparent.
 */
export function inkMaterial(color = 0x2a2a2e) {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) }, uWidth: { value: 0.22 }, uOpacity: { value: 1.0 } },
    vertexShader: /* glsl */ `
      varying vec3 vNormal;
      varying vec3 vViewDir;
      #include <clipping_planes_pars_vertex>
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vec4 mvPosition = mv;
        #include <clipping_planes_vertex>
        vNormal = normalize(normalMatrix * normal);
        vViewDir = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uWidth, uOpacity;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      #include <clipping_planes_pars_fragment>
      void main() {
        #include <clipping_planes_fragment>
        float facing = abs(dot(normalize(vNormal), normalize(vViewDir)));
        float line = 1.0 - smoothstep(uWidth, uWidth + 0.08, facing);  // crisp edge, slightly soft
        if (line < 0.02) discard;                                      // no line: draw nothing
        gl_FragColor = vec4(uColor, line * uOpacity);
      }`,
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    clipping: true,
  });
}

/** Toon shading: lighting snapped to a few flat bands instead of a smooth gradient. */
export function toonMaterial(color: THREE.Color, bands = 3, extra: THREE.MeshToonMaterialParameters = {}) {
  const data = new Uint8Array(bands);
  for (let i = 0; i < bands; i++) data[i] = Math.round(((i + 1) / bands) * 255);
  const gradient = new THREE.DataTexture(data, bands, 1, THREE.RedFormat);
  gradient.minFilter = THREE.NearestFilter; // no blending between bands: hard steps
  gradient.magFilter = THREE.NearestFilter;
  gradient.needsUpdate = true;
  return new THREE.MeshToonMaterial({ color, gradientMap: gradient, ...extra });
}

/**
 * Outline by the "inverted hull" trick: draw a slightly fattened copy of the mesh,
 * back faces only, in ink colour. Only its rim peeks out around the real mesh.
 */
export function outlineMaterial(thickness = 0.6, color = 0x1d1d1f) {
  return new THREE.ShaderMaterial({
    uniforms: { uThickness: { value: thickness }, uColor: { value: new THREE.Color(color) } },
    vertexShader: /* glsl */ `
      uniform float uThickness;
      #include <clipping_planes_pars_vertex>
      void main() {
        vec3 pushed = position + normal * uThickness;  // move each vertex outward (millimetres)
        vec4 mvPosition = modelViewMatrix * vec4(pushed, 1.0);
        #include <clipping_planes_vertex>
        gl_Position = projectionMatrix * mvPosition;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      #include <clipping_planes_pars_fragment>
      void main() {
        #include <clipping_planes_fragment>
        gl_FragColor = vec4(uColor, 1.0);
      }`,
    side: THREE.BackSide,
    clipping: true,
  });
}

/**
 * Hatching: shading drawn with pen strokes, like an engraving or a pencil sketch.
 * Lit parts get no strokes (just tinted paper), mid-tones one layer of parallel strokes,
 * darker parts cross-hatching (a second, then a third direction), and strokes thicken as it gets darker.
 *
 * Strokes are laid on the surface itself, in millimetres, so they stay attached to the tumour
 * while it rotates instead of sliding across it. A surface can face any way, so the strokes are
 * projected from three directions (x, y, z) and blended by how much the surface faces each one
 * ("triplanar projection").
 */
export function hatchMaterial(color: THREE.Color, spacing = 2.0) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uInk: { value: new THREE.Color() },   // stroke colour
      uPaper: { value: new THREE.Color() }, // colour between the strokes
      uSpacing: { value: spacing },         // millimetres between strokes
      uOpacity: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
      varying vec3 vNormal;     // camera space: for the lighting
      varying vec3 vObjNormal;  // mesh space: which way to project the strokes
      varying vec3 vPos;        // mesh space, millimetres: where the strokes are
      #include <clipping_planes_pars_vertex>
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        #include <clipping_planes_vertex>
        vNormal = normalize(normalMatrix * normal);
        vObjNormal = normal;
        vPos = position;
        gl_Position = projectionMatrix * mvPosition;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uInk, uPaper;
      uniform float uSpacing, uOpacity;
      varying vec3 vNormal;
      varying vec3 vObjNormal;
      varying vec3 vPos;
      #include <clipping_planes_pars_fragment>

      // How much of this pixel is covered by parallel strokes running at angle a.
      // width: share of the gap between strokes that a stroke covers (0 = none, 1 = solid)
      float strokes(vec2 p, float a, float width) {
        if (width <= 0.0) return 0.0;
        float u = dot(p, vec2(cos(a), sin(a))) / uSpacing;  // position across the strokes, in strokes
        float d = abs(fract(u) - 0.5);                      // 0 in the middle of a stroke, 0.5 between two
        float aa = max(fwidth(u), 1e-4);                    // one pixel, measured in strokes: soft edges
        float cover = 1.0 - smoothstep(width * 0.5 - aa, width * 0.5 + aa, d);
        // strokes thinner than a pixel would flicker: fade to their average grey instead
        return mix(cover, width, clamp(aa * 2.0 - 0.5, 0.0, 1.0));
      }

      // tone 1 = fully lit (no strokes) ... 0 = darkest (three crossing layers of thick strokes)
      float hatch(vec2 p, float tone) {
        float c = strokes(p, 0.785, clamp((0.85 - tone) / 0.35, 0.0, 1.0) * 0.55);      // 45 degrees
        c = max(c, strokes(p, -0.785, clamp((0.60 - tone) / 0.35, 0.0, 1.0) * 0.50));  // crossing: -45
        c = max(c, strokes(p, 0.0, clamp((0.35 - tone) / 0.30, 0.0, 1.0) * 0.45));     // third: 0
        return c;
      }

      void main() {
        #include <clipping_planes_fragment>
        vec3 light = normalize(vec3(1.0, 1.0, 2.0));  // like the scene's key light: up-right of the camera
        float tone = 0.2 + 0.8 * max(dot(normalize(vNormal), light), 0.0);
        // triplanar: weight each projection by how squarely the surface faces it. Sharpened (^8) so
        // mostly one projection shows: blending several gives a criss-cross grid instead of strokes
        vec3 w = pow(abs(normalize(vObjNormal)), vec3(8.0));
        w /= (w.x + w.y + w.z);
        float c = w.x * hatch(vPos.yz, tone) + w.y * hatch(vPos.xz, tone) + w.z * hatch(vPos.xy, tone);
        gl_FragColor = vec4(mix(uPaper, uInk, c), uOpacity);
        #include <colorspace_fragment>
      }`,
    clipping: true,
  });
  setHatchColor(material, color);
  return material;
}

/** Ink = a darker shade of the zone's colour, paper = a pale tint of it. */
export function setHatchColor(material: THREE.ShaderMaterial, color: THREE.Color) {
  material.uniforms.uInk.value.copy(color).multiplyScalar(0.45);
  material.uniforms.uPaper.value.copy(color).lerp(new THREE.Color(1, 1, 1), 0.8);
}
