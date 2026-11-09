(() => {
  const canvas = document.getElementById("stageCanvas");
  const ctx = canvas.getContext("2d");
  const fileListEl = document.getElementById("fileList");
  const searchEl = document.getElementById("search");
  const pathEl = document.getElementById("currentPath");
  const statusEl = document.getElementById("saveStatus");
  const polyListEl = document.getElementById("polyList");
  const customTagEl = document.getElementById("customTag");
  const fileStatsEl = document.getElementById("fileStats");
  const emptyStateEl = document.getElementById("emptyState");

  const state = {
    files: [],
    total: 0,
    currentPath: null,
    image: null,
    polygons: [],
    selectedId: null,
    draft: [],
    mode: "draw", // draw | select
    clipboard: null,
    dirty: false,
    view: { scale: 1, ox: 0, oy: 0, imgW: 0, imgH: 0 },
    drag: null,
    pan: null,
  };

  function uid() {
    return `p_${Math.random().toString(36).slice(2, 9)}`;
  }

  function setStatus(text, saved = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("saved", saved);
  }

  function markDirty(v = true) {
    state.dirty = v;
    if (v) setStatus("저장 안 됨", false);
  }

  async function api(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    return res.json();
  }

  async function loadFiles(q = "") {
    const data = await api(`/api/images?q=${encodeURIComponent(q)}&limit=500`);
    state.files = data.items;
    state.total = data.total;
    fileStatsEl.textContent = `${data.items.length} / ${data.total}`;
    renderFileList();
  }

  function renderFileList() {
    fileListEl.innerHTML = "";
    for (const f of state.files) {
      const el = document.createElement("div");
      el.className = "file-item" + (f.path === state.currentPath ? " active" : "");
      el.innerHTML = `
        <span class="dot ${f.annotated ? "on" : ""}"></span>
        <div>
          <div class="name">${escapeHtml(f.name)}</div>
          <div class="sub">${escapeHtml(f.path)}</div>
        </div>
      `;
      el.onclick = () => openImage(f.path);
      fileListEl.appendChild(el);
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  async function openImage(path) {
    if (state.dirty && state.currentPath) {
      const ok = confirm("저장하지 않은 색인이 있습니다. 이동할까요?");
      if (!ok) return;
    }
    state.currentPath = path;
    pathEl.textContent = path;
    emptyStateEl.style.display = "none";
    renderFileList();

    const img = new Image();
    img.decoding = "async";
    const done = new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
    });
    img.src = `/api/image?path=${encodeURIComponent(path)}`;
    await done;
    state.image = img;

    const ann = await api(`/api/annotation?path=${encodeURIComponent(path)}`);
    state.polygons = (ann.polygons || []).map((p) => ({
      id: p.id || uid(),
      points: p.points,
      tags: p.tags || [],
      label: p.label || "parking_space",
    }));
    state.selectedId = state.polygons[0]?.id || null;
    state.draft = [];
    markDirty(false);
    setStatus("불러옴", true);
    fitImage();
    renderInspector();
    draw();
  }

  function fitImage() {
    if (!state.image) return;
    const wrap = canvas.parentElement;
    const cw = wrap.clientWidth;
    const ch = wrap.clientHeight;
    canvas.width = cw * devicePixelRatio;
    canvas.height = ch * devicePixelRatio;
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

    const iw = state.image.naturalWidth;
    const ih = state.image.naturalHeight;
    const scale = Math.min(cw / iw, ch / ih) * 0.98;
    state.view = {
      scale,
      ox: (cw - iw * scale) / 2,
      oy: (ch - ih * scale) / 2,
      imgW: iw,
      imgH: ih,
    };
  }

  function toImage(mx, my) {
    const { scale, ox, oy, imgW, imgH } = state.view;
    const x = (mx - ox) / scale / imgW;
    const y = (my - oy) / scale / imgH;
    return { x, y };
  }

  function toScreen(nx, ny) {
    const { scale, ox, oy, imgW, imgH } = state.view;
    return { x: ox + nx * imgW * scale, y: oy + ny * imgH * scale };
  }

  function occupancyOf(poly) {
    if (poly.tags.includes("occupied")) return "occupied";
    if (poly.tags.includes("empty") || poly.tags.includes("not occupied")) return "empty";
    if (poly.tags.includes("unknown")) return "unknown";
    return "unknown";
  }

  function colorFor(poly, alpha = 0.28) {
    const o = occupancyOf(poly);
    if (o === "occupied") return `rgba(46, 196, 182, ${alpha})`;
    if (o === "empty") return `rgba(231, 111, 81, ${alpha})`;
    return `rgba(154, 165, 181, ${alpha})`;
  }

  function strokeFor(poly) {
    const o = occupancyOf(poly);
    if (o === "occupied") return "#2ec4b6";
    if (o === "empty") return "#e76f51";
    return "#9aa5b5";
  }

  function draw() {
    const cw = canvas.width / devicePixelRatio;
    const ch = canvas.height / devicePixelRatio;
    ctx.clearRect(0, 0, cw, ch);

    if (!state.image) return;

    const { scale, ox, oy, imgW, imgH } = state.view;
    ctx.drawImage(state.image, ox, oy, imgW * scale, imgH * scale);

    for (const poly of state.polygons) {
      if (!poly.points?.length) continue;
      const selected = poly.id === state.selectedId;
      ctx.beginPath();
      poly.points.forEach((pt, i) => {
        const s = toScreen(pt[0], pt[1]);
        if (i === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
      });
      ctx.closePath();
      ctx.fillStyle = colorFor(poly, selected ? 0.38 : 0.22);
      ctx.fill();
      ctx.lineWidth = selected ? 2.5 : 1.5;
      ctx.strokeStyle = strokeFor(poly);
      ctx.stroke();

      // vertex handles when selected
      if (selected) {
        for (const pt of poly.points) {
          const s = toScreen(pt[0], pt[1]);
          ctx.beginPath();
          ctx.arc(s.x, s.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = "#fff";
          ctx.fill();
        }
      }

      // label
      const c = centroid(poly.points);
      const sc = toScreen(c[0], c[1]);
      ctx.fillStyle = "rgba(8,12,18,0.7)";
      ctx.fillRect(sc.x - 28, sc.y - 10, 56, 18);
      ctx.fillStyle = "#e7eef7";
      ctx.font = "11px IBM Plex Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText(occupancyOf(poly).slice(0, 8), sc.x, sc.y + 3);
    }

    // draft
    if (state.draft.length) {
      ctx.beginPath();
      state.draft.forEach((pt, i) => {
        const s = toScreen(pt[0], pt[1]);
        if (i === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
      });
      ctx.strokeStyle = "#f4a261";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const pt of state.draft) {
        const s = toScreen(pt[0], pt[1]);
        ctx.beginPath();
        ctx.arc(s.x, s.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#f4a261";
        ctx.fill();
      }
    }
  }

  function centroid(points) {
    let x = 0;
    let y = 0;
    for (const p of points) {
      x += p[0];
      y += p[1];
    }
    return [x / points.length, y / points.length];
  }

  function pointInPoly(x, y, points) {
    let inside = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
      const xi = points[i][0];
      const yi = points[i][1];
      const xj = points[j][0];
      const yj = points[j][1];
      const intersect =
        yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function hitTest(nx, ny) {
    for (let i = state.polygons.length - 1; i >= 0; i--) {
      const p = state.polygons[i];
      if (pointInPoly(nx, ny, p.points)) return p;
    }
    return null;
  }

  function closeDraft() {
    if (state.draft.length < 3) {
      setStatus("점은 3개 이상 필요");
      return;
    }
    const poly = {
      id: uid(),
      points: state.draft.map((p) => [clamp01(p[0]), clamp01(p[1])]),
      tags: ["empty"],
      label: "parking_space",
    };
    state.polygons.push(poly);
    state.selectedId = poly.id;
    state.draft = [];
    markDirty();
    renderInspector();
    draw();
  }

  function clamp01(v) {
    return Math.min(1, Math.max(0, v));
  }

  function copySelected() {
    const p = state.polygons.find((x) => x.id === state.selectedId);
    if (!p) {
      setStatus("복사할 폴리곤 없음");
      return;
    }
    state.clipboard = {
      points: p.points.map((pt) => [...pt]),
      tags: [...p.tags],
      label: p.label,
    };
    setStatus("폴리곤 복사됨");
  }

  function pasteClipboard() {
    if (!state.clipboard) {
      setStatus("클립보드 비어 있음");
      return;
    }
    const dx = 0.02;
    const dy = 0.02;
    const poly = {
      id: uid(),
      points: state.clipboard.points.map((pt) => [
        clamp01(pt[0] + dx),
        clamp01(pt[1] + dy),
      ]),
      tags: [...state.clipboard.tags],
      label: state.clipboard.label || "parking_space",
    };
    state.polygons.push(poly);
    state.selectedId = poly.id;
    // 연속 붙여넣기 시 점점 밀리도록 clipboard도 갱신
    state.clipboard = {
      points: poly.points.map((pt) => [...pt]),
      tags: [...poly.tags],
      label: poly.label,
    };
    markDirty();
    renderInspector();
    draw();
    setStatus("붙여넣기 완료");
  }

  function deleteSelected() {
    if (!state.selectedId) return;
    state.polygons = state.polygons.filter((p) => p.id !== state.selectedId);
    state.selectedId = state.polygons.at(-1)?.id || null;
    markDirty();
    renderInspector();
    draw();
  }

  function setOccupancyTag(tag) {
    const p = state.polygons.find((x) => x.id === state.selectedId);
    if (!p) return;
    const occ = new Set(["occupied", "empty", "not occupied", "unknown"]);
    p.tags = p.tags.filter((t) => !occ.has(t));
    p.tags.push(tag);
    markDirty();
    renderInspector();
    draw();
  }

  function addCustomTag() {
    const p = state.polygons.find((x) => x.id === state.selectedId);
    if (!p) return;
    const t = customTagEl.value.trim();
    if (!t) return;
    if (!p.tags.includes(t)) p.tags.push(t);
    customTagEl.value = "";
    markDirty();
    renderInspector();
  }

  function removeTag(tag) {
    const p = state.polygons.find((x) => x.id === state.selectedId);
    if (!p) return;
    p.tags = p.tags.filter((t) => t !== tag);
    markDirty();
    renderInspector();
    draw();
  }

  function renderInspector() {
    polyListEl.innerHTML = "";
    state.polygons.forEach((p, idx) => {
      const el = document.createElement("div");
      el.className = "poly-chip" + (p.id === state.selectedId ? " selected" : "");
      el.innerHTML = `<span>#${idx + 1} · ${occupancyOf(p)}</span><span>${p.points.length}pts</span>`;
      el.onclick = () => {
        state.selectedId = p.id;
        renderInspector();
        draw();
      };
      polyListEl.appendChild(el);
    });

    const p = state.polygons.find((x) => x.id === state.selectedId);
    document.querySelectorAll("[data-occ]").forEach((btn) => {
      btn.classList.remove("on-occupied", "on-empty", "on-unknown");
      if (!p) return;
      if (btn.dataset.occ === occupancyOf(p) || (btn.dataset.occ === "empty" && occupancyOf(p) === "empty")) {
        btn.classList.add(`on-${btn.dataset.occ}`);
      }
    });

    const tagsEl = document.getElementById("currentTags");
    tagsEl.innerHTML = "";
    if (p) {
      for (const t of p.tags) {
        const chip = document.createElement("button");
        chip.className = "tag";
        chip.textContent = `${t} ×`;
        chip.onclick = () => removeTag(t);
        tagsEl.appendChild(chip);
      }
    }
  }

  async function save() {
    if (!state.currentPath) return;
    await api("/api/annotation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: state.currentPath,
        polygons: state.polygons,
        meta: { updated_at: new Date().toISOString() },
      }),
    });
    markDirty(false);
    setStatus("저장됨", true);
    const f = state.files.find((x) => x.path === state.currentPath);
    if (f) f.annotated = true;
    renderFileList();
  }

  // events
  canvas.addEventListener("mousedown", (e) => {
    if (!state.image) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (e.button === 1 || e.altKey) {
      state.pan = { mx, my, ox: state.view.ox, oy: state.view.oy };
      return;
    }

    const { x, y } = toImage(mx, my);
    if (x < 0 || y < 0 || x > 1 || y > 1) return;

    if (state.mode === "select" || e.shiftKey) {
      const hit = hitTest(x, y);
      state.selectedId = hit?.id || null;
      if (hit) {
        state.drag = { id: hit.id, lx: x, ly: y };
      }
      renderInspector();
      draw();
      return;
    }

    // draw mode
    state.draft.push([x, y]);
    draw();
  });

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (state.pan) {
      state.view.ox = state.pan.ox + (mx - state.pan.mx);
      state.view.oy = state.pan.oy + (my - state.pan.my);
      draw();
      return;
    }

    if (state.drag) {
      const { x, y } = toImage(mx, my);
      const dx = x - state.drag.lx;
      const dy = y - state.drag.ly;
      const p = state.polygons.find((pp) => pp.id === state.drag.id);
      if (p) {
        p.points = p.points.map((pt) => [clamp01(pt[0] + dx), clamp01(pt[1] + dy)]);
        state.drag.lx = x;
        state.drag.ly = y;
        markDirty();
        draw();
      }
    }
  });

  window.addEventListener("mouseup", () => {
    state.drag = null;
    state.pan = null;
  });

  canvas.addEventListener("dblclick", (e) => {
    e.preventDefault();
    if (state.mode === "draw") closeDraft();
  });

  canvas.addEventListener(
    "wheel",
    (e) => {
      if (!state.image) return;
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const before = toImage(mx, my);
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      state.view.scale = Math.min(12, Math.max(0.05, state.view.scale * factor));
      const after = toScreen(before.x, before.y);
      state.view.ox += mx - after.x;
      state.view.oy += my - after.y;
      draw();
    },
    { passive: false }
  );

  window.addEventListener("keydown", async (e) => {
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;

    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      await save();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "c") {
      e.preventDefault();
      copySelected();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v") {
      e.preventDefault();
      pasteClipboard();
    } else if (e.key === "Enter") {
      closeDraft();
    } else if (e.key === "Escape") {
      state.draft = [];
      draw();
    } else if (e.key === "Delete" || e.key === "Backspace") {
      deleteSelected();
    } else if (e.key.toLowerCase() === "d") {
      setMode("draw");
    } else if (e.key.toLowerCase() === "v") {
      setMode("select");
    }
  });

  function setMode(mode) {
    state.mode = mode;
    document.getElementById("modeDraw").classList.toggle("active", mode === "draw");
    document.getElementById("modeSelect").classList.toggle("active", mode === "select");
  }

  document.getElementById("modeDraw").onclick = () => setMode("draw");
  document.getElementById("modeSelect").onclick = () => setMode("select");
  document.getElementById("btnClose").onclick = closeDraft;
  document.getElementById("btnCopy").onclick = copySelected;
  document.getElementById("btnPaste").onclick = pasteClipboard;
  document.getElementById("btnDelete").onclick = deleteSelected;
  document.getElementById("btnSave").onclick = () => save();
  document.getElementById("btnFit").onclick = () => {
    fitImage();
    draw();
  };
  document.getElementById("btnAddTag").onclick = addCustomTag;
  customTagEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") addCustomTag();
  });
  document.querySelectorAll("[data-occ]").forEach((btn) => {
    btn.onclick = () => setOccupancyTag(btn.dataset.occ === "empty" ? "empty" : btn.dataset.occ);
  });

  let searchTimer;
  searchEl.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadFiles(searchEl.value.trim()), 250);
  });

  window.addEventListener("resize", () => {
    fitImage();
    draw();
  });

  setMode("draw");
  loadFiles().catch((err) => {
    setStatus(`목록 오류: ${err.message}`);
  });
})();
