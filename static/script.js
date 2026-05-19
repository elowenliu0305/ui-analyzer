// UI Analyzer — 前端交互逻辑
let currentData = null;

// ─── 视图切换（区域 ↔ 段落 ↔ 分隔线） ───
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".view-btn");
  if (!btn) return;
  if (btn.classList.contains("active")) return;

  const toggle = btn.parentElement;
  toggle.querySelectorAll(".view-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  const view = btn.dataset.view;
  const img = document.getElementById("annotatedImage");

  if (view === "lines" && currentData && currentData.lines_image) {
    img.src = currentData.lines_image + "?t=" + Date.now();
    img.dataset.currentView = "lines";
  } else if (view === "sections" && currentData && currentData.sections_image) {
    img.src = currentData.sections_image + "?t=" + Date.now();
    img.dataset.currentView = "sections";
  } else if (view === "annotated" && currentData) {
    img.src = currentData.annotated + "?t=" + Date.now();
    img.dataset.currentView = "annotated";
  }
});

// ─── 上传 ───
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");

uploadBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

uploadZone.addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON" && e.target.tagName !== "INPUT" && e.target.tagName !== "LABEL") {
    fileInput.click();
  }
});

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
  fileInput.value = "";
});

// ─── 分析流程 ───
async function uploadFile(file) {
  const modeToggle = document.getElementById("modeToggle");
  const mode = modeToggle.checked ? "sam" : "opencv";
  const modeName = mode === "sam" ? "SAM 精确检测" : "OpenCV 快速检测";
  showLoading(modeName + "...");

  document.getElementById("resultSection").style.display = "none";

  const form = new FormData();
  form.append("image", file);
  form.append("mode", mode);

  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: form });
    const text = await resp.text();
    let data;
    try { data = JSON.parse(text); }
    catch (e) {
      console.error("Response not JSON:", text.slice(0, 200));
      alert("服务器返回异常");
      hideLoading();
      return;
    }
    if (data.error) { alert(data.error); hideLoading(); return; }

    currentData = data;
    showResult(data, modeName);
    hideLoading();
  } catch (e) {
    alert("请求失败: " + e.message);
    hideLoading();
  }
}

// ─── 显示结果 ───
function showResult(data, modeName) {
  document.getElementById("resultSection").style.display = "block";

  const img = document.getElementById("annotatedImage");
  img.src = data.annotated + "?t=" + Date.now();
  img.dataset.currentView = "annotated";

  // 预加载段落图和线图，切换时不卡顿
  if (data.sections_image) {
    const preload = new Image();
    preload.src = data.sections_image;
  }
  if (data.lines_image) {
    const preload = new Image();
    preload.src = data.lines_image;
  }

  // 图例
  const typeColors = {
    "nav": "#ff0000", "search": "#00ffff", "content": "#00ff00",
    "card": "#ffa500", "button": "#ff00ff", "input": "#ffff00",
    "icon": "#ff8000", "text": "#c8c800", "list": "#00c8c8", "avatar": "#c86464",
    "footer": "#0000ff", "unknown": "#808080"
  };

  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  const seen = new Set();
  data.regions.forEach(r => {
    if (seen.has(r.type)) return;
    seen.add(r.type);
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-dot" style="background:${typeColors[r.type]||'#808080'}"></span>${r.type}`;
    legend.appendChild(item);
  });

  // 分隔线统计
  if (data.lines && data.lines.length > 0) {
    const lineTypes = {};
    data.lines.forEach(l => {
      const key = l.interface_subtype ? `${l.line_type}/${l.interface_subtype}` : l.line_type;
      lineTypes[key] = (lineTypes[key] || 0) + 1;
    });
    const lineStats = Object.entries(lineTypes).map(([k, v]) => `${k}:${v}`).join(" | ");
    const lineInfo = document.createElement("details");
    lineInfo.className = "lines-info";
    lineInfo.innerHTML = `
      <summary>📏 分隔线参考（${data.lines.length} 条）</summary>
      <div style="margin-top:4px;font-size:11px;color:#999">${lineStats}</div>
      <div style="margin-top:6px">
        <span class="line-dot" style="background:#ff0000"></span> 实体线
        <span class="line-dot" style="background:#ffa500;margin-left:8px"></span> 界面线(纯色/混色)
        <span class="line-dot" style="background:#ffff00;margin-left:8px"></span> 界面线(混色/混色)
      </div>
    `;
    legend.appendChild(lineInfo);
  }

  renderRegionList(data.regions, typeColors);
}

// ─── 区域列表（按段落分组 → 按交互功能分组） ───
function renderRegionList(regions, typeColors) {
  const list = document.getElementById("regionList");
  list.innerHTML = "";

  if (!typeColors) {
    typeColors = {
      "nav": "#ff0000", "search": "#00ffff", "content": "#00ff00",
      "card": "#ffa500", "button": "#ff00ff", "input": "#ffff00",
      "icon": "#ff8000", "text": "#c8c800", "list": "#00c8c8", "avatar": "#c86464",
      "footer": "#0000ff", "unknown": "#808080"
    };
  }

  // 功能分类规则
  function classifyRegion(r) {
    const action = (r.llm_action || "").toLowerCase();
    const type = (r.type || "").toLowerCase();
    const desc = (r.llm_desc || "").toLowerCase();
    const text = action + " " + type + " " + desc;

    if (/输入|键入|填写|搜索\b|键入/.test(text)) return "input";
    if (/点击|按钮|提交|确认|取消|删除|保存|关闭|打开|切换|跳转|选择|复制/.test(text)) return "click";
    if (/链接|导航|菜单|标签|返回|首页|tab|menu/.test(text)) return "nav";
    if (/滚动|滑动|拖动|轮播|翻页|滚动/.test(text)) return "scroll";
    if (/内容|文章|文本|文案|描述|说明|详情|介绍|标题|段落|文字/.test(text)) return "read";
    if (/图片|图像|照片|图标|头像|logo|avatar|icon/.test(text)) return "image";
    if (/卡片|card|列表|list|网格|grid/.test(text)) return "card";
    return "other";
  }

  // 分类定义
  const categories = {
    input: { label: "可输入", icon: "⌨", color: "#00b894", items: [] },
    click:  { label: "可点击", icon: "🖱", color: "#e17055", items: [] },
    nav:    { label: "导航跳转", icon: "🧭", color: "#6c5ce7", items: [] },
    scroll: { label: "可滚动/滑动", icon: "↕", color: "#0984e3", items: [] },
    read:   { label: "内容阅读", icon: "📖", color: "#2d3436", items: [] },
    image:  { label: "图片/图标展示", icon: "🖼", color: "#00cec9", items: [] },
    card:   { label: "卡片/列表", icon: "📇", color: "#fdcb6e", items: [] },
    other:  { label: "其他", icon: "▪", color: "#b2bec3", items: [] },
  };

  // 先按段落分组
  const sectionMap = {};
  regions.forEach(r => {
    const sid = (r.section && r.section.id !== undefined) ? r.section.id : 0;
    const slabel = (r.section && r.section.label) || "全页";
    if (!sectionMap[sid]) {
      sectionMap[sid] = { label: slabel, regions: [] };
    }
    sectionMap[sid].regions.push(r);
  });

  // 按段 ID 排序
  const sortedSectionIds = Object.keys(sectionMap).map(Number).sort((a, b) => a - b);
  const sectionColors = [
    "#e17055", "#0984e3", "#00b894", "#6c5ce7",
    "#fdcb6e", "#00cec9", "#e84393", "#636e72"
  ];

  sortedSectionIds.forEach((sid, si) => {
    const sec = sectionMap[sid];
    const sectionColor = sectionColors[si % sectionColors.length];

    // 段落标题
    const secGroup = document.createElement("div");
    secGroup.className = "section-group";

    const secHeader = document.createElement("div");
    secHeader.className = "section-group-header";
    secHeader.style.borderLeft = `4px solid ${sectionColor}`;
    secHeader.textContent = `${sec.label}（${sec.regions.length} 个元素）`;
    secGroup.appendChild(secHeader);

    // 段内按交互功能分组
    const catCopy = JSON.parse(JSON.stringify(categories));
    sec.regions.forEach(r => {
      const cat = classifyRegion(r);
      if (catCopy[cat]) catCopy[cat].items.push(r);
      else catCopy.other.items.push(r);
    });

    const priority = ["click", "input", "nav", "read", "card", "image", "scroll", "other"];
    priority.forEach(key => {
      const cat = catCopy[key];
      if (cat.items.length === 0) return;

      const secDiv = document.createElement("div");
      secDiv.className = "region-section";

      const header = document.createElement("div");
      header.className = "section-header";
      header.style.color = cat.color;
      header.textContent = `${cat.icon} ${cat.label}（${cat.items.length}）`;
      secDiv.appendChild(header);

      cat.items.forEach(r => {
        const div = document.createElement("div");
        div.className = "region-item";
        div.onclick = () => document.getElementById("imageWrapper").scrollIntoView({ behavior: "smooth" });

        const hasLLM = r.llm_desc && r.llm_action;

        div.innerHTML = `
          <div class="region-header">
            <span class="region-id">#${r.id}</span>
            <span class="cv-type-badge" style="border-left: 3px solid ${typeColors[r.type]||'#808080'}">${r.type}</span>
          </div>
          ${hasLLM ? `
            <div class="region-llm">
              <div class="llm-desc">${r.llm_desc}</div>
              <div class="llm-action">${cat.icon} ${r.llm_action}</div>
            </div>
          ` : `
            <div style="font-size:12px;color:#999">${r.action}</div>
          `}
          <div class="region-bbox">[${r.bbox.join(", ")}]</div>
          ${r.nearby_lines && r.nearby_lines.length > 0 ? `
            <div style="font-size:10px;color:#bbb;margin-top:2px">
              ${r.nearby_lines.map(l => `${l.line_type}(${l.distance}px)`).join(", ")}
            </div>
          ` : ""}
        `;
        secDiv.appendChild(div);
      });

      secGroup.appendChild(secDiv);
    });

    list.appendChild(secGroup);
  });
}

// ─── Loading ───
function showLoading(text) {
  const el = document.getElementById("loading");
  document.getElementById("loadingText").textContent = text || "正在分析...";
  el.style.display = "flex";
}
function hideLoading() {
  document.getElementById("loading").style.display = "none";
}
