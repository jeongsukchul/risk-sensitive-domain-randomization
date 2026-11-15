"use strict";

const demoData = window.RSDR_DEMOS || {};
const gallery = document.getElementById("demo-gallery");
const dialog = document.getElementById("media-dialog");
const dialogContent = document.getElementById("dialog-content");
let originatingMedia = null;
let mediaTrigger = null;
function rewardComparison(rewards) {
  const comparison = document.createElement("div");
  comparison.className = "demo-rewards";
  comparison.setAttribute("aria-label", "Episode reward");
  for (const [key, label] of [["udr", "UDR"], ["rsdr", "RSDR"]]) {
    const entry = document.createElement("span");
    const policy = document.createElement("small");
    policy.textContent = label;
    const value = document.createElement("strong");
    value.textContent = rewards[key].toFixed(1);
    entry.append(policy, value);
    comparison.append(entry);
  }
  return comparison;
}
function openMedia(element, title, trigger) {
  if (typeof dialog.showModal !== "function") {
    window.open(element.currentSrc || element.src, "_blank", "noopener");
    return;
  }
  originatingMedia = element;
  mediaTrigger = trigger;
  document.getElementById("dialog-title").textContent = title;
  dialogContent.replaceChildren();
  if (element.tagName === "VIDEO") {
    const expanded = document.createElement("video");
    expanded.controls = true;
    expanded.setAttribute("controlsList", "nodownload");
    expanded.muted = true;
    expanded.loop = true;
    expanded.playsInline = true;
    expanded.src = element.currentSrc || element.querySelector("source").src;
    expanded.addEventListener("loadedmetadata", () => {
      expanded.currentTime = element.currentTime;
      expanded.play().catch(() => {});
    }, {once:true});
    element.pause();
    const legend = document.createElement("div");
    legend.className = "comparison-labels";
    for (const text of ["UDR", "RSDR (Ours)"]) {
      const label = document.createElement("span");
      label.textContent = text;
      legend.append(label);
    }
    dialogContent.append(legend, expanded);
    const demo = demoData[element.closest("[data-demo]")?.dataset.demo];
    if (demo?.conditionNote) {
      const note = document.createElement("p");
      note.className = "condition-note";
      note.textContent = demo.conditionNote;
      dialogContent.append(note);
    }
    const rewards = demo?.rewards;
    if (rewards) dialogContent.append(rewardComparison(rewards));
  } else {
    const image = document.createElement("img");
    image.src = element.currentSrc || element.src;
    image.alt = element.alt;
    dialogContent.append(image);
  }
  dialog.showModal();
}
function closeMedia() { dialog.close(); }
function cleanupMedia() {
  const expanded = dialogContent.querySelector("video");
  if (expanded) {
    if (originatingMedia) originatingMedia.currentTime = expanded.currentTime;
    expanded.pause();
    expanded.removeAttribute("src");
    expanded.load();
  }
  dialogContent.replaceChildren();
  originatingMedia = null;
  if (mediaTrigger) mediaTrigger.focus();
  mediaTrigger = null;
}
document.getElementById("close-media").addEventListener("click", closeMedia);
dialog.addEventListener("close", cleanupMedia);
dialog.addEventListener("click", event => {if (event.target === dialog) closeMedia();});
if (Object.keys(demoData).length) {
  gallery.replaceChildren();
  for (const [name, demo] of Object.entries(demoData)) {
    const figure = document.createElement("figure");
    figure.dataset.demo = name;
    const video = document.createElement("video");
    video.controls = true;
    video.setAttribute("controlsList", "nodownload");
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.poster = demo.poster;
    video.setAttribute("aria-label", demo.title + ": UDR on the left, RSDR on the right");
    const source = document.createElement("source");
    source.src = demo.video;
    source.type = "video/mp4";
    video.append(source);
    const caption = document.createElement("figcaption");
    const title = document.createElement("strong");
    title.textContent = demo.title;
    const tags = document.createElement("div");
    tags.className = "case-tags";
    for (const label of demo.caseTags || []) {
      const tag = document.createElement("span");
      tag.textContent = label;
      tags.append(tag);
    }
    caption.append(title, tags);
    if (demo.conditionNote) {
      const note = document.createElement("p");
      note.className = "condition-note";
      note.textContent = demo.conditionNote;
      caption.append(note);
    }
    if (demo.rewards) {
      const rewardLabel = document.createElement("p");
      rewardLabel.className = "reward-label";
      rewardLabel.textContent = "Episode reward";
      caption.append(rewardLabel, rewardComparison(demo.rewards));
    }
    const tools = document.createElement("div");
    tools.className = "demo-tools";
    const expand = document.createElement("button");
    expand.type = "button";
    expand.className = "expand-video";
    expand.textContent = "↗ Expand";
    expand.setAttribute("aria-label", "Expand " + demo.title + " video");
    expand.addEventListener("click", () => openMedia(video, demo.title + " — " + (demo.caseTags || []).join(" · "), expand));
    tools.append(expand);
    const labels = document.createElement("div");
    labels.className = "comparison-labels";
    const udr = document.createElement("span");
    udr.textContent = "UDR";
    const rsdr = document.createElement("span");
    rsdr.append("RSDR ");
    const ours = document.createElement("small");
    ours.textContent = "(Ours)";
    rsdr.append(ours);
    labels.append(udr, rsdr);
    figure.append(labels, video, caption, tools);
    gallery.append(figure);
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches && "IntersectionObserver" in window) {
      const observer = new IntersectionObserver(entries => {
        for (const entry of entries) {
          if (entry.isIntersecting && !dialog.open) video.play().catch(() => {});
          else video.pause();
        }
      }, {threshold: 0.3});
      observer.observe(video);
    }
  }
}
for (const link of document.querySelectorAll("[data-zoom]")) {
  link.addEventListener("click", event => {
    if (typeof dialog.showModal !== "function") return;
    event.preventDefault();
    const image = link.querySelector("img");
    openMedia(image, image.alt, link);
  });
}

const resultData = {
  cvar: [[386.42,48.05,360.84,88.85,496.01,45.81],[499.04,41.93,548.44,26.35,570.84,19.43],[641.75,33.99,697.49,62.73,722.14,32.45],[629.43,15.26,637.63,11.96,671.70,10.07]],
  minimum: [[170.02,47.82,178.76,57.95,383.93,59.77],[283.61,41.58,304.59,15.96,337.47,38.53],[446.00,71.00,484.97,77.23,563.43,52.16],[296.47,28.78,308.38,33.25,432.03,42.81]]
};
for (const button of document.querySelectorAll("[data-metric]")) button.addEventListener("click", () => {
  const metric = button.dataset.metric;
  for (const other of document.querySelectorAll("[data-metric]")) other.setAttribute("aria-pressed", String(other === button));
  document.querySelectorAll("#result-rows tr").forEach((row, i) => {
    row.querySelectorAll("td").forEach((cell, j) => {
      const values = resultData[metric][i];
      cell.replaceChildren();
      const value = document.createElement(j === 2 ? "strong" : "span");
      value.textContent = values[j * 2].toFixed(2);
      const deviation = document.createElement("small");
      deviation.textContent = " ± " + values[j * 2 + 1].toFixed(2);
      cell.append(value, " ", deviation);
    });
  });
  document.getElementById("results-caption").textContent = (metric === "cvar" ? "CVaR₁₀" : "Minimum return") + " ↑ · manuscript results";
  document.getElementById("metric-settings").textContent = metric === "cvar"
    ? "RSDR β = −30 / −20 / −30 / −20; best EPOpt ε = 0.9 / 0.8 / 0.4 / 0.7, in task order. Hyperparameters are selected separately for each metric."
    : "RSDR β = −30 for all four tasks; best EPOpt ε = 0.9 / 0.8 / 0.4 / 0.8, in task order. Hyperparameters are selected separately for each metric.";
});

document.getElementById("copy-citation").addEventListener("click", async () => {
  const text = document.getElementById("bibtex").textContent;
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = "BibTeX copied.";
  } catch (_) {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById("bibtex"));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Citation selected. Press Ctrl+C (or ⌘C) to copy.";
  }
});
