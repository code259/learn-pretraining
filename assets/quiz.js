/* Shared exercise widgets. Include with <script src="../assets/quiz.js" defer></script>.
 *
 * Multiple choice:
 *   <div class="ex mc" data-answer="2">           (0-based index of the correct option)
 *     <p class="q">Question…</p>
 *     <div class="opts"><button>…</button>…</div>
 *     <div class="explain">Why…</div>
 *   </div>
 *
 * Numeric (accepts 2.35e20, 2.35×10^20, 2,352, 829 etc.):
 *   <div class="ex num" data-answer="2.352e20" data-tol="0.03" data-unit="FLOPs">
 *     <p class="q">Question…</p>
 *     <div class="explain">Worked solution…</div>
 *   </div>
 *   data-tol is relative tolerance (default 0.03 = ±3%).
 *
 * A <div class="score"></div> anywhere on the page shows first-try results.
 */
(function () {
  const results = new Map(); // element -> true (first try correct) | false

  function updateScore() {
    const el = document.querySelector(".score");
    if (!el) return;
    const total = document.querySelectorAll(".ex").length;
    const done = results.size;
    const right = [...results.values()].filter(Boolean).length;
    el.textContent = `First-try correct: ${right} / ${done} attempted (${total} total)`;
  }

  function record(ex, ok) {
    if (!results.has(ex)) results.set(ex, ok);
    updateScore();
  }

  function parseNum(s) {
    if (!s) return NaN;
    s = s.trim().toLowerCase()
      .replace(/,/g, "")
      .replace(/\s+/g, "")
      .replace(/(×|x|\*)10\^?/g, "e")
      .replace(/\^/g, "");
    // suffixes: k, m, b/g, t
    const m = s.match(/^([-+]?\d*\.?\d+(?:e[-+]?\d+)?)([kmbgt]?)$/);
    if (!m) return NaN;
    const mult = { "": 1, k: 1e3, m: 1e6, b: 1e9, g: 1e9, t: 1e12 }[m[2]];
    return parseFloat(m[1]) * mult;
  }

  function initMC(ex) {
    const answer = parseInt(ex.dataset.answer, 10);
    const buttons = [...ex.querySelectorAll(".opts button")];
    const fb = document.createElement("div");
    fb.className = "fb";
    ex.querySelector(".opts").after(fb);
    buttons.forEach((b, i) => {
      b.type = "button";
      b.addEventListener("click", () => {
        if (ex.classList.contains("done")) return;
        if (i === answer) {
          b.classList.add("correct");
          fb.className = "fb good";
          fb.textContent = "Correct.";
          ex.classList.add("done");
          buttons.forEach((x) => (x.disabled = true));
          record(ex, true);
        } else {
          b.classList.add("wrong");
          b.disabled = true;
          fb.className = "fb bad";
          fb.textContent = "Not quite. Think it through and try again.";
          record(ex, false);
        }
      });
    });
  }

  function initNum(ex) {
    const answer = parseFloat(ex.dataset.answer);
    const tol = parseFloat(ex.dataset.tol || "0.03");
    const unit = ex.dataset.unit || "";
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML =
      `<input type="text" inputmode="decimal" placeholder="e.g. 3.1e19" aria-label="answer">` +
      (unit ? `<span class="unit">${unit}</span>` : "") +
      `<button type="button" class="check">Check</button>` +
      `<button type="button" class="reveal">Show working</button>`;
    const fb = document.createElement("div");
    fb.className = "fb";
    ex.querySelector(".q").after(row, fb);
    const input = row.querySelector("input");
    let attempts = 0;

    function check() {
      if (ex.classList.contains("done")) return;
      const v = parseNum(input.value);
      if (isNaN(v)) {
        fb.className = "fb bad";
        fb.textContent = "Couldn't parse that. Try a form like 2.4e20 or 830.";
        return;
      }
      attempts++;
      const rel = Math.abs(v - answer) / Math.abs(answer);
      if (rel <= tol) {
        fb.className = "fb good";
        fb.textContent = "Correct.";
        ex.classList.add("done");
        input.disabled = true;
        record(ex, attempts === 1);
      } else {
        const ratio = v / answer;
        let hint = v > answer ? "Too high" : "Too low";
        const oom = Math.round(Math.log10(Math.abs(ratio)));
        if (Math.abs(oom) >= 1) hint += ` by roughly 10^${Math.abs(oom)}. Check your exponents`;
        else if (Math.abs(ratio - 2) < 0.1 || Math.abs(ratio - 0.5) < 0.05) hint += ` by about 2×. Check a factor of 2`;
        else if (Math.abs(ratio - 3) < 0.15 || Math.abs(ratio - 1 / 3) < 0.02) hint += ` by about 3×. Check the 6 vs 2`;
        fb.className = "fb bad";
        fb.textContent = hint + ".";
        record(ex, false);
      }
    }
    row.querySelector(".check").addEventListener("click", check);
    input.addEventListener("keydown", (e) => e.key === "Enter" && check());
    row.querySelector(".reveal").addEventListener("click", () => {
      record(ex, false);
      ex.classList.add("done");
    });
  }

  /* Sequence answer (e.g. position ids):
   *   <div class="ex seq" data-answer="0,1,2,3,4,5,3,4,5">
   *     <p class="q">…</p><div class="explain">…</div>
   *   </div>
   * Accepts commas and/or spaces, optional [ ]. Reports the first wrong index.
   */
  function initSeq(ex) {
    const answer = ex.dataset.answer.split(",").map((s) => s.trim());
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML =
      `<input type="text" placeholder="e.g. 0, 1, 2, …" aria-label="answer" style="width:100%;max-width:26rem">` +
      `<button type="button" class="check">Check</button>` +
      `<button type="button" class="reveal">Show working</button>`;
    const fb = document.createElement("div");
    fb.className = "fb";
    ex.querySelector(".q").after(row, fb);
    const input = row.querySelector("input");
    let attempts = 0;
    function check() {
      if (ex.classList.contains("done")) return;
      const got = input.value.replace(/[\[\]()]/g, "").split(/[\s,]+/).filter(Boolean);
      if (!got.length) return;
      attempts++;
      if (got.length !== answer.length) {
        fb.className = "fb bad";
        fb.textContent = `Expected ${answer.length} entries, got ${got.length}.`;
        record(ex, false);
        return;
      }
      const bad = got.findIndex((g, i) => g !== answer[i]);
      if (bad === -1) {
        fb.className = "fb good";
        fb.textContent = "Correct.";
        ex.classList.add("done");
        input.disabled = true;
        record(ex, attempts === 1);
      } else {
        fb.className = "fb bad";
        fb.textContent = `First mismatch at index ${bad}.`;
        record(ex, false);
      }
    }
    row.querySelector(".check").addEventListener("click", check);
    input.addEventListener("keydown", (e) => e.key === "Enter" && check());
    row.querySelector(".reveal").addEventListener("click", () => {
      record(ex, false);
      ex.classList.add("done");
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".ex.mc").forEach(initMC);
    document.querySelectorAll(".ex.num").forEach(initNum);
    document.querySelectorAll(".ex.seq").forEach(initSeq);
    updateScore();
  });
})();
