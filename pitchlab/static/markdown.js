// Minimal, safe Markdown -> DOM (headings, paragraphs, ordered/unordered lists, **bold**, *italic*).
// Builds nodes with textContent only, so model output can never inject HTML.

function inline(parent, str) {
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;
  let last = 0;
  for (const m of str.matchAll(re)) {
    if (m.index > last) parent.append(document.createTextNode(str.slice(last, m.index)));
    const tok = m[0];
    const node = document.createElement(tok.startsWith("**") ? "strong" : "em");
    node.textContent = tok.replace(/^\*\*?|\*\*?$/g, "");
    parent.append(node);
    last = m.index + tok.length;
  }
  if (last < str.length) parent.append(document.createTextNode(str.slice(last)));
}

export function renderMarkdown(src) {
  const root = document.createElement("div");
  root.className = "md";
  let list = null;
  let para = null;
  const flush = () => { list = null; para = null; };
  for (const raw of String(src || "").split(/\r?\n/)) {
    const line = raw.trimEnd();
    if (!line.trim()) { flush(); continue; }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      flush();
      const level = Math.min(4, h[1].length + 1);
      const el = document.createElement(`h${level}`);
      inline(el, h[2]);
      root.append(el);
      continue;
    }
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (ul || ol) {
      para = null;
      const tag = ul ? "ul" : "ol";
      if (!list || list.tagName.toLowerCase() !== tag) {
        list = document.createElement(tag);
        root.append(list);
      }
      const li = document.createElement("li");
      inline(li, (ul ? ul[1] : ol[2]));
      list.append(li);
      continue;
    }
    if (list && /^\s{2,}\S/.test(raw)) {           // continuation of a list item
      const li = list.lastElementChild;
      li.append(document.createTextNode(" "));
      inline(li, line.trim());
      continue;
    }
    list = null;
    if (!para) {
      para = document.createElement("p");
      root.append(para);
    } else {
      para.append(document.createTextNode(" "));
    }
    inline(para, line.trim());
  }
  return root;
}
