---
layout: default
---

<style>
/* 1. Permanent Dark Theme Variables */
:root {
    --bg-color: #0d1117; /* GitHub Dark Background */
    --text-color: #c9d1d9;
    --link-color: #58a6ff;
    --sidebar-bg: #161b22;
    --border-color: #30363d;
    --vscode-bg: #1e1e1e;
    --vscode-text: #d4d4d4;
}

/* 2. Global Styling */
body { 
    background-color: var(--bg-color) !important; 
    color: var(--text-color) !important; 
}
h1, h2, h3, h4, h5, strong { color: #f0f6fc !important; }
a { color: var(--link-color) !important; text-decoration: none; }
a:hover { text-decoration: underline; }

/* 3. Sticky Sidebar */
#markdown-toc {
    position: fixed;
    top: 50px;
    left: 20px;
    width: 220px;
    max-height: 80vh;
    overflow-y: auto;
    font-size: 0.85em;
    list-style-type: none;
    padding: 15px;
    border-left: 2px solid var(--border-color);
    background: var(--sidebar-bg);
}

/* 4. VS Code Code Blocks */
.highlight, pre.highlight {
    background-color: var(--vscode-bg) !important;
    color: var(--vscode-text) !important;
    padding: 16px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
    overflow-x: auto;
}
.highlight span { background-color: transparent !important; }
.highlight .k { color: #569cd6 !important; } /* Keywords */
.highlight .nc { color: #4ec9b0 !important; } /* Types */
.highlight .nf { color: #dcdcaa !important; } /* Functions */
.highlight .s { color: #ce9178 !important; }  /* Strings */
.highlight .mi { color: #b5cea8 !important; } /* Numbers */
.highlight .c1 { color: #6a9955 !important; font-style: italic; } /* Comments */

/* 5. Modern GitHub-style Alerts */
blockquote {
    padding: 0.5em 1em;
    color: #8b949e;
    border-left: 0.25em solid var(--border-color);
    background: rgba(48, 54, 61, 0.2);
    margin: 1.5em 0;
}
/* Style for NOTE/TIP specifically */
blockquote p strong {
    display: flex;
    align-items: center;
    gap: 5px;
}
</style>

<script>
  window.MathJax = {
    loader: {load: ['[tex]/color']},
    tex: {
      packages: {'[+]': ['color']},
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Table of Contents
* TOC
{:toc}

---

{% include_relative README.md %}