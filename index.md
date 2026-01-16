---
layout: default
---

<style>
/* 1. Define Color Variables */
:root {
    --bg-color: #ffffff;
    --text-color: #2b2b2b;
    --link-color: #007bff;
    --sidebar-bg: #ffffff;
    --border-color: #eee;
    --code-bg: #f8f8f8;
}

[data-theme="dark"] {
    --bg-color: #1a1a1a;
    --text-color: #e0e0e0;
    --link-color: #4db3ff;
    --sidebar-bg: #252525;
    --border-color: #444;
    --code-bg: #2d2d2d;
}

/* 2. Apply Variables to the Theme */
body { 
    background-color: var(--bg-color) !important; 
    color: var(--text-color) !important; 
}
h1, h2, h3, h4, h5, strong { color: var(--text-color) !important; }
a { color: var(--link-color) !important; }
code { background-color: var(--code-bg) !important; color: #ff79c6; }
pre { background-color: var(--code-bg) !important; border: 1px solid var(--border-color); }

/* 3. Sticky Sidebar Styling */
#markdown-toc {
    position: fixed;
    top: 80px;
    left: 20px;
    width: 220px;
    max-height: 70vh;
    overflow-y: auto;
    font-size: 0.85em;
    list-style-type: none;
    padding: 15px;
    border-left: 2px solid var(--border-color);
    background: var(--sidebar-bg);
}

/* 4. The Toggle Button */
#theme-toggle {
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 1000;
    padding: 8px 12px;
    cursor: pointer;
    background: var(--sidebar-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 5px;
}

@media screen and (max-width: 1100px) {
    #markdown-toc { position: relative; width: auto; top: 0; left: 0; }
    #theme-toggle { position: relative; top: 0; left: 0; margin-bottom: 10px; }
}
</style>

<button id="theme-toggle">🌙 Dark Mode</button>

<script>
  // MathJax Config
  window.MathJax = {
    tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }
  };

  // Theme Switching Logic
  const toggleBtn = document.getElementById('theme-toggle');
  const currentTheme = localStorage.getItem('theme');

  if (currentTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    toggleBtn.textContent = "☀️ Light Mode";
  }

  toggleBtn.addEventListener('click', () => {
    let theme = document.documentElement.getAttribute('data-theme');
    if (theme === 'dark') {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('theme', 'light');
        toggleBtn.textContent = "🌙 Dark Mode";
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
        toggleBtn.textContent = "☀️ Light Mode";
    }
  });
</script>

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Table of Contents
* TOC
{:toc}

---

{% include_relative README.md %}