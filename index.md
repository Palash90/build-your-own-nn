---
layout: default
---

<style>
:root {
    --bg-color: #0d1117;
    --text-color: #c9d1d9;
    --link-color: #58a6ff;
    --sidebar-bg: #161b22;
    --border-color: #30363d;
    --vscode-bg: #1e1e1e;
    --vscode-text: #d4d4d4;
    --toc-width: 20vw; /* Fixed width is safer than vw for overlap */
}

/* Base styling */
body { 
    background-color: var(--bg-color) !important; 
    color: var(--text-color) !important; 
    padding: 20px !important;
    transition: margin-left 0.3s ease;
}

/* Desktop: Push content to the right */
@media screen and (min-width: 1012px) {
    body { margin-left: calc(var(--toc-width) + 60px) !important; }
}

/* Mobile: Reset margin so it doesn't look weird */
@media screen and (max-width: 1011px) {
    body { margin-left: 0 !important; }
}

h1, h2, h3, h4, h5, strong { color: #51cbdb !important; }
a { color: var(--link-color) !important; text-decoration: none; }
</style>
<style>
/* --- 2. Navigation & Alerts --- */
#markdown-toc {
    position: fixed;
    top: 50px;
    left: 20px;
    width: var(--toc-width);
    max-height: 80vh;
    overflow-y: auto;
    font-size: 0.85em;
    list-style-type: none;
    padding: 15px;
    border-left: 2px solid var(--border-color);
    background: var(--sidebar-bg);
    z-index: 100;
}
#markdown-toc li:hover {
    background: #30363d;
    border-color: #8b949e;
    cursor: pointer;
}

#markdown-toc li {
    background: #161b22 !important;
    border: 0px solid #30363d !important;
    padding: 4px 12px !important;
    border-radius: 2px !important;
    transition: all 0.2s ease !important;
}

#markdown-toc li:hover {
    border-color: #58a6ff !important;
    background: #1c2128 !important;
}

#markdown-toc li a {
    color: #8b949e !important;
    font-size: 12px !important;
}

#markdown-toc li a:hover {
    color: #58a6ff !important;
    text-decoration: none !important;
}

/* Mobile Breadcrumb Style */
@media screen and (max-width: 1011px) {
    #markdown-toc {
        position: relative;
        width: auto;
        left: 0;
        top: 0;
        display: flex; /* Makes it horizontal */
        flex-wrap: wrap;
        gap: 10px;
        padding: 10px;
        border-left: none;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 20px;
        background: transparent;
    }
    #markdown-toc li { display: inline; }
    #markdown-toc li::after { content: " / "; margin-left: 10px; color: #666; }
    #markdown-toc li:last-child::after { content: ""; }
}

/* Hide only level 5 and deeper in the TOC */
#markdown-toc ul ul ul ul {
    display: none !important;
}

/* Optional: Desktop indentation for hierarchy */
@media screen and (min-width: 1012px) {
    #markdown-toc ul {
        list-style-type: none !important;
        padding-left: 1.2em !important;
        margin-top: 5px;
    }
}

/* MathJax Wrapping Fix */
.MathJax {
    white-space: normal;
    overflow-x: auto;
    overflow-y: hidden;
    max-width: 100%;
}
.mjx-container {
    display: inline-grid !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    white-space: normal !important;
}

/* Optional: Subtle scrollbar for wide matrices that cannot be broken */
.mjx-container::-webkit-scrollbar {
    height: 4px;
}
.mjx-container::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 10px;
}

blockquote {
    padding: 0.5em 1em;
    color: #8b949e;
    border-left: 0.25em solid var(--border-color);
    background: rgba(48, 54, 61, 0.2);
    margin: 1.5em 0;
}


</style>


<style>
/* 1. The GitHub Alert Box */
blockquote {
    padding: 0.8em 1em !important;
    margin: 1.5em 0 !important;
    background-color: rgba(56, 139, 253, 0.1) !important; /* GitHub's blue tint */
    border-left: 0.25em solid #1f6feb !important;        /* GitHub's blue border */
    border-radius: 6px !important;
    color: #c9d1d9 !important;
}

/* 2. Target the "NOTE" text specifically */
blockquote strong {
    color: #58a6ff !important; /* The cyan/blue from your screenshot */
    font-size: 0.9em !important;
    letter-spacing: 0.5px !important;
    display: inline-flex !important;
    align-items: center !important;
}

/* 3. Optional: Add the GitHub info icon */
blockquote strong::before {
    content: "ⓘ"; 
    margin-right: 8px;
    font-size: 1.1em;
}
</style>

<style>
.highlight, pre.highlight {
    background-color: var(--vscode-bg) !important;
    color: var(--vscode-text) !important;
    padding: 16px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
    overflow-x: auto;
}

.highlight span { background-color: transparent !important; }

/* VS Code Colors */
.highlight .k, .highlight .kd { color: #569cd6 !important; } /* keywords: fn, let, impl, pub */
.highlight .nc, .highlight .nn, .highlight .kt, .highlight .nb { color: #4ec9b0 !important; } /* Structs, Types, Vec */
.highlight .nf { color: #dcdcaa !important; } /* Functions */
.highlight .s, .highlight .sc { color: #ce9178 !important; } /* Strings and Chars */
.highlight .mi, .highlight .mf, .highlight .mh { color: #b5cea8 !important; } /* Numbers */
.highlight .c1, .highlight .cm { color: #6a9955 !important; font-style: italic; } /* Comments */
.highlight .o, .highlight .p { color: #d4d4d4 !important; } /* Operators (+, *) and Punctuation ([, <, >) */
.highlight .nt { color: #569cd6 !important; } /* Generic brackets < > */
.highlight .vi, .highlight .vc, .highlight .nv { color: #9cdcfe !important; } /* Variables and self */
.highlight .nd { color: #c586c0 !important; } /* Macros and Attributes #[...] */
</style>

<style>
    .giscus {
    max-width: 800px;
    margin-top: 40px;
    /* On desktop, this ensures it aligns with your pushed-right content */
}

@media screen and (min-width: 1012px) {
    .giscus-frame {
        margin-left: 0; 
    }
}
</style>
<style>
/* Container for the code block and button */
.highlight {
    position: relative;
}

/* The Copy Button */
.copy-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    padding: 4px 8px;
    font-size: 12px;
    color: #c9d1d9;
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    cursor: pointer;
    opacity: 0; /* Hidden by default */
    transition: opacity 0.2s, background-color 0.2s;
    z-index: 10;
}

/* Show button on hover */
.highlight:hover .copy-btn {
    opacity: 1;
}

.copy-btn:hover {
    background-color: #30363d;
    border-color: #8b949e;
}

.copy-btn.copied {
    color: #3fb950;
    border-color: #238636;
}
</style>

<script>
  <script>
  window.MathJax = {
    loader: {load: ['[tex]/color']},
    tex: {
      packages: {'[+]': ['color']},
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    },
    // This tells MathJax to break long lines automatically
    chtml: {
        displayAlign: 'left',
        linebreaks: { 
            allow: true, 
            width: 'container' 
        }
    },
    svg: {
        linebreaks: { 
            allow: true, 
            width: 'container' 
        }
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

* TOC
{:toc}

{% include_relative README.md %}

<script>
document.querySelectorAll('blockquote').forEach(bq => {
  const text = bq.innerText.toUpperCase();
  if (text.includes('NOTE')) {
    bq.style.backgroundColor = 'rgba(56, 139, 253, 0.1)';
    bq.style.borderLeftColor = '#1f6feb';
    bq.querySelector('strong').style.color = '#58a6ff';
  } else if (text.includes('TIP')) {
    bq.style.backgroundColor = 'rgba(63, 185, 80, 0.1)';
    bq.style.borderLeftColor = '#238636';
    bq.querySelector('strong').style.color = '#3fb950';
  }
});
</script>

<script>
document.querySelectorAll('.highlight').forEach((codeBlock) => {
    // Create the button
    const button = document.createElement('button');
    button.className = 'copy-btn';
    button.type = 'button';
    button.innerText = 'Copy';

    // Add click event
    button.addEventListener('click', () => {
        const code = codeBlock.querySelector('code').innerText;
        navigator.clipboard.writeText(code).then(() => {
            button.innerText = 'Copied!';
            button.classList.add('copied');

            // Reset after 2 seconds
            setTimeout(() => {
                button.innerText = 'Copy';
                button.classList.remove('copied');
            }, 2000);
        });
    });

    codeBlock.appendChild(button);
});
</script>

<section style="margin-top: 50px; border-top: 1px solid var(--border-color); padding-top: 20px;">
  <script src="https://giscus.app/client.js"
        data-repo="palash90/build-your-own-nn"
        data-repo-id="R_kgDOQ42YxA"
        data-category="[ENTER CATEGORY NAME HERE]"
        data-category-id="[ENTER CATEGORY ID HERE]"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="transparent_dark"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script>
</section>