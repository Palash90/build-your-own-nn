---
layout: default
---
<style>
/* Sidebar Container */
#markdown-toc {
    position: fixed;
    top: 50px;
    left: 20px;
    width: 200px;
    max-height: 80vh;
    overflow-y: auto;
    font-size: 0.85em;
    list-style-type: none;
    padding-left: 10px;
    border-left: 2px solid #eee;
    background: white;
}

/* Hide the 'Table of Contents' header from the TOC itself */
#markdown-toc li a[href="#table-of-contents"] {
    display: none;
}

/* Responsive: Hide sidebar on small screens to avoid overlapping text */
@media screen and (max-width: 1100px) {
    #markdown-toc {
        position: relative;
        width: auto;
        top: 0;
        left: 0;
        border: 1px solid #eee;
        padding: 20px;
        margin-bottom: 20px;
    }
}
</style>

<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    }
  };
</script>

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Table of Contents
* TOC
{:toc}

---

{% include_relative README.md %}