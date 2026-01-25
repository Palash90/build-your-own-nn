---
layout: default
---

<style>
    .github-corner-wrap {
        position: fixed;
        top: 0;
        right: 0;
        border: 0;
        z-index: 1001;
        /* Higher than the author bar */
    }

    /* Octocat animation */
    .github-corner:hover .octo-arm {
        animation: octocat-wave 560ms ease-in-out;
    }

    @keyframes octocat-wave {

        0%,
        100% {
            transform: rotate(0);
        }

        20%,
        60% {
            transform: rotate(-25deg);
        }

        40%,
        80% {
            transform: rotate(10deg);
        }
    }

    /* Mobile Adjustments */
    @media screen and (max-width: 1011px) {
        .github-corner-wrap svg {
            width: 60px;
            /* Smaller size on mobile */
            height: 60px;
        }
    }
</style>

<style>
    :root {
        --bg-color: #0d1117;
        --text-color: #c9d1d9;
        --link-color: #5187c5;
        --sidebar-bg: #161b22;
        --border-color: #30363d;
        --vscode-bg: #1e1e1e;
        --vscode-text: #d4d4d4;
        --toc-width: 20vw;
        /* Fixed width is safer than vw for overlap */
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
        body {
            margin-left: calc(var(--toc-width) + 60px) !important;
        }
    }

    /* Mobile: Reset margin so it doesn't look weird */
    @media screen and (max-width: 1011px) {
        body {
            margin-left: 0 !important;
        }
    }

    h1,
    h2,
    h3,
    h4,
    h5,
    strong {
        color: #becbda !important;
    }

    a {
        color: var(--link-color) !important;
        text-decoration: none;
    }
</style>

<style>
    /* --- 2. Navigation & Alerts --- */
    #markdown-toc {
        position: fixed;
        top: 10px;
        left: 20px;
        width: var(--toc-width);
        max-height: 85vh;
        overflow-y: auto;
        font-size: 0.85em;
        list-style-type: none;
        padding: 15px;
        border-left: 2px solid var(--border-color);
        background: var(--sidebar-bg);
        z-index: 100;
        border-radius: 5px;
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
        /* background: #1c2128 !important; */
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
            /* STOP it from floating */
            position: relative !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;

            /* Layout */
            display: block !important;
            box-sizing: border-box;
            margin: 20px 0 !important;
            padding: 15px !important;

            /* Visuals */
            background: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 8px;
            z-index: 1;
            /* Ensure it doesn't overlap headers */
        }

        /* Clear any floats that might be lingering */
        #markdown-toc::after {
            content: "";
            display: table;
            clear: both;
        }
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
        background-color: rgba(56, 139, 253, 0.1) !important;
        /* GitHub's blue tint */
        border-left: 0.25em solid #85a1c7 !important;
        /* GitHub's blue border */
        border-radius: 6px !important;
        color: #c9d1d9 !important;
    }

    /* 2. Target the "NOTE" text specifically */
    blockquote strong {
        color: #58a6ff !important;
        /* The cyan/blue from your screenshot */
        font-size: 0.9em !important;
        letter-spacing: 0.5px !important;
        display: inline-flex !important;
        align-items: center !important;
        margin-bottom: 8px !important;
        text-transform: uppercase;
    }

    /* 3. Optional: Add the GitHub info icon */
    blockquote strong::before {
        content: "ⓘ";
        margin-right: 8px;
        font-size: 1.1em;
    }

    /* Note - Info Icon */
    .bq-note strong::before {
        content: "ⓘ";
    }

    /* Tip - Lightbulb Icon */
    .bq-tip strong::before {
        content: "💡";
    }

    /* Warning - Warning Triangle Icon */
    .bq-warning strong::before {
        content: "⚠️";
    }

    /* Checkpoint - Checkmark or Trophy Icon */
    .bq-checkpoint strong::before {
        content: "🏁";
        /* Or use "✅" */
    }

    /* Specific Icon for Caution */
    .bq-caution strong::before {
        content: "🚫";
        /* You can also use "❗" or "⚠️" */
    }

    /* Note/Blue (Default) */
    .bq-note {
        background-color: rgba(56, 139, 253, 0.1) !important;
        border-left: 0.25em solid #1f6feb !important;
    }

    .bq-note strong {
        color: #58a6ff !important;
    }

    /* Tip/Green */
    .bq-tip {
        background-color: rgba(63, 185, 80, 0.1) !important;
        border-left: 0.25em solid #238636 !important;
    }

    .bq-tip strong {
        color: #3fb950 !important;
    }

    /* Warning/Orange */
    .bq-warning {
        background-color: rgba(210, 153, 34, 0.1) !important;
        border-left: 0.25em solid #d29922 !important;
    }

    .bq-warning strong {
        color: #d29922 !important;
    }

    .bq-checkpoint {
        background-color: rgba(138, 18, 218, 0.34) !important;
        border-left: 0.25em solid #a11edd !important;
    }

    .bq-checkpoint strong {
        color: #772a9b !important;
    }

    /* Caution/Red Alert */
    .bq-caution {
        background-color: rgba(248, 81, 73, 0.1) !important;
        /* GitHub's red tint */
        border-left: 0.25em solid #da3633 !important;
        /* GitHub's red border */
    }

    .bq-caution strong {
        color: #f85149 !important;
        /* Bold red text */
    }
</style>

<style>
    .highlight,
    pre.highlight {
        position: relative !important;
        display: flex !important;
        flex-direction: column;
        background: #0d1117;
        border: 1px solid #30363d;
        background-color: var(--vscode-bg) !important;
        color: var(--vscode-text) !important;
        padding: 2px;
        border-radius: 6px;
        border: 1px solid var(--border-color);
        overflow-x: auto;
    }

    .highlight pre {
        margin: 0 !important;
        padding: 16px !important;
        overflow-x: auto !important;
        /* This allows code to scroll */
        white-space: pre !important;
    }

    .highlight span {
        background-color: transparent !important;
    }

    /* VS Code Colors */
    .highlight .k,
    .highlight .kd {
        color: #569cd6 !important;
    }

    /* keywords: fn, let, impl, pub */
    .highlight .nc,
    .highlight .nn,
    .highlight .kt,
    .highlight .nb {
        color: #4ec9b0 !important;
    }

    /* Structs, Types, Vec */
    .highlight .nf {
        color: #dcdcaa !important;
    }

    /* Functions */
    .highlight .s,
    .highlight .sc {
        color: #ce9178 !important;
    }

    /* Strings and Chars */
    .highlight .mi,
    .highlight .mf,
    .highlight .mh {
        color: #b5cea8 !important;
    }

    /* Numbers */
    .highlight .c1,
    .highlight .cm {
        color: #6a9955 !important;
        font-style: italic;
    }

    /* Comments */
    .highlight .o,
    .highlight .p {
        color: #d4d4d4 !important;
    }

    /* Operators (+, *) and Punctuation ([, <, >) */
    .highlight .nt {
        color: #569cd6 !important;
    }

    /* Generic brackets < > */
    .highlight .vi,
    .highlight .vc,
    .highlight .nv {
        color: #9cdcfe !important;
    }

    /* Variables and self */
    .highlight .nd {
        color: #c586c0 !important;
    }

    /* Macros and Attributes #[...] */

    /* Force code blocks to be independent "islands" */
    .highlight {
        display: block !important;
        background-color: #191a1d !important;
        unicode-bidi: embed;
        white-space: pre;
        /* Keeps code structure */
        margin-bottom: 24px !important;
        border-radius: 6px;
        overflow: hidden;
    }

    /* Reset the internal pre so it doesn't double-wrap */
    .highlight pre {
        margin: 0 !important;
        padding: 16px !important;
        background-color: #0d1117 !important;
        overflow-x: auto !important;
        /* Horizontal scroll */
        background-color: transparent !important;
        /* Uses container color */
        white-space: pre !important;
        word-wrap: normal !important;
    }
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
        position: absolute !important;
        top: 8px;
        right: 8px;
        z-index: 20;
        /* Keeps it above scrolling code */
        padding: 4px 8px;
        font-size: 12px;
        color: #c9d1d9;
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        cursor: pointer;
        display: flex;
        opacity: 0;
        /* Hidden by default */
        transition: opacity 0.2s, background-color 0.2s;
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

<style>
    .page-header {
        max-width: 900px;
        margin-bottom: 40px;
    }

    .page-title {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 10px;
    }

    .page-subtitle {
        font-size: 1rem;
        color: #86c1ff;
        font-weight: 400;
        text-align: right;
        font-style: oblique;
        margin-bottom: 100px;
    }
</style>

<style>
    /* Base style: Mobile First (Horizontal Bar) */
.sticky-author-container {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 40px;
    background: var(--sidebar-bg);
    border-top: 1px solid var(--border-color);
    z-index: 1000;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.3);
    display: flex;
    align-items: center;
}

.sticky-author-container a {
    display: flex !important;
    flex-direction: row !important;
    align-items: center;
    width: 100%;
    padding: 0 15px;
    gap: 10px;
    text-decoration: none;
}

    .sticky-author-image {
        width: 25px;
        height: 25px;
        border-radius: 50%;
        border: 1.5px solid var(--link-color);
        flex-shrink: 0;
    }

.sticky-author-name {
    font-size: 0.9em;
    font-weight: bold;
    color: var(--text-color);
    margin: 0;
    /* Mobile: prevent wrapping to keep bar slim */
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Desktop override: Sidebar Card (Vertical Stack) */
@media screen and (min-width: 1012px) {
    .sticky-author-container {
        top: calc(85vh + 20px); /* Position below TOC */
        left: 20px;
        bottom: auto;
        width: var(--toc-width);
        height: auto; /* Let it expand vertically */
        background: var(--sidebar-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 15px 0;
    }

    .sticky-author-container a {
        flex-direction: column !important; /* Stack image over name */
        gap: 8px;
        text-align: center;
    }

    .sticky-author-image {
        width: 50px;
        height: 50px;
    }

    .sticky-author-name {
        white-space: normal; /* Allow name to wrap on desktop if long */
        overflow: visible;
        font-size: 0.95em;
        line-height: 1.2;
    }
}
</style>

<style>
    .author-card {
        display: flex;
        align-items: center;
        gap: 20px;
        background: var(--sidebar-bg);
        /* Uses your existing dark gray */
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 24px;
        margin: 40px 0;
    }

    .author-image {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: 2px solid var(--link-color);
        object-fit: cover;
    }

    .author-info {
        flex: 1;
    }

    .author-name {
        color: var(--link-color) !important;
        font-size: 1.2em;
        font-weight: bold;
        margin: 0 0 8px 0 !important;
    }

    .author-bio {
        font-size: 0.95em;
        line-height: 1.5;
        color: #8b949e;
        margin-bottom: 12px;
    }

    .author-socials a {
        margin-right: 15px;
        font-size: 0.85em;
        opacity: 0.8;
        transition: opacity 0.2s;
    }

    .author-socials a:hover {
        opacity: 1;
        text-decoration: underline;
    }

    /* Mobile responsive */
    @media screen and (max-width: 600px) {
        .author-card {
            flex-direction: column;
            text-align: center;
        }
    }
</style>


<script>
    window.MathJax = {
        loader: { load: ['[tex]/color'] },
        tex: {
            packages: { '[+]': ['color'] },
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
<script src='https://cdn.jsdelivr.net/npm/plotly.js-dist@3.3.1/plotly.min.js'></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>



<header class="page-header">

    <p class="page-subtitle">{{ site.subtitle }}</p>
</header>

* TOC
{:toc}

<div class="sticky-author-container">
    <a href="https://github.com/{{ site.github_username }}">
    <div>
        <img src="https://github.com/{{ site.github_username }}.png" alt="Author" class="sticky-author-image">
        </div>
        <div style="flex-grow: 1;">
            <p class="sticky-author-name">{{ site.author_name }}</p>
           
        </div>
    </a>
</div>

{% include_relative README.md %}

<script>
    document.querySelectorAll('blockquote').forEach(bq => {
        // Get the first strong tag or the beginning of the text
        const header = bq.querySelector('strong');
        if (!header) return;

        const type = header.innerText.toUpperCase();

        if (type.includes('NOTE')) {
            bq.classList.add('bq-note');
        } else if (type.includes('TIP')) {
            bq.classList.add('bq-tip');
        } else if (type.includes('WARNING') || type.includes('IMPORTANT')) {
            bq.classList.add('bq-warning');
        } else if (type.includes('CHECKPOINT') || type.includes('IMPORTANT')) {
            bq.classList.add('bq-checkpoint');
        } else if (type.includes('CAUTION') || type.includes('IMPORTANT')) {
            bq.classList.add('bq-caution');
        }
    });
</script>

<script>
    document.querySelectorAll('pre.highlight').forEach((codeBlock) => {
        // 1. Ensure the parent container is relative so the button sticks to it
        codeBlock.style.position = 'relative';

        // 2. Create the button
        const button = document.createElement('button');
        button.className = 'copy-btn';
        button.type = 'button';
        button.innerText = 'Copy';

        // 3. Logic to grab ONLY the text inside this specific block
        button.addEventListener('click', () => {
            const code = codeBlock.innerText.replace('Copy', '').trim();
            navigator.clipboard.writeText(code).then(() => {
                button.innerText = 'Copied!';
                button.classList.add('copied');
                setTimeout(() => {
                    button.innerText = 'Copy';
                    button.classList.remove('copied');
                }, 2000);
            });
        });

        codeBlock.appendChild(button);
    });
</script>

<script type="module">
    document.addEventListener("DOMContentLoaded", async () => {
        const blocks = document.querySelectorAll('.language-pbm');

        for (const block of blocks) {
            const rows = block.textContent.trim().split('\n');
            const container = document.createElement('div');
            container.style.display = 'flex';
            container.style.flexDirection = 'column';
            container.style.gap = '20px';
            container.style.margin = '20px 0';

            for (const row of rows) {
                const rowDiv = document.createElement('div');
                rowDiv.style.display = 'flex';
                rowDiv.style.alignItems = 'center'; // Vertically align to bottom
                rowDiv.style.gap = '15px';
                rowDiv.style.flexWrap = 'wrap';

                const fileNames = row.split(',').map(s => s.trim());

                for (const imgDetails of fileNames) {
                    if (!imgDetails) continue;

                    let parts = imgDetails.split(':');

                    let fileName = parts[0]?.trim() || "";
                    let caption = parts[1]?.trim() || "";

                    const canvas = await createPBMCanvas(fileName);
                    if (canvas) {
                        const figure = document.createElement('div');
                        figure.style.textAlign = 'center';

                        // Optional: Add filename label under image
                        const label = document.createElement('div');
                        label.textContent = caption;
                        label.style.fontSize = '10px';
                        label.style.color = '#8b949e';
                        label.style.marginTop = '5px';

                        figure.appendChild(canvas);
                        figure.appendChild(label);
                        rowDiv.appendChild(figure);
                    }
                }
                container.appendChild(rowDiv);
            }
            block.parentElement.replaceWith(container);
        }
    });

    async function createPBMCanvas(url) {
        try {
            const response = await fetch(url);
            const text = await response.text();
            const tokens = text.split(/\s+/);
            if (tokens[0] !== 'P1') return null;

            const w = parseInt(tokens[1]);
            const h = parseInt(tokens[2]);
            const pixels = tokens.slice(3);

            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            canvas.style.width = (w * 2) + "px"; // Scale up for visibility
            canvas.style.imageRendering = 'pixelated';
            canvas.style.border = '1px solid #30363d';

            const ctx = canvas.getContext('2d');
            const imgData = ctx.createImageData(w, h);

            for (let i = 0; i < pixels.length; i++) {
                const idx = i * 4;
                const isBlack = pixels[i] === '1';
                // RGB: Blue for 1s (#58a6ff), Dark for 0s (#0d1117)
                imgData.data[idx] = isBlack ? 88 : 13;
                imgData.data[idx + 1] = isBlack ? 166 : 17;
                imgData.data[idx + 2] = isBlack ? 255 : 23;
                imgData.data[idx + 3] = 255;
            }
            ctx.putImageData(imgData, 0, 0);
            return canvas;
        } catch (e) {
            console.error("Error loading PBM:", url, e);
            return null;
        }
    }


</script>

<script type="module">
    var layout = {
        colorway: [
            '#3c93f0',
            '#7ad369',
            '#E28E8E',
            '#D4A373',
            '#A3B18A',
            '#778DA9'
        ],
        font: { size: 18, color: '#DCDCDC' },
        paper_bgcolor: 'rgba(64, 64, 128, 0.05)',
        plot_bgcolor: 'rgba(64, 64, 128, 0.05)',
        modebar: {
            bgcolor: 'rgba(64, 64, 128, 0.05)'
        },
        autosize: true,
        xaxis: {
            automargin: true,
            gridcolor: '#333333', // Darker, subtle grid lines
            zerolinecolor: '#444444'
        },
        yaxis: {
            automargin: true,
            gridcolor: '#333333',
            zerolinecolor: '#444444'
        },
        margin: { t: 60, b: 60, l: 60, r: 60 }
    };

    var config = { responsive: true, staticPlot: true }

    const blocks = document.querySelectorAll('.language-plotly');

    blocks.forEach((block) => {
        let data;
        try {
            data = JSON.parse(block.textContent);
        } catch (e) {
            console.error("Invalid JSON in plotly block", e);
            return;
        }
        block.textContent = '';

        var title = data.title;
        layout = { ...layout, title: { text: title } };

        var traces = data.traces.map(t => {
            if (t.type === 'scatter') {
                return {
                    ...t,
                    mode: 'markers',
                    type: 'scatter',
                    marker: { size: 4, symbol: "square", ...t.marker }
                };
            }
            if (t.type === 'line') {
                return {
                    ...t,
                    mode: 'lines+markers',
                    type: 'scatter',
                    marker: { line: { width: 2.5 }, ...t.marker }
                };
            }
            if (t.type === 'bar') {
                return {
                    ...t,
                    width: t.width !== undefined ? t.width : 0.2,
                    marker: { ...t.marker }
                };
            }
            return { ...t }; // Return the original trace if it's not a line or scatter
        });

        if (data.removeGrid) {
            layout = { ...layout, xaxis: { showGrid: false, visible: false }, yaxis: { showGrid: false, visible: false } }
        }

        const plotDiv = document.createElement('div');
        plotDiv.className = 'plotly';
        plotDiv.style.width = '100%';
        plotDiv.style.height = '600px';
        const targetToReplace = block.parentElement;
        targetToReplace.replaceWith(plotDiv);

        Plotly.newPlot(plotDiv, traces, layout, config)
        //targetToReplace.replaceWith(plotDiv);

    });
</script>


<div class="github-corner-wrap">
    <a href="https://github.com/{{ site.github_username }}/{{ site.github_repo }}" class="github-corner"
        aria-label="View source on GitHub">
        <svg width="80" height="80" viewBox="0 0 250 250" style="fill:var(--link-color); color:var(--bg-color);"
            aria-hidden="true">
            <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
            <path
                d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2"
                fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path>
            <path
                d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.3 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z"
                fill="currentColor" class="octo-body"></path>
        </svg>
    </a>
</div>

<div class="author-card">
    <img src="https://github.com/{{ site.github_username }}.png" alt="Author" class="author-image">

    <div class="author-info">
        <h3 class="author-name">Written by {{ site.author_name }}</h3>
        <p class="author-bio">
            {{ site.author_bio }}
        </p>
        <div class="author-socials">
            <a href="https://linkedin.com/in/{{ site.linkedin_username }}">LinkedIn</a>
            <a href="https://dev.to/{{ site.devto_username }}">DEV.to</a>
            <a href="https://github.com/{{ site.github_username }}">GitHub</a>
        </div>
    </div>
</div>

<section style="margin-top: 50px; border-top: 1px solid var(--border-color); padding-top: 20px;">
    <script src="https://giscus.app/client.js" data-repo="palash90/build-your-own-nn" data-repo-id={{site.data_repo_id}}
        data-category="Announcements" data-category-id={{site.category_id}} data-mapping="pathname" data-strict="0"
        data-reactions-enabled="1" data-emit-metadata="0" data-input-position="bottom" data-theme="transparent_dark"
        data-lang="en" crossorigin="anonymous" async>
        </script>
</section>