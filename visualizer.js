const MAX_LAYERS = 6;
const MAX_NEURONS = 8;
const ACTIVATIONS = ['linear', 'relu', 'leakyRelu', 'sigmoid', 'softmax', 'tanh'];
const LAYER_COLORS = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#f1c40f', '#e74c3c'];

let topology = [
    { count: 3, activation: 'linear' },
    { count: 4, activation: 'relu' },
    { count: 2, activation: 'softmax' }
];

let lastInspected = null; // Store {layerIdx, neuronIdx}

function syncAndDraw() {
    const counts = topology.map(t => t.count);
    const acts = topology.slice(1).map(t => t.activation);

    // Create NN instance from nn.js
    // Note: your nn.js uses this.levels to store Layer objects
    window.network = new NeuralNetwork(counts, acts, 'mse');

    renderUI();
    drawNetwork();

    // REFRESH DATA TABLE: Clear existing rows and update headers to match new counts
    tableBody.innerHTML = '';
    updateTableHeaders();
    updateDataCount();

    // AUTO-REFRESH INSPECTOR: If the user was looking at a neuron, update its data
    if (lastInspected) {
        inspectNeuron(lastInspected.layerIdx, lastInspected.neuronIdx);
    }
}

/**
 * Updates the table headers based on current Input/Output counts.
 * Uses the colors from LAYER_COLORS for visual consistency.
 */
function updateTableHeaders() {
    const inputLen = topology[0].count;
    const outputLen = topology[topology.length - 1].count;

    // In0 is blue (#3498db), OutX is red (#e74c3c)
    let html = '<tr>';
    for (let i = 0; i < inputLen; i++) {
        html += `<th style="padding:10px; color:${LAYER_COLORS[0]}; border-bottom:2px solid #333;">In ${i}</th>`;
    }
    for (let i = 0; i < outputLen; i++) {
        const outColor = LAYER_COLORS[(topology.length - 1) % LAYER_COLORS.length];
        html += `<th style="padding:10px; color:${outColor}; border-bottom:2px solid #333;">Out ${i}</th>`;
    }
    html += '<th style="width:40px; border-bottom:2px solid #333;"></th></tr>';

    tableHead.innerHTML = html;
}

function renderUI() {
    const row = document.getElementById('topology-row');
    row.innerHTML = '';
    topology.forEach((layer, idx) => {
        const isInput = idx === 0;
        const isOutput = idx === topology.length - 1;
        const layerDiv = document.createElement('div');
        layerDiv.className = 'layer-ctrl';
        layerDiv.style.borderTopColor = LAYER_COLORS[idx % LAYER_COLORS.length];
        layerDiv.innerHTML = `
            <span style="font-size: 0.7em;">${isInput ? 'INPUT' : isOutput ? 'OUTPUT' : 'HIDDEN'}</span>
            <button onclick="changeNeuron(${idx}, 1)" ${layer.count >= MAX_NEURONS ? 'disabled' : ''}>+</button>
            <div class="neuron-count">${layer.count}</div>
            <button onclick="changeNeuron(${idx}, -1)" ${layer.count <= 1 ? 'disabled' : ''}>-</button>
            <select onchange="changeActivation(${idx}, this.value)">
                ${ACTIVATIONS.map(act => `<option value="${act}" ${layer.activation === act ? 'selected' : ''}>${act}</option>`).join('')}
            </select>
            ${(!isInput && !isOutput) ? `<button style="margin-top:10px; color:#ff4444;" onclick="removeLayer(${idx})">×</button>` : ''}
        `;
        row.appendChild(layerDiv);
    });
}

function drawNetwork() {
    const svg = document.getElementById('network-svg');
    if (!svg) return;
    svg.innerHTML = '';
    const width = svg.clientWidth;
    const height = svg.clientHeight;
    const layerSpacing = width / (topology.length + 1);
    let layerCoords = [];

    topology.forEach((layer, i) => {
        const x = (i + 1) * layerSpacing;
        const neurons = [];
        const vSpacing = height / (layer.count + 1);
        for (let j = 0; j < layer.count; j++) {
            neurons.push({ x, y: (j + 1) * vSpacing });
        }
        layerCoords.push(neurons);
    });

    // Draw Synapses
    for (let i = 0; i < layerCoords.length - 1; i++) {
        layerCoords[i].forEach(n1 => {
            layerCoords[i + 1].forEach(n2 => {
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", n1.x); line.setAttribute("y1", n1.y);
                line.setAttribute("x2", n2.x); line.setAttribute("y2", n2.y);
                line.setAttribute("stroke", "#333");
                svg.appendChild(line);
            });
        });
    }

    // Draw Neurons
    layerCoords.forEach((layer, i) => {
        layer.forEach((n, j) => {
            const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
            group.style.cursor = "pointer";
            group.onclick = () => inspectNeuron(i, j);

            const isSelected = lastInspected && lastInspected.layerIdx === i && lastInspected.neuronIdx === j;

            const outer = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            outer.setAttribute("cx", n.x); outer.setAttribute("cy", n.y);
            outer.setAttribute("r", "20");
            outer.setAttribute("fill", "#111");
            outer.setAttribute("stroke", isSelected ? "#fff" : LAYER_COLORS[i % LAYER_COLORS.length]);
            outer.setAttribute("stroke-width", isSelected ? "4" : "2");

            const dot1 = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            dot1.setAttribute("cx", n.x - 6); dot1.setAttribute("cy", n.y);
            dot1.setAttribute("r", "4"); dot1.setAttribute("fill", "#555");

            const dot2 = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            dot2.setAttribute("cx", n.x + 6); dot2.setAttribute("cy", n.y);
            dot2.setAttribute("r", "4"); dot2.setAttribute("fill", "#00ff00");

            group.appendChild(outer); group.appendChild(dot1); group.appendChild(dot2);
            svg.appendChild(group);
        });
    });
}

function inspectNeuron(layerIdx, neuronIdx) {
    lastInspected = { layerIdx, neuronIdx };
    const content = document.getElementById('inspect-content');
    const panel = document.getElementById('inspector-panel');
    panel.style.display = 'block';

    if (layerIdx === 0) {
        content.innerHTML = `<h4>Input Node ${neuronIdx}</h4><p>Passes data forward directly.</p>`;
        drawNetwork(); // Refresh highlights
        return;
    }

    // MAP TO nn.js: layerIdx 1 is window.network.levels[0]
    const layerData = window.network.levels[layerIdx - 1];
    const weights = layerData.weights.data[neuronIdx]; // Array of weights from prev layer
    const bias = layerData.biases.data[neuronIdx][0];
    const activation = topology[layerIdx].activation;

    content.innerHTML = `
        <h4 style="color:${LAYER_COLORS[layerIdx % LAYER_COLORS.length]}">Layer ${layerIdx} | Neuron ${neuronIdx}</h4>
        <span class="inspector-label">Activation Function</span>
        <div style="font-weight:bold;">${activation}</div>
        
        <span class="inspector-label">Bias (b)</span>
        <div style="color:#00ff00; font-family:monospace;">${bias.toFixed(6)}</div>

        <span class="inspector-label">Weights (W) from Layer ${layerIdx - 1}</span>
        <div class="weight-grid">
            ${weights.map((w, idx) => `<div class="weight-item"><div style="font-size:9px; color:#888;">w${idx}</div>${w.toFixed(4)}</div>`).join('')}
        </div>
    `;
    drawNetwork(); // Update the white highlight on the SVG
}

// Event handlers
window.changeNeuron = (i, d) => { topology[i].count += d; syncAndDraw(); };
window.changeActivation = (i, val) => { topology[i].activation = val; syncAndDraw(); };
window.removeLayer = (i) => { topology.splice(i, 1); syncAndDraw(); };
document.getElementById('addLayerBtn').onclick = () => {
    if (topology.length < MAX_LAYERS) {
        topology.splice(topology.length - 1, 0, { count: 4, activation: 'relu' });
        syncAndDraw();
    }
};


/// --- Table Management Logic ---

const tableHead = document.getElementById('table-head');
const tableBody = document.getElementById('table-body');
const dataStatus = document.getElementById('data-status');

/**
 * Creates the table headers based on current Input/Output counts.
 */
function updateTableHeaders() {
    const inputLen = topology[0].count;
    const outputLen = topology[topology.length - 1].count;

    let html = '<tr>';
    for (let i = 0; i < inputLen; i++) html += `<th style="padding:10px; color:#3498db; border:1px solid #333;">In ${i}</th>`;
    for (let i = 0; i < outputLen; i++) html += `<th style="padding:10px; color:#e74c3c; border:1px solid #333;">Out ${i}</th>`;
    html += '<th style="width:40px; border:1px solid #333;"></th></tr>';

    tableHead.innerHTML = html;
}

/**
 * Adds a new row to the table. 
 * @param {Array} values - Optional array of [inputs..., outputs...]
 */
/**
 * Adds a new row to the table with a softer visual style.
 */
function addDataRow(values = null) {
    const rowCount = tableBody.rows.length;
    if (rowCount >= 50) return;

    const inputLen = topology[0].count;
    const outputLen = topology[topology.length - 1].count;
    const totalCols = inputLen + outputLen;

    const tr = document.createElement('tr');
    tr.style.borderBottom = "1px solid #333";

    for (let i = 0; i < totalCols; i++) {
        const td = document.createElement('td');
        td.style.padding = "4px";
        const val = values ? values[i] : 0;

        // Using the new .data-cell-input class instead of inline black background
        td.innerHTML = `<input type="number" step="0.1" value="${val}" class="data-cell-input">`;
        tr.appendChild(td);
    }

    const delTd = document.createElement('td');
    delTd.innerHTML = `<button onclick="this.parentElement.parentElement.remove(); updateDataCount();" 
                       style="border:none; color:#ff4444; background:none; cursor:pointer; font-size:1.2em;">&times;</button>`;
    tr.appendChild(delTd);

    tableBody.appendChild(tr);
    updateDataCount();
}

function updateDataCount() {
    dataStatus.innerText = `${tableBody.rows.length} / 50 rows used.`;
    drawDataPlot();
}

const getTrainingData = () => {
    const data = [];
    const inputLen = topology[0].count;
    const outputLen = topology[topology.length - 1].count;

    // Iterate through rows in the table body
    for (let row of tableBody.rows) {
        const inputs = [];
        const outputs = [];

        // Get all input elements in this row
        const rowInputs = row.querySelectorAll('input[type="number"]');

        if (rowInputs.length >= (inputLen + outputLen)) {
            // Extract Input values
            for (let i = 0; i < inputLen; i++) {
                inputs.push(parseFloat(rowInputs[i].value) || 0);
            }
            // Extract Output values (targets)
            for (let i = 0; i < outputLen; i++) {
                outputs.push(parseFloat(rowInputs[inputLen + i].value) || 0);
            }
            data.push({ x: inputs, y: outputs });
        }
    }
    return data;
};

// Event Listeners
document.getElementById('addRowBtn').onclick = () => addDataRow();
document.getElementById('clearDataBtn').onclick = () => { tableBody.innerHTML = ''; updateDataCount(); };
document.getElementById('randomizeDataBtn').onclick = () => {
    const inputLen = topology[0].count;
    const outputLen = topology[topology.length - 1].count;
    for (let i = 0; i < 5; i++) {
        const randVals = [...Array(inputLen).fill(0).map(() => Math.random().toFixed(2)), ...Array(outputLen).fill(0).map(() => Math.round(Math.random()))];
        addDataRow(randVals);
    }
};


function drawDataPlot() {
    const dataset = getTrainingData();
    const inCount = topology[0].count;
    const container = 'data-plot-container';
    let traces = [];
    
    let layout = { 
        paper_bgcolor: 'rgba(0,0,0,0)', 
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#888' },
        margin: { t: 0, r: 0, b: 0, l: 0 },
        scene: {
            xaxis: { title: 'In 0', gridcolor: '#333', backgroundcolor: '#111' },
            yaxis: { title: 'In 1', gridcolor: '#333', backgroundcolor: '#111' },
            zaxis: { title: 'Output', gridcolor: '#333', backgroundcolor: '#111' }
        }
    };

    if (inCount === 1) {
        // 1D Case: Standard 2D Line Chart (X vs Output)
        const lx = [], ly = [];
        for (let i = 0; i <= 50; i++) {
            const val = i / 50;
            lx.push(val);
            ly.push(window.network.predict([val])[0]);
        }
        traces.push({ x: lx, y: ly, mode: 'lines', line: { color: '#00ff00', width: 3 }, name: 'Network' });
        traces.push({
            x: dataset.map(p => p.x[0]), y: dataset.map(p => p.y[0]),
            mode: 'markers', marker: { color: '#ff4444', size: 10, line: { color: '#fff', width: 1 } }, name: 'Data'
        });

    } else if (inCount === 2) {
        // 2-Input Case: 3D Surface (X, Y are inputs, Z is output)
        const res = 20;
        let zMatrix = [];
        let xRange = [], yRange = [];
        for (let i = 0; i < res; i++) {
            xRange.push(i / (res - 1));
            yRange.push(i / (res - 1));
        }

        for (let i = 0; i < res; i++) {
            let row = [];
            for (let j = 0; j < res; j++) {
                row.push(window.network.predict([xRange[j], yRange[i]])[0]);
            }
            zMatrix.push(row);
        }

        // The "Plane" (Network Prediction)
        traces.push({
            z: zMatrix, x: xRange, y: yRange,
            type: 'surface', colorscale: 'Viridis', opacity: 0.8, showscale: false
        });

        // The "Scatters" (Training Data)
        traces.push({
            x: dataset.map(p => p.x[0]),
            y: dataset.map(p => p.x[1]),
            z: dataset.map(p => p.y[0]),
            mode: 'markers', type: 'scatter3d',
            marker: { size: 6, color: '#ff4444', symbol: 'circle', line: { color: '#000', width: 1 } }
        });

    } else if (inCount === 3) {
        // 3-Input Case: "Cool" 3D Color Clusters
        traces.push({
            x: dataset.map(p => p.x[0]),
            y: dataset.map(p => p.x[1]),
            z: dataset.map(p => p.x[2]),
            mode: 'markers', type: 'scatter3d',
            marker: {
                size: 8,
                color: dataset.map(p => p.y[0]), // Output determines color
                colorscale: 'Portland',
                showscale: true
            }
        });
        layout.scene.zaxis.title = "In 2";

    } else {
        // High-D Case: Scatter Plot Matrix (SPLOM)
        const dims = [];
        for(let i=0; i < inCount; i++) {
            dims.push({ label: `In ${i}`, values: dataset.map(p => p.x[i]) });
        }
        traces.push({
            type: 'splom', dimensions: dims,
            marker: { color: dataset.map(p => p.y[0]), colorscale: 'Viridis', size: 5 }
        });
        layout.dragmode = 'select';
    }

    Plotly.react(container, traces, layout);
}

document.getElementById('trainBtn').onclick = async () => {
    const trainingData = getTrainingData();
    if (trainingData.length === 0) return alert("Please add training data to the table first!");

    const epochs = parseInt(document.getElementById('epochInput').value) || 50;
    const lr = 0.1;
    const status = document.getElementById('training-status');
    const btn = document.getElementById('trainBtn');

    btn.disabled = true;
    btn.style.opacity = 0.5;

    for (let e = 1; e <= epochs; e++) {
        // Shuffle for Stochastic Gradient Descent
        const shuffled = [...trainingData].sort(() => Math.random() - 0.5);
        
        for (let sample of shuffled) {
            window.network.train(sample.x, sample.y, lr);
        }

        status.innerText = `Epoch: ${e}/${epochs}`;
        
        // Change 1: Redraw after each epoch
        drawDataPlot();
        
        // Allow the browser to render the frame
        await new Promise(r => requestAnimationFrame(r));
    }

    status.innerText = "Training Complete!";
    btn.disabled = false;
    btn.style.opacity = 1;
};

// Start
syncAndDraw();