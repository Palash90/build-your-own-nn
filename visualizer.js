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

    // AUTO-REFRESH INSPECTOR: If the user was looking at a neuron, update its data
    if (lastInspected) {
        inspectNeuron(lastInspected.layerIdx, lastInspected.neuronIdx);
    }
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

// Start
syncAndDraw();