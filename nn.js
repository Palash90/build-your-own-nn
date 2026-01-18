let currentStep = 0;
let isForwardPass = true;
let errorSignal = [];
let stepperInput = []; // To remember the original input for backprop

function highlightLayer(idx, isBackward = false) {
    const color = isBackward ? "#ff00ff" : "#00ff00";
    const nodes = document.querySelectorAll(`.layer-group-${idx} .neuron-shell`);

    nodes.forEach(n => {
        n.style.filter = `drop-shadow(0 0 8px ${color})`;
    });
}

function stepDebugger() {
    const inspector = document.getElementById('inspect-content');

    if (isForwardPass) {
        if (currentStep === 0) {
            stepperInput = document.getElementById('vectorInput').value
                .split(',').map(v => parseFloat(v.trim()));
            window.activeSignal = [...stepperInput];
            highlightLayer(0);
            inspector.innerHTML = `<h4 style="color:#0088ff">FORWARD: Input Layer</h4>
                                   <p>Data: [${window.activeSignal.map(v => v.toFixed(2))}]</p>`;
            currentStep = 1;
            return;
        }

        const layerIdx = currentStep - 1;
        if (layerIdx < network.length) {
            highlightLayer(currentStep);
            window.activeSignal = network[layerIdx].forward(window.activeSignal);
            inspector.innerHTML = `<h4 style="color:#0088ff">FORWARD: Layer ${currentStep} (${network[layerIdx].activationType})</h4>
                                   <p>Output: [${window.activeSignal.map(v => v.toFixed(4))}]</p>`;
            currentStep++;

            if (currentStep === currentConfigs.length) {
                isForwardPass = false;
                // Set currentStep to the index of the very last layer (Output Layer)
                currentStep = network.length;
                const targetVector = document.getElementById('targetInput').value
                    .split(',').map(v => parseFloat(v.trim()));
                calculateLoss(window.activeSignal, targetVector);
                inspector.innerHTML += `<p style="color:yellow">Forward Pass Complete. Ready for BACKWARD PASS.</p>`;
            }
        }
        return;
    }

    if (!isForwardPass) {
        highlightLayer(currentStep, true);

        const layerIdx = currentStep - 1; // network array is 0-indexed starting from first hidden layer
        const layer = network[layerIdx];
        const isOutput = (currentStep === network.length);

        if (isOutput) {
            errorSignal = document.getElementById('targetInput').value
                .split(',').map(v => parseFloat(v.trim()));
        }

        // Calculate gradients and update weights
        errorSignal = layer.backward(errorSignal, 0.1, isOutput);

        inspector.innerHTML = `<h4 style="color:#ff00ff">BACKWARD: Layer ${currentStep}</h4>
                               <p>Gradients calculated and weights updated.</p>`;

        // Move to the previous layer
        currentStep--;

        // If we've reached the input layer (0), the cycle is done
        if (currentStep < 1) {
            setTimeout(() => {
                inspector.innerHTML += `<h4 style="color:#00ff00">Full Cycle Complete!</h4>`;
                resetDebugger();
                draw(); // Redraw to clear highlights
            }, 500);
        }
    }
}
function resetDebugger() {
    currentStep = 0;
    isForwardPass = true;


    document.querySelectorAll('.neuron-shell').forEach(el => {
        el.classList.remove('active-layer');
        el.style.filter = '';
        el.style.stroke = '';
    });

}

function calculateLoss(output, target) {
    // 1. Identify the last layer's activation type
    const lastLayer = network[network.length - 1];
    const type = lastLayer.activationType;
    let loss = 0;

    try {
        if (type === 'softmax') {
            // --- Categorical Cross-Entropy (CCE) ---
            // Formula: -Sum(target * log(output))
            // Added 1e-15 to prevent log(0) which results in Infinity
            for (let i = 0; i < output.length; i++) {
                loss -= target[i] * Math.log(output[i] + 1e-15);
            }
        }
        else if (type === 'sigmoid') {
            // --- Binary Cross-Entropy (BCE) ---
            // Formula: -avg(target * log(output) + (1-target) * log(1-output))
            let sum = 0;
            for (let i = 0; i < output.length; i++) {
                sum += target[i] * Math.log(output[i] + 1e-15) +
                    (1 - target[i]) * Math.log(1 - output[i] + 1e-15);
            }
            loss = -sum / output.length;
        }
        else {
            // --- Mean Squared Error (MSE) ---
            // Default for 'linear', 'relu', 'tanh'
            let sumSqError = 0;
            for (let i = 0; i < output.length; i++) {
                sumSqError += Math.pow(output[i] - target[i], 2);
            }
            loss = sumSqError / output.length;
        }
    } catch (e) {
        console.error(e)
    }


    // 2. Update the UI
    const lossLabel = type === 'softmax' ? 'CCE' : (type === 'sigmoid' ? 'BCE' : 'MSE');
    document.getElementById('lossValue').innerHTML =
        `${loss.toFixed(6)} <small style="color:#666">(${lossLabel})</small>`;
}

const Activations = {
    linear: (x) => x,
    relu: (x) => Math.max(0, x),
    sigmoid: (x) => 1 / (1 + Math.exp(-x)),
    tanh: (x) => Math.tanh(x),
    softmax: (arr) => {
        const exponents = arr.map(v => Math.exp(v));
        const sum = exponents.reduce((a, b) => a + b, 0);
        return exponents.map(v => v / sum);
    }
};

class Layer {
    constructor(inputSize, outputSize, activationType, layerIndex) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activationType = activationType;
        this.layerIndex = layerIndex;

        // Weights: Rows = inputSize + 1 (Weight index 0 is the bias)
        this.weights = Array.from({ length: inputSize + 1 }, () =>
            Array.from({ length: outputSize }, () => (Math.random() * 2 - 1))
        );

        this.lastInput = []; //
        this.lastZ = new Array(outputSize).fill(0); //
        this.lastA = new Array(outputSize).fill(0); //
        this.deltas = new Array(outputSize).fill(0);
    }

    // --- Forward Pass (Existing) ---
    forward(input) {
        this.lastInput = [...input]; //
        const biasedInput = [1.0, ...input]; //

        const zValues = []; //
        for (let j = 0; j < this.outputSize; j++) {
            let sum = 0;
            for (let i = 0; i < biasedInput.length; i++) {
                sum += biasedInput[i] * this.weights[i][j]; //
            }
            zValues.push(sum);
        }

        this.lastZ = zValues; //
        this.lastA = this.activationType === 'softmax'  //
            ? Activations.softmax(this.lastZ)
            : this.lastZ.map(z => Activations[this.activationType](z));
        return this.lastA;
    }

    // Inside the Layer class
    // Replace the backward method in your Layer class
    backward(targetOrNextError, learningRate, isOutputLayer = false) {
        const newDeltas = new Array(this.outputSize);

        // 1. Calculate Deltas
        for (let j = 0; j < this.outputSize; j++) {
            let error = isOutputLayer
                ? this.lastA[j] - targetOrNextError[j]
                : targetOrNextError[j];

            let derivative = 1;
            if (this.activationType === 'relu') derivative = this.lastZ[j] > 0 ? 1 : 0;
            if (this.activationType === 'sigmoid') derivative = this.lastA[j] * (1 - this.lastA[j]);
            if (this.activationType === 'tanh') derivative = 1 - Math.pow(this.lastA[j], 2); // Fixed variable name

            newDeltas[j] = error * derivative;
        }

        // 2. Calculate error for PREVIOUS layer BEFORE updating weights
        const prevLayerError = new Array(this.inputSize).fill(0);
        for (let i = 1; i <= this.inputSize; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                prevLayerError[i - 1] += this.weights[i][j] * newDeltas[j];
            }
        }

        // 3. Update Weights
        const biasedInput = [1.0, ...this.lastInput];
        for (let i = 0; i < biasedInput.length; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                const gradient = newDeltas[j] * biasedInput[i];
                this.weights[i][j] -= learningRate * gradient;
            }
        }

        this.deltas = newDeltas;
        return prevLayerError;
    }
}

let network = [];
let currentConfigs = [];

function initNetwork() {
    const inputStr = document.getElementById('topoInput').value;
    currentConfigs = inputStr.split(',').map(part => {
        const [size, act] = part.trim().split(':');
        return { size: parseInt(size), act: act.trim().toLowerCase() };
    });

    network = [];
    for (let i = 1; i < currentConfigs.length; i++) {
        network.push(new Layer(currentConfigs[i - 1].size, currentConfigs[i].size, currentConfigs[i].act, i));
    }


    document.getElementById('inspect-content').innerHTML = '<p style="color:#00ff00">Network Initialized with Random Weights.</p>';

    calculateLoss();
    resetDebugger();
    draw();
}

function draw() {
    const svg = document.getElementById('mainSvg');
    const nodesG = document.getElementById('nodesGroup');
    const linksG = document.getElementById('linksGroup');

    nodesG.innerHTML = '';
    linksG.innerHTML = '';

    const rect = svg.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    if (w === 0 || h === 0) {
        setTimeout(draw, 100);
        return;
    }

    const margin = 80;
    const nodeSpacing = 50;

    // 1. Calculate Coordinates for all nodes
    const layerCoords = currentConfigs.map((config, lIdx) => {
        const x = margin + (lIdx * (w - 2 * margin) / (currentConfigs.length - 1));

        const layerHeight = (config.size - 1) * nodeSpacing;

        const startY = (h - layerHeight) / 2;

        return Array.from({ length: config.size }, (_, nIdx) => ({
            x,
            y: config.size === 1 ? h / 2 : startY + (nIdx * nodeSpacing),
            lIdx,
            nIdx,
            act: config.act
        }));
    });

    layerCoords.forEach((layer, lIdx) => {
        if (lIdx === 0) return; // Skip input layer

        const prevLayer = layerCoords[lIdx - 1];
        const layerObj = network[lIdx - 1];

        layer.forEach((node, nIdx) => {
            prevLayer.forEach((prevNode, pIdx) => {
                // 1. Get the weight first!
                const weight = layerObj.weights[pIdx + 1][nIdx];

                let color, opacity, thickness;

                if (isNaN(weight) || !isFinite(weight)) {
                    // High visibility for errors
                    color = "#ffff00"; // Bright Yellow for NaN/Inf
                    opacity = 1.0;
                    thickness = 4;
                } else {
                    // Normal weight visualization
                    color = weight > 0 ? "#0088ff" : "#ff4444";
                    opacity = Math.max(Math.abs(weight), 0.15);
                    thickness = Math.max(Math.abs(weight) * 3, 0.5);
                }

                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", prevNode.x + 25);
                line.setAttribute("y1", prevNode.y);
                line.setAttribute("x2", node.x - 25);
                line.setAttribute("y2", node.y);

                if (currentStep === lIdx) {
                    line.setAttribute("stroke", "#00ff00"); // Bright green for active layer
                    line.setAttribute("stroke-width", thickness + 2);
                    line.style.opacity = 1.0;
                } else {
                    line.setAttribute("stroke", color);
                    line.setAttribute("stroke-width", thickness);
                    line.style.opacity = opacity;
                }

                // 3. Apply the styles
                line.setAttribute("stroke", color);
                line.setAttribute("stroke-width", thickness);
                line.style.opacity = opacity;
                line.setAttribute("class", "link");

                linksG.appendChild(line);
            });
        });
    });

    // 3. Draw Neurons
    layerCoords.forEach((layer, lIdx) => {
        layer.forEach((node, nIdx) => {
            const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
            g.setAttribute("class", `layer-group-${lIdx}`);

            const shell = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            shell.setAttribute("x", node.x - 25); shell.setAttribute("y", node.y - 15);
            shell.setAttribute("width", 50); shell.setAttribute("height", 30); shell.setAttribute("rx", 15);
            shell.setAttribute("class", "neuron-shell");

            // Highlight the active layer nodes
            if (currentStep === lIdx) {
                shell.classList.add('active-layer');
            }

            g.appendChild(shell);

            // Linear Stage (Blue)
            const cLin = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            cLin.setAttribute("cx", node.x - 10); cLin.setAttribute("cy", node.y);
            cLin.setAttribute("r", 7); cLin.setAttribute("class", "stage-linear");
            cLin.onclick = () => inspectStage(lIdx, nIdx, 'linear');
            g.appendChild(cLin);

            // Activation Stage (Color based on type)
            const cAct = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            cAct.setAttribute("cx", node.x + 10); cAct.setAttribute("cy", node.y);
            cAct.setAttribute("r", 7);
            cAct.setAttribute("class", `stage-activation act-${node.act}`);
            cAct.onclick = () => inspectStage(lIdx, nIdx, 'activation');
            g.appendChild(cAct);

            nodesG.appendChild(g);
        });
    });
}


function inspectStage(lIdx, nIdx, stage) {
    const content = document.getElementById('inspect-content');
    if (lIdx === 0) {
        content.innerHTML = `<h4>Input Node ${nIdx}</h4><p>Value passed directly from input vector.</p>`;
        return;
    }

    const layerObj = network[lIdx - 1];
    const nodeWeights = layerObj.weights.map(row => row[nIdx]);
    const biasWeight = nodeWeights[0];
    const actualWeights = nodeWeights.slice(1);

    content.innerHTML = `
                <h4>L:${lIdx} Node:${nIdx}</h4>
                <div style="color:${stage === 'linear' ? '#0088ff' : '#ff00ff'}">STAGE: ${stage.toUpperCase()}</div>
                <hr style="border:0; border-top:1px solid #444; margin:10px 0;">
                
                <b>Activation Type:</b>
                <span class="data-val">${layerObj.activationType}</span>

                <b style="margin-top:10px; display:block;">Bias Trick Weight (w0):</b>
                <span class="data-val">${biasWeight}</span>

                <b style="margin-top:10px; display:block;">Weights from Prev Layer:</b>
                <span class="data-val">${actualWeights.join(' | ')}</span>

                <b style="margin-top:10px; display:block;">Linear Sum (z):</b>
                <span class="data-val" style="color:#0088ff">${layerObj.lastZ[nIdx].toFixed(6)}</span>

                <b style="margin-top:10px; display:block;">Activated Output (a):</b>
                <span class="data-val" style="color:#ff00ff">${layerObj.lastA[nIdx].toFixed(6)}</span>
            `;
}

window.onload = initNetwork; W