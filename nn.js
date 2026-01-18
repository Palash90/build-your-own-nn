const Activations = {
    sigmoid: {
        func: x => 1 / (1 + Math.exp(-x)),
        prime: x => {
            const s = 1 / (1 + Math.exp(-x));
            return s * (1 - s);
        }
    },
    tanh: {
        func: x => Math.tanh(x),
        prime: x => 1 - Math.pow(Math.tanh(x), 2)
    },
    relu: {
        func: x => Math.max(0, x),
        prime: x => (x > 0 ? 1 : 0)
    },
    leakyRelu: {
        func: x => (x > 0 ? x : 0.01 * x),
        prime: x => (x > 0 ? 1 : 0.01)
    },
    linear: {
        func: x => x,
        prime: x => 1
    },
    // Softmax is unique as it depends on the whole vector
    softmax: {
        func: (arr) => {
            const maxVal = Math.max(...arr); // for numerical stability
            const exps = arr.map(x => Math.exp(x - maxVal));
            const sum = exps.reduce((a, b) => a + b);
            return exps.map(x => x / sum);
        }
    }
};

const Losses = {
    // Mean Squared Error: (Target - Output)^2
    // Used mainly for regression
    mse: {
        func: (target, output) => {
            return target.reduce((acc, t, i) => acc + Math.pow(t - output[i], 2), 0) / target.length;
        },
        prime: (target, output) => {
            // Derivative: 2 * (Output - Target)
            return output.map((o, i) => 2 * (o - target[i]));
        }
    },

    // Binary Cross-Entropy
    // Used for binary classification (0 or 1)
    bce: {
        func: (target, output) => {
            return -target.reduce((acc, t, i) => {
                return acc + (t * Math.log(output[i] + 1e-15) + (1 - t) * Math.log(1 - output[i] + 1e-15));
            }, 0) / target.length;
        },
        prime: (target, output) => {
            return output.map((o, i) => (o - target[i]) / ((o * (1 - o)) + 1e-15));
        }
    },

    // Categorical Cross-Entropy
    // Used for multi-class classification (with Softmax)
    cce: {
        func: (target, output) => {
            return -target.reduce((acc, t, i) => acc + t * Math.log(output[i] + 1e-15), 0);
        },
        prime: (target, output) => {
            // Note: CCE + Softmax simplifies beautifully to (Output - Target)
            return output.map((o, i) => o - target[i]);
        }
    }
};

class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        // Initialize with a 2D array of zeros
        this.data = Array.from({ length: rows }, () => new Array(cols).fill(0));
    }

    // Fill the matrix with random values between -1 and 1
    randomize() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
        return this; // Return 'this' for method chaining
    }

    // Matrix Multiplication: (A x B) * (B x C) = (A x C)
    static multiply(a, b) {
        if (a.cols !== b.rows) {
            console.error("Columns of A must match Rows of B.");
            return null;
        }

        let result = new Matrix(a.rows, b.cols);
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    clone() {
        let copy = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                copy.data[i][j] = this.data[i][j];
            }
        }
        return copy;
    }

    // Helper to create a Matrix from a flat array (useful for inputs)
    static fromArray(arr) {
        let m = new Matrix(arr.length, 1);
        for (let i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }

    // Helper to turn Matrix back into a flat array (useful for SVG drawing)
    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }

    // Inside Matrix class in nn.js
    map(fn) {
        if (typeof fn !== 'function') {
            console.error("Matrix.map received a non-function:", fn);
            return this;
        }
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = fn(val);
            }
        }
        return this;
    }

    static add(a, b) {
        let result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return result;
    }

    // Inside Matrix class
    static transpose(matrix) {
        let result = new Matrix(matrix.cols, matrix.rows);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                result.data[j][i] = matrix.data[i][j];
            }
        }
        return result;
    }

    static multiplyElementwise(a, b) {
        let result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] * b.data[i][j];
            }
        }
        return result;
    }

    // Simple subtraction for error calculation (Target - Output)
    static subtract(a, b) {
        let result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }
}



class Layer {
    constructor(inputCount, outputCount, activationKey = 'sigmoid') {
        this.weights = new Matrix(outputCount, inputCount).randomize();
        this.biases = new Matrix(outputCount, 1).randomize();

        // Ensure we grab the object, not just the string
        this.activationName = activationKey;
        this.activation = Activations[activationKey];

        if (!this.activation) {
            console.error(`Activation '${activationKey}' not found! Defaulting to linear.`);
            this.activation = Activations.linear;
        }
    }

    feedForward(inputMatrix) {
        // Z = W * I + B
        let z = Matrix.multiply(this.weights, inputMatrix);
        z = Matrix.add(z, this.biases);

        // Softmax check
        if (this.activationName === 'softmax') {
            const arr = z.toArray();
            const softRes = Activations.softmax.func(arr);
            return Matrix.fromArray(softRes);
        }

        // Standard element-wise activation
        return z.clone().map(this.activation.func);
    }
}

class NeuralNetwork {
    constructor(neuronCounts, activationTypes = [], lossType = 'mse') {
        this.levels = [];
        this.loss = Losses[lossType];
        for (let i = 0; i < neuronCounts.length - 1; i++) {
            const act = activationTypes[i] || 'sigmoid';
            this.levels.push(new Layer(neuronCounts[i], neuronCounts[i + 1], act));
        }
    }

    // Helper to run forward pass while saving intermediate states for backprop
    forwardPassInternal(inputArray) {
        let inputs = Matrix.fromArray(inputArray);
        let layerData = [];

        let currentInput = inputs;
        for (let layer of this.levels) {
            let z = Matrix.multiply(layer.weights, currentInput);
            z = Matrix.add(z, layer.biases);

            let output;
            if (layer.activation === Activations.softmax) {
                // Softmax needs the whole array at once
                let rawValues = z.toArray();
                let softValues = layer.activation.func(rawValues);
                output = Matrix.fromArray(softValues);
            } else {
                output = z.clone().map(layer.activation.func);
            }

            layerData.push({ input: currentInput, z: z, output: output });
            currentInput = output;
        }
        return { layerData, finalOutput: currentInput.toArray() };
    }

    predict(inputArray) {
        return this.forwardPassInternal(inputArray).finalOutput;
    }

    train(inputArray, targetArray, learningRate = 0.1) {
    const { layerData, finalOutput } = this.forwardPassInternal(inputArray);

    // 1. Initial Error: Derivative of Loss w.r.t Output
    let errorGrads = this.loss.prime(targetArray, finalOutput);
    let error = Matrix.fromArray(errorGrads);

    // 2. Backward Pass
    for (let i = this.levels.length - 1; i >= 0; i--) {
        let layer = this.levels[i];
        let data = layerData[i];

        // f'(z)
        let actDeriv;
        if (layer.activationName === 'softmax') {
            actDeriv = data.z.clone().map(x => 1); 
        } else {
            actDeriv = data.z.clone().map(layer.activation.prime);
        }

        // Delta = error * f'(z)
        let delta = Matrix.multiplyElementwise(error, actDeriv);

        // Calculate Weight Gradients: Delta * Input^T
        let inputT = Matrix.transpose(data.input);
        let weightGradients = Matrix.multiply(delta, inputT);

        // --- UPDATE WEIGHTS & BIASES ---
        // Scale gradients by learning rate
        weightGradients.map(x => x * learningRate);
        delta.map(x => x * learningRate);

        // Subtract gradients from weights/biases (Gradient Descent)
        layer.weights = Matrix.subtract(layer.weights, weightGradients);
        layer.biases = Matrix.subtract(layer.biases, delta);

        // Propagate error back to previous layer: W^T * Delta
        let weightsT = Matrix.transpose(layer.weights);
        error = Matrix.multiply(weightsT, delta);
    }
}
}
console.group("Component Test: Matrix");

const A = new Matrix(2, 3); // 2 rows, 3 cols
A.data = [[1, 2, 3], [4, 5, 6]];

const B = new Matrix(3, 2); // 3 rows, 2 cols
B.data = [[7, 8], [9, 10], [11, 12]];

const C = Matrix.multiply(A, B);

if (C && C.rows === 2 && C.cols === 2 && C.data[0][0] === 58) {
    console.log("✅ Matrix Multiplication: Success (58 computed correctly)");
} else {
    console.error("❌ Matrix Multiplication: Failed");
}

const mapTest = new Matrix(2, 2);
mapTest.data = [[1, 2], [3, 4]];
mapTest.map(x => x * 2);
if (mapTest.data[0][0] === 2 && mapTest.data[1][1] === 8) {
    console.log("✅ Matrix Map: Success");
} else {
    console.error("❌ Matrix Map: Failed");
}

console.groupEnd();

console.group("Component Test: Activations");

const testVal = 0;
const results = {
    sigmoid: Activations.sigmoid.func(testVal) === 0.5,
    tanh: Activations.tanh.func(testVal) === 0,
    relu: Activations.relu.func(-5) === 0 && Activations.relu.func(5) === 5,
    leaky: Activations.leakyRelu.func(-1) === -0.01,
    linear: Activations.linear.func(10) === 10
};

Object.entries(results).forEach(([name, passed]) => {
    console.log(`${passed ? '✅' : '❌'} ${name.toUpperCase()}: ${passed ? 'Passed' : 'Failed'}`);
});

// Softmax special check
const sm = Activations.softmax.func([1, 1, 1]); // Should be [0.33, 0.33, 0.33]
const smSum = sm.reduce((a, b) => a + b);
console.log(Math.abs(smSum - 1) < 0.0001 ? "✅ SOFTMAX: Probability sums to 1" : "❌ SOFTMAX: Sum failed");

console.groupEnd();

console.group("Component Test: Neural Network");

const architecture = [3, 5, 2]; // 3 inputs -> 5 hidden -> 2 outputs
const net = new NeuralNetwork(architecture, ['relu', 'softmax']);

const input = [1, 0.5, -1];
const output = net.predict(input);

console.log("Input:", input);
console.log("Output:", output);

if (output.length === 2) {
    console.log("✅ Feed Forward: Correct output dimensions");
} else {
    console.error("❌ Feed Forward: Dimension mismatch");
}

console.groupEnd();

function finalMathCheck() {
    console.group("Final Math Integrity Check");

    try {
        // 1. Check Matrix
        const m = new Matrix(2, 2).randomize();
        m.map(x => x * 2);
        console.log("✅ Matrix Map works");

        // 2. Check Activation Access
        if (typeof Activations.sigmoid.func === 'function') {
            console.log("✅ Activations Object is accessible");
        }

        // 3. Check Network with Softmax (The tricky one)
        const net = new NeuralNetwork([2, 3, 2], ['relu', 'softmax'], 'cce');
        const output = net.predict([1, 0.5]);

        console.log("Output values:", output);
        const sum = output.reduce((a, b) => a + b, 0);
        if (Math.abs(sum - 1) < 0.001) {
            console.log("✅ Softmax Math: Correct (Sums to 1)");
        }

        // 4. Check Training Step
        net.train([1, 0.5], [0, 1], 0.1);
        console.log("✅ Training Epoch: Executed without errors");

    } catch (e) {
        console.error("❌ Math Test Failed:", e.message);
    }

    console.groupEnd();
}

finalMathCheck();


// Example: Training the network to solve XOR
const xorNet = new NeuralNetwork([2, 4, 1], ['sigmoid', 'sigmoid'], 'bce');

const trainingData = [
    { input: [0, 0], target: [0] },
    { input: [0, 1], target: [1] },
    { input: [1, 0], target: [1] },
    { input: [1, 1], target: [0] }
];

// Training loop (10,000 iterations)
for (let i = 0; i < 10000; i++) {
    const data = trainingData[Math.floor(Math.random() * trainingData.length)];
    xorNet.train(data.input, data.target, 0.1);
}

// Test results
console.log("0,0 ->", xorNet.predict([0, 0]));
console.log("0,1 ->", xorNet.predict([0, 1]));