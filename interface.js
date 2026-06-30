// -------------------------------------------------------------- //
// -------------------- OUTPUT CHART SETUP ---------------------- //
// -------------------------------------------------------------- //
const chartData = {
    labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    datasets: [{
        data: Array(10).fill(0),
        backgroundColor: 'rgba(10, 10, 10, 0.9)'
    }]
};

const chartConfig = {
    type: 'bar',
    data: chartData,
    options: {
        interaction: { mode: 'index', intersect: false },
        devicePixelRatio: window.devicePixelRatio || 1,
        plugins: { 
            legend: { display: false },
            tooltip: {
                displayColors: false,
                callbacks: {
                    title: (items) => `eval of ${items[0].label}`,
                    label: (item) => `🚀 ${item.formattedValue}`,
                },
                padding: 7
            }
        },
        layout: { padding: { top: 15, right: 15, bottom: 0, left: 5 } },
        scales: {
            y: {
                ticks: {
                    color: 'rgb(10, 10, 10)',
                    font: { size: 12, weight: 'bold' },
                    stepSize: 0.2, padding: 5,
                },
                min: 0, max: 1
            },
            x : {
                ticks: {
                    color: 'rgb(10, 10, 10)',
                    font: { size: 12, weight: 'bold' },
                    padding: 5,
                }
            }
        },
    }
};

const outCanvas = document.getElementById('net-output');
const outChart = new Chart(outCanvas, chartConfig);


// -------------------------------------------------------------- //
// ------------------------ PYODIDE SETUP ----------------------- //
// -------------------------------------------------------------- //
const pyodide = await loadPyodide();

async function loadPythonFile(name) {
  const r = await fetch(name, { cache: "no-cache" });
  const text = await r.text();

  if (!r.ok) throw new Error(`Failed to load ${name}: ${r.status}\n${text.slice(0,200)}`);
  pyodide.FS.writeFile(name.replace(".txt", ".py"), text);
}

await pyodide.loadPackage("numpy");
await loadPythonFile("custom_types.txt");
await loadPythonFile("validation.txt");
await loadPythonFile("factory.txt");
await loadPythonFile("activation.txt");
await loadPythonFile("initializer.txt");
await loadPythonFile("neural.txt");

const zip = await fetch("mnist-200-100.zip").then(r => r.arrayBuffer());
pyodide.FS.writeFile("mnist-200-100.zip", new Uint8Array(zip));

pyodide.runPython(`
    import numpy as np
    from neural import NeuralNetwork
    mnist_network = NeuralNetwork.from_data("mnist-200-100.zip")

    def query_mnist(input):
        x = np.array(input)
        x = x.reshape(len(x), 1)
        return mnist_network.query(x)
`);

const queryMnist = pyodide.globals.get("query_mnist");


// -------------------------------------------------------------- //
// --------------------- DRAW CANVAS SETUP ---------------------- //
// -------------------------------------------------------------- //
const inputCanvas = document.getElementById('input-canvas')
const mnistCanvas = document.getElementById('mnist-canvas')
const deleteBtn = document.getElementById('delete');


function getProperSizeCtx(canvas, border = 1) {
    // resize canvas pixel buffer to match html canvas
    // size even on high device pixel ratio (dpr) 
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // border is not part of drawing area
    const width = rect.width - 2 * border;
    const height = rect.height - 2 * border;
    
    canvas.width  = Math.round(width  * dpr);
    canvas.height = Math.round(height * dpr);

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    return ctx
}

// get input canvas ctx (well scaled)
const inputCtx = getProperSizeCtx(inputCanvas);

// get mnist canvas ctx (28 x 28 pixel)
mnistCanvas.width = 28;
mnistCanvas.height = 28;
const mnistCtx = mnistCanvas.getContext('2d');

// global draw flag and input canvas style
let drawing = false;
inputCtx.strokeStyle = 'rgb(10, 10, 10)';
inputCtx.lineWidth = 7;
inputCtx.lineCap = 'round';
inputCtx.lineJoin = 'round';

let strokes = []; // {x,y,t}
let recordStrokes = false;
const exampleStrokes = 
    await fetch("strokes.json").then(r => r.json());

function getMousePos(event) {
    // get the mouse position relative to the canvas
    const rect = inputCanvas.getBoundingClientRect();
    return { 
        x: event.clientX - rect.left,
        y: event.clientY - rect.top };
}

inputCanvas.onpointerdown = (event) => {
    drawing = true; if(recordStrokes) strokes = [];

    // reset current path and move canvas pen 
    // to current mouse position without drawing
    const pos = getMousePos(event);
    inputCtx.beginPath();
    inputCtx.moveTo(pos.x, pos.y);

    // grab all further pointer events
    // auto releases on pointer up event
    inputCanvas.setPointerCapture(event.pointerId);
}

inputCanvas.onpointermove = (event) => {
    if (!drawing) return;

    // draw a line frome last canvas pen 
    // position up to current mouse position
    const pos = getMousePos(event);
    inputCtx.lineTo(pos.x, pos.y);
    inputCtx.stroke();

    update(); if(recordStrokes) record(event);
}

function stopDrawing() { 
    drawing = false; 
    
    if(recordStrokes) {
        const startTime = strokes[0].t;
        strokes = strokes.map(({x,y,t}) => ({x,y,t: t-startTime}));
        console.log(JSON.stringify(strokes)); 
    }
}

inputCanvas.onpointerup = stopDrawing;
inputCanvas.onpointercancel = stopDrawing;

deleteBtn.onclick = () => {
    const width = inputCanvas.width;
    const height = inputCanvas.height;
    inputCtx.clearRect(0, 0, width, height);
    update();
}

function record(event) {
    strokes.push({ ...getMousePos(event), t: performance.now() });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function drawStrokes(strokes) {
    // block pointer events while auto draw
    inputCanvas.style.pointerEvents = "none";
    
    // go to first point in strokes
    let pos = strokes[0];
    inputCtx.beginPath();
    inputCtx.moveTo(pos.x, pos.y);

    // draw lines to all other points
    for (let i = 1; i < strokes.length; i++) {
        pos = strokes[i];
        inputCtx.lineTo(pos.x, pos.y);
        inputCtx.stroke();
        
        // update output then sleep
        update();
        await sleep(pos.t / 1000);
    }

    // unblock pointer events after auto draw
    inputCanvas.style.pointerEvents = "auto";
}

function randInt(min, max) {
    // min and max are inclusive here
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// -------------------------------------------------------------- //
// --------------------- NETWORK PIPELINE ----------------------- //
// -------------------------------------------------------------- //

function findBounds(imageData, padding = 20) {
    const { width, height, data } = imageData;

    // find bounds of pixels not empty in image data
    let minX = width, minY = height;
    let maxX = -1, maxY = -1;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // grab rgba data from given list
            const i = (y * width + x) * 4;
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const a = data[i + 3];

            // test if empty pixel (r,g,b,a = 0)
            if (a != 0 && (r + g + b) != 0) {
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
        }
    }

    // fully empty image, bounds full
    if (maxX == -1 && maxY == -1) 
        return { top: 0, bottom: height, left: 0, right: width };

    // not fully empty image, bounds calc
    return {
        top: Math.max(minY - padding, 0),
        bottom: Math.max(height - 1 - maxY -padding, 0),
        left: Math.max(minX - padding, 0),
        right: Math.max(width - 1 - maxX - padding, 0)
    };
}

function intensifyPixels(imageData, factor = 1) {
    const { width, height, data } = imageData;
    const clamp = (x) => Math.max(0, Math.min(x, 255));

    // go over each pixel and increase its value
    for (let i = 0; i < width * height * 4; i++) 
        data[i] = clamp(data[i] * factor);
}

function getNetworkInput(data, inLow = -0.5, inHigh = 0.5) {
    // go over each value in given data and 
    // transform to expected network range
    let networkInput = [];
    for (let i = 0; i < data.length; i += 4) {
        const value = data[i]; // r=g=b
        const alpha = data[i + 3] / 255.0;
        networkInput.push(value * alpha / 255.0  * (inHigh - inLow) + inLow);
    }

    return networkInput;
}

function argmax(arr) {
    // find the index of the maximum element in arr
    let max = -Infinity;
    let idx = -1;

    for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v > max) { max = v; idx = i; }
    }

    return idx;
}

function queryNetwork(input) {
    // query network for output vector
    const output = queryMnist(input);
    const signal = output.toJs()[0];
    output.destroy();

    return signal.map(x => x[0]);
}


// -------------------------------------------------------------- //
// --------------------- MAIN UPDATE CODE ----------------------- //
// -------------------------------------------------------------- //
function update() {
    // get input data from input canvas
    const inputWidth = inputCanvas.width;
    const inputHeight = inputCanvas.height;
    const inputData = inputCtx.getImageData(0, 0, inputWidth, inputHeight);

    // create temporary canvas to transform data
    const tempCanvas = document.createElement('canvas');
    const dWidth = 28, dHeight = 28;
    tempCanvas.width = dWidth;
    tempCanvas.height = dHeight;

    // get temporary context and set it up
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.imageSmoothingEnabled = 'false';
    tempCtx.filter = "blur(0.5px) grayscale(100%) invert(100%)";

    // crop and scale input
    const bounds = findBounds(inputData);
    const sx = bounds.left, sy = bounds.top;
    const sWidth = inputWidth - sx - bounds.right;
    const sHeight = inputHeight - sy - bounds.bottom;
    
    const dx = 0, dy = 0;
    tempCtx.drawImage(
        inputCanvas, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight);

    // show transformed input on mnist canvas
    const tempData = tempCtx.getImageData(dx, dy, dWidth, dHeight);
    intensifyPixels(tempData, 8);
    mnistCtx.putImageData(tempData, 0, 0);
    
    // prepare network input
    const networkInput = getNetworkInput(tempData.data);
    let networkOutput = queryNetwork(networkInput);
    
    // if canvas empty show 0-eval
    if (inputData.data.every(x => x == 0))
        networkOutput = new Array(10).fill(0);

    // show network output on chart
    outChart.data.datasets[0].data = networkOutput;
    outChart.data.datasets[0].backgroundColor =
        networkOutput.map((_, i) => i == argmax(networkOutput) 
            ? 'rgba(70, 135, 235, 0.9)' : 'rgba(10, 10, 10, 0.9)');
    outChart.update('active');
}

// hide load indicator and draw random 
// digitonce setup has finished loading
const loadIndicator = document.getElementById('load-indicator');
loadIndicator.classList.add('hidden');

await drawStrokes(exampleStrokes[randInt(0, 9)]);
