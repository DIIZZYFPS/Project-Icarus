import { app, BrowserWindow, ipcMain } from "electron";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow;
let pythonProcess = null;

// ═══════════════════════════════════════════════════════════════════════════════
// Python Client Process Management
// ═══════════════════════════════════════════════════════════════════════════════

function getPythonCommand() {
    // Try python3 first (Linux/Mac), fall back to python (Windows)
    return process.platform === 'win32' ? 'python' : 'python3';
}

function getClientPath() {
    // Navigate from frontend/electron to Client/client.py
    return path.resolve(__dirname, '..', '..', 'Client', 'client.py');
}

function spawnPythonClient() {
    const pythonCmd = getPythonCommand();
    const clientPath = getClientPath();
    
    console.log(`[Electron] Spawning Python client: ${pythonCmd} ${clientPath}`);
    
    pythonProcess = spawn(pythonCmd, [clientPath], {
        cwd: path.dirname(clientPath),
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    // Buffer for handling partial lines
    let stdoutBuffer = '';

    pythonProcess.stdout.on('data', (data) => {
        stdoutBuffer += data.toString();
        
        // Process complete lines
        const lines = stdoutBuffer.split('\n');
        stdoutBuffer = lines.pop(); // Keep incomplete line in buffer
        
        for (const line of lines) {
            if (line.startsWith('IPC_JSON:')) {
                try {
                    const jsonStr = line.substring(9); // Remove 'IPC_JSON:' prefix
                    const message = JSON.parse(jsonStr);
                    handleIPCMessage(message);
                } catch (e) {
                    console.error('[Electron] Failed to parse IPC message:', line, e);
                }
            } else if (line.trim()) {
                // Log non-IPC output for debugging
                console.log(`[Python] ${line}`);
            }
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`[Python Error] ${data.toString()}`);
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('voice:error', data.toString());
        }
    });

    pythonProcess.on('close', (code) => {
        console.log(`[Electron] Python client exited with code ${code}`);
        pythonProcess = null;
        
        // Notify renderer that client disconnected
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('voice:state', 'DISCONNECTED');
        }
    });

    pythonProcess.on('error', (err) => {
        console.error('[Electron] Failed to start Python client:', err);
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('voice:error', `Failed to start voice client: ${err.message}`);
        }
    });
}

function handleIPCMessage(message) {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    
    const { type, payload } = message;
    
    switch (type) {
        case 'READY':
            mainWindow.webContents.send('voice:ready');
            break;
        case 'STATE':
            mainWindow.webContents.send('voice:state', payload);
            break;
        case 'TRANSCRIPT':
            mainWindow.webContents.send('voice:transcript', payload);
            break;
        case 'RESPONSE':
            mainWindow.webContents.send('voice:response', payload);
            break;
        case 'WAKE_WORD':
            mainWindow.webContents.send('voice:wakeword', payload);
            break;
        case 'ERROR':
            mainWindow.webContents.send('voice:error', payload);
            break;
        case 'LOG':
            mainWindow.webContents.send('voice:log', payload);
            break;
        default:
            console.log('[Electron] Unknown IPC message type:', type);
    }
}

function killPythonClient() {
    if (pythonProcess) {
        console.log('[Electron] Killing Python client...');
        pythonProcess.kill('SIGTERM');
        pythonProcess = null;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Electron Window Management
// ═══════════════════════════════════════════════════════════════════════════════

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        }
    });

    mainWindow.loadURL("http://localhost:5173");
    // Remove the bar
    mainWindow.setMenuBarVisibility(false);

    mainWindow.on("closed", () => {
        mainWindow = null;
    });
    
    // Spawn Python client after window is ready
    mainWindow.webContents.on('did-finish-load', () => {
        spawnPythonClient();
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// IPC Handlers (from Renderer)
// ═══════════════════════════════════════════════════════════════════════════════

ipcMain.on('voice:start', () => {
    console.log('[Electron] Received voice:start from renderer');
    // Python client handles wake word detection automatically
    // This could be used to send commands to Python if needed
});

ipcMain.on('voice:stop', () => {
    console.log('[Electron] Received voice:stop from renderer');
    // Could send interrupt signal to Python if needed
});

// ═══════════════════════════════════════════════════════════════════════════════
// App Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

app.on("ready", createWindow);

app.on("window-all-closed", () => {
    killPythonClient();
    if (process.platform !== "darwin") {
        app.quit();
    }
});

app.on("before-quit", () => {
    killPythonClient();
});

app.on("activate", () => {
    if (mainWindow === null) {
        createWindow();
    }
});