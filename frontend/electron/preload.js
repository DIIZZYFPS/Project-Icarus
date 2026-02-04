const { contextBridge, ipcRenderer } = require('electron');

/**
 * Preload script - Exposes a secure IPC bridge to the renderer process.
 * Uses contextBridge to safely expose specific IPC channels.
 */
contextBridge.exposeInMainWorld('electronAPI', {
    // Voice control methods
    startListening: () => ipcRenderer.send('voice:start'),
    stopListening: () => ipcRenderer.send('voice:stop'),

    // Event listeners for Python client messages
    onTranscript: (callback) => {
        ipcRenderer.on('voice:transcript', (_event, text) => callback(text));
    },
    onResponse: (callback) => {
        ipcRenderer.on('voice:response', (_event, text) => callback(text));
    },
    onStateChange: (callback) => {
        ipcRenderer.on('voice:state', (_event, state) => callback(state));
    },
    onWakeWord: (callback) => {
        ipcRenderer.on('voice:wakeword', (_event, data) => callback(data));
    },
    onError: (callback) => {
        ipcRenderer.on('voice:error', (_event, error) => callback(error));
    },
    onReady: (callback) => {
        ipcRenderer.on('voice:ready', (_event) => callback());
    },
    onLog: (callback) => {
        ipcRenderer.on('voice:log', (_event, message) => callback(message));
    },

    // Cleanup method to remove all listeners
    removeAllListeners: () => {
        ipcRenderer.removeAllListeners('voice:transcript');
        ipcRenderer.removeAllListeners('voice:response');
        ipcRenderer.removeAllListeners('voice:state');
        ipcRenderer.removeAllListeners('voice:wakeword');
        ipcRenderer.removeAllListeners('voice:error');
        ipcRenderer.removeAllListeners('voice:ready');
        ipcRenderer.removeAllListeners('voice:log');
    }
});
