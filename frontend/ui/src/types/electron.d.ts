/**
 * Type declarations for the Electron IPC bridge exposed via preload.js
 */

export interface WakeWordData {
    model: string;
    score: number;
}

export type VoiceState = 'IDLE' | 'CONNECTING' | 'LISTENING' | 'SPEAKING' | 'DISCONNECTED';

export interface ElectronAPI {
    // Voice control methods
    startListening: () => void;
    stopListening: () => void;

    // Event listeners
    onTranscript: (callback: (text: string) => void) => void;
    onResponse: (callback: (text: string) => void) => void;
    onStateChange: (callback: (state: VoiceState) => void) => void;
    onWakeWord: (callback: (data: WakeWordData) => void) => void;
    onError: (callback: (error: string) => void) => void;
    onReady: (callback: () => void) => void;
    onLog: (callback: (message: string) => void) => void;

    // Cleanup
    removeAllListeners: () => void;
}

declare global {
    interface Window {
        electronAPI: ElectronAPI;
    }
}

export {};
