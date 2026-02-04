import { useState, useCallback, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { VoiceIndicator } from "./VoiceIndicator";
import { ChatMessage } from "./ChatMessage";
import { TranscriptionDisplay } from "./TranscriptionDisplay";
import { Mic, MicOff, Volume2, VolumeX, Bot } from "lucide-react";
import { toast } from "sonner";
import type { VoiceState } from "@/types/electron";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

export const VoiceConversation = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [currentTranscription, setCurrentTranscription] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [voiceState, setVoiceState] = useState<VoiceState>("DISCONNECTED");
  const [volume, setVolume] = useState(0.8);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [messages]);

  const addMessage = useCallback((text: string, isUser: boolean) => {
    const newMessage: Message = {
      id: Math.random().toString(36).substr(2, 9),
      text,
      isUser,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
  }, []);

  // ═══════════════════════════════════════════════════════════════════════════════
  // Electron IPC Event Listeners
  // ═══════════════════════════════════════════════════════════════════════════════
  useEffect(() => {
    // Check if running in Electron
    if (!window.electronAPI) {
      console.warn("Not running in Electron - IPC not available");
      return;
    }

    // Python client is ready
    window.electronAPI.onReady(() => {
      setIsConnected(true);
      toast.success("Voice client connected", {
        description: "Say 'Hey Icarus' to activate",
      });
    });

    // Voice state changes from Python client
    window.electronAPI.onStateChange((state: VoiceState) => {
      setVoiceState(state);
      
      switch (state) {
        case "IDLE":
          setIsListening(false);
          setIsSpeaking(false);
          setCurrentTranscription("");
          break;
        case "CONNECTING":
          setIsListening(false);
          setIsSpeaking(false);
          break;
        case "LISTENING":
          setIsListening(true);
          setIsSpeaking(false);
          break;
        case "SPEAKING":
          setIsListening(false);
          setIsSpeaking(true);
          break;
        case "DISCONNECTED":
          setIsConnected(false);
          setIsListening(false);
          setIsSpeaking(false);
          toast.error("Voice client disconnected");
          break;
      }
    });

    // Real-time transcription from Python client
    window.electronAPI.onTranscript((text: string) => {
      setCurrentTranscription(text);
      // Add user message when transcription is complete
      if (text.trim()) {
        addMessage(text, true);
        setCurrentTranscription("");
      }
    });

    // AI response from Python client
    window.electronAPI.onResponse((text: string) => {
      addMessage(text, false);
    });

    // Wake word detected
    window.electronAPI.onWakeWord((data) => {
      toast.info("Wake word detected", {
        description: `${data.model} (confidence: ${Math.round(data.score * 100)}%)`,
      });
    });

    // Error from Python client
    window.electronAPI.onError((error: string) => {
      console.error("Voice client error:", error);
      toast.error("Voice client error", {
        description: error.substring(0, 100),
      });
    });

    // Debug logs
    window.electronAPI.onLog((message: string) => {
      console.log("[Python]", message);
    });

    // Cleanup listeners on unmount
    return () => {
      window.electronAPI?.removeAllListeners();
    };
  }, [addMessage]);

  const startListening = useCallback(async () => {
    if (!window.electronAPI) {
      toast.error("Not running in Electron", {
        description: "Voice features require the desktop app",
      });
      return;
    }
    
    window.electronAPI.startListening();
    toast.info("Wake word detection active", {
      description: "Say 'Hey Icarus' to start speaking",
    });
  }, []);

  const stopListening = useCallback(() => {
    if (window.electronAPI) {
      window.electronAPI.stopListening();
    }
    setIsListening(false);
    setCurrentTranscription("");
  }, []);

  const toggleMute = useCallback(() => {
    setVolume(prev => prev > 0 ? 0 : 0.8);
  }, []);

  // Status text based on current state
  const getStatusText = () => {
    switch (voiceState) {
      case "IDLE":
        return "Say 'Hey Icarus' to activate";
      case "CONNECTING":
        return "Connecting to server...";
      case "LISTENING":
        return "Listening...";
      case "SPEAKING":
        return "AI is responding...";
      case "DISCONNECTED":
        return "Voice client not connected";
      default:
        return "Click to start conversation";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-bg flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border/50">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              Voice Assistant
            </h1>
            <p className="text-muted-foreground">
              {isConnected ? getStatusText() : "Voice client not connected"}
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <VoiceIndicator 
              isActive={isListening || isSpeaking} 
              size="lg"
              className="animate-float"
            />
            <Button
              variant="outline"
              size="icon"
              onClick={toggleMute}
              className="border-border/50"
            >
              {volume > 0 ? (
                <Volume2 className="w-4 h-4" />
              ) : (
                <VolumeX className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-6">
        <div className="max-w-4xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
          
          {/* Chat Messages */}
          <Card className="lg:col-span-2 border-border/50 bg-card/50 backdrop-blur-sm">
            <div className="p-4 border-b border-border/50">
              <h2 className="font-semibold">Conversation History</h2>
            </div>

            <ScrollArea ref={scrollAreaRef} className="h-[75vh] p-4">
              <div className="space-y-4">
                {messages.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    <Bot className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No messages yet. Start a conversation!</p>
                  </div>
                ) : (
                  messages.map((message) => (
                    <ChatMessage
                      key={message.id}
                      message={message.text}
                      isUser={message.isUser}
                      timestamp={message.timestamp}
                    />
                  ))
                )}
                
                {isSpeaking && (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <VoiceIndicator isActive={true} size="sm" />
                    <span className="text-sm">AI is responding...</span>
                  </div>
                )}
              </div>
            </ScrollArea>
          </Card>

          {/* Controls & Transcription */}
          <div className="space-y-6">
            
            {/* Voice Control */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <div className="p-6 text-center space-y-4">
                <Button
                  size="lg"
                  variant={isListening ? "destructive" : "default"}
                  onClick={isListening ? stopListening : startListening}
                  className={`w-20 h-20 rounded-full transition-all duration-300 ${
                    isListening 
                      ? "shadow-glow-voice animate-pulse-glow" 
                      : "shadow-glow-primary hover:shadow-glow-voice"
                  }`}
                >
                  {isListening ? (
                    <MicOff className="w-8 h-8" />
                  ) : (
                    <Mic className="w-8 h-8" />
                  )}
                </Button>
                
                <div>
                  <p className="font-medium">
                    {isListening ? "Listening" : isSpeaking ? "Speaking" : "Idle"}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {getStatusText()}
                  </p>
                </div>
              </div>
            </Card>

            {/* Live Transcription */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <div className="p-4 border-b border-border/50">
                <h3 className="font-semibold">Live Transcription</h3>
              </div>
              <div className="p-4">
                <TranscriptionDisplay
                  transcription={currentTranscription}
                  isListening={isListening}
                />
              </div>
            </Card>

          </div>
        </div>
      </div>
    </div>
  );
};