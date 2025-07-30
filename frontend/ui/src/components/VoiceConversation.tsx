import { useState, useCallback, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { VoiceIndicator } from "./VoiceIndicator";
import { ChatMessage } from "./ChatMessage";
import { TranscriptionDisplay } from "./TranscriptionDisplay";
import { Mic, MicOff, Volume2, VolumeX, Bot } from "lucide-react";
import { toast } from "sonner";

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

  // Simulate voice interaction for demo purposes
  const simulateVoiceInteraction = useCallback(() => {
    // Simulate transcription
    setTimeout(() => {
      setCurrentTranscription("Hello, how are you today?");
    }, 1000);

    // Simulate message completion
    setTimeout(() => {
      addMessage("Hello, how are you today?", true);
      setCurrentTranscription("");
      setIsSpeaking(true);
    }, 3000);

    // Simulate AI response
    setTimeout(() => {
      addMessage("Hello! I'm doing well, thank you for asking. How can I assist you today?", false);
      setIsSpeaking(false);
    }, 5000);
  }, [addMessage]);

  const startListening = useCallback(async () => {
    try {
      // Request microphone permission
      await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsListening(true);
      setIsConnected(true);
      
      // Here we would integrate with ElevenLabs conversation
      // For now, simulate voice interaction
      simulateVoiceInteraction();
      
      toast.success("Voice activated", {
        description: "Start speaking to the AI assistant",
      });
    } catch (error) {
      toast.error("Microphone access denied", {
        description: "Please allow microphone access to use voice features",
      });
    }
  }, [simulateVoiceInteraction]);

  const stopListening = useCallback(() => {
    setIsListening(false);
    setIsConnected(false);
    setCurrentTranscription("");
  }, []);

  const toggleMute = useCallback(() => {
    setVolume(prev => prev > 0 ? 0 : 0.8);
  }, []);

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
              {isConnected ? "Connected and ready" : "Click the microphone to start"}
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
            
            <ScrollArea ref={scrollAreaRef} className="h-[500px] p-4">
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
                    {isListening ? "Listening" : "Press to Talk"}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {isConnected ? "Voice connection active" : "Click to start conversation"}
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