import { cn } from "@/lib/utils";
import { Mic, MicOff } from "lucide-react";

interface TranscriptionDisplayProps {
  transcription: string;
  isListening: boolean;
  className?: string;
}

export const TranscriptionDisplay = ({ 
  transcription, 
  isListening, 
  className 
}: TranscriptionDisplayProps) => {
  return (
    <div className={cn(
      "bg-transcription border border-border rounded-lg p-4 backdrop-blur-sm",
      className
    )}>
      <div className="flex items-center gap-2 mb-2">
        {isListening ? (
          <Mic className="w-4 h-4 text-voice-active animate-pulse-glow" />
        ) : (
          <MicOff className="w-4 h-4 text-voice-inactive" />
        )}
        <span className="text-sm font-medium text-muted-foreground">
          {isListening ? "Listening..." : "Voice inactive"}
        </span>
      </div>
      
      <div className="min-h-[3rem] flex items-center">
        {transcription ? (
          <p className="text-foreground leading-relaxed">{transcription}</p>
        ) : (
          <p className="text-muted-foreground italic">
            {isListening ? "Start speaking..." : "Press the microphone to start"}
          </p>
        )}
      </div>
    </div>
  );
};