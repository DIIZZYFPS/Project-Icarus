import { cn } from "@/lib/utils";
import { Bot, User } from "lucide-react";

interface ChatMessageProps {
  message: string;
  isUser: boolean;
  timestamp: Date;
  className?: string;
}

export const ChatMessage = ({ message, isUser, timestamp, className }: ChatMessageProps) => {
  return (
    <div className={cn(
      "flex gap-3 p-4 rounded-lg transition-all duration-300 hover:scale-[1.02]",
      isUser 
        ? "bg-message-user/10 border border-message-user/20 ml-8" 
        : "bg-message-ai/10 border border-message-ai/20 mr-8",
      className
    )}>
      <div className={cn(
        "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
        isUser 
          ? "bg-gradient-voice" 
          : "bg-gradient-primary"
      )}>
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-white" />
        )}
      </div>
      
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">
            {isUser ? "You" : "AI Assistant"}
          </span>
          <span className="text-xs text-muted-foreground">
            {timestamp.toLocaleTimeString()}
          </span>
        </div>
        <p className="text-foreground leading-relaxed">{message}</p>
      </div>
    </div>
  );
};