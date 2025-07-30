import { cn } from "@/lib/utils";

interface VoiceIndicatorProps {
  isActive: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export const VoiceIndicator = ({ isActive, size = "md", className }: VoiceIndicatorProps) => {
  const sizeClasses = {
    sm: "w-4 h-8",
    md: "w-6 h-12", 
    lg: "w-8 h-16"
  };

  const baseClasses = {
    sm: "w-1",
    md: "w-1.5",
    lg: "w-2"
  };

  return (
    <div className={cn("flex items-center justify-center gap-1", sizeClasses[size], className)}>
      {[...Array(5)].map((_, i) => (
        <div
          key={i}
          className={cn(
            "rounded-full transition-all duration-300",
            baseClasses[size],
            isActive 
              ? "bg-voice-active animate-voice-wave shadow-glow-voice" 
              : "bg-voice-inactive",
            isActive && `h-${Math.floor(Math.random() * 8) + 4}`
          )}
          style={{
            animationDelay: `${i * 0.1}s`,
            height: isActive ? `${Math.random() * 60 + 20}%` : "30%"
          }}
        />
      ))}
    </div>
  );
};