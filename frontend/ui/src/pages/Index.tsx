import { Button } from "@/components/ui/button";
import { Mic } from "lucide-react";

export default function Index() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-primary ">
      <h1 className="text-4xl font-bold mb-4 text-secondary">Welcome to Project Icarus</h1>
      <Button variant="ghost" className="text-white hover:bg-accent w-60 h-60 p-0 flex items-center justify-center" onClick={() => alert("mic on")}>
        <Mic className="!w-52 !h-52" />
      </Button>
    </div>
  );
}