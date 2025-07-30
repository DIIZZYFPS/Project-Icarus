import Index from "./pages/Index"
import { ThemeProvider } from "./components/theme/themeprovider"

function App() {


  return (
    <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
      <Index />
    </ThemeProvider>
  )
}

export default App

