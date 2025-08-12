import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Camera, Play, Pause, AlertTriangle, CheckCircle } from "lucide-react"

const cognitiveStates = [
  { name: "Estrés", value: 73, color: "bg-red-500", status: "high" },
  { name: "Fatiga Mental", value: 45, color: "bg-orange-500", status: "medium" },
  { name: "Ansiedad Visible", value: 28, color: "bg-yellow-500", status: "low" },
  { name: "Concentración", value: 82, color: "bg-green-500", status: "high" },
  { name: "Estado Relajado", value: 35, color: "bg-blue-500", status: "medium" },
  { name: "Sobrecarga Cognitiva", value: 67, color: "bg-purple-500", status: "high" },
]

export function CognitiveAnalyzer() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)

  useEffect(() => {
    if (isAnalyzing) {
      const interval = setInterval(() => {
        setCurrentFrame((prev) => (prev + 1) % 100)
      }, 100)
      return () => clearInterval(interval)
    }
  }, [isAnalyzing])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "high":
        return <AlertTriangle className="w-4 h-4 text-red-500" />
      case "medium":
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />
      default:
        return <CheckCircle className="w-4 h-4 text-green-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "high":
        return "bg-red-100 text-red-800 border-red-200"
      case "medium":
        return "bg-yellow-100 text-yellow-800 border-yellow-200"
      default:
        return "bg-green-100 text-green-800 border-green-200"
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Camera className="w-5 h-5" />
            <span>Análisis en Tiempo Real</span>
          </CardTitle>
          <Button
            onClick={() => setIsAnalyzing(!isAnalyzing)}
            variant={isAnalyzing ? "destructive" : "default"}
            size="sm"
          >
            {isAnalyzing ? (
              <>
                <Pause className="w-4 h-4 mr-2" />
                Pausar
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Iniciar
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Video Feed Simulation */}
        <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 to-purple-900/20">
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-white">
                <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium">{isAnalyzing ? "Analizando..." : "Cámara Inactiva"}</p>
                {isAnalyzing && <p className="text-sm opacity-75">Frame: {currentFrame}</p>}
              </div>
            </div>
          </div>
          {isAnalyzing && (
            <div className="absolute top-4 left-4">
              <Badge className="bg-red-600 text-white animate-pulse">● REC</Badge>
            </div>
          )}
        </div>

        {/* Cognitive States Analysis */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Estados Cognitivos Detectados</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {cognitiveStates.map((state, index) => (
              <div key={index} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(state.status)}
                    <span className="text-sm font-medium text-gray-700">{state.name}</span>
                  </div>
                  <Badge className={getStatusColor(state.status)}>{state.value}%</Badge>
                </div>
                <Progress value={state.value} className="h-2" />
              </div>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 mb-2">Recomendaciones del Sistema</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Se detecta nivel alto de estrés - considerar una pausa</li>
            <li>• Sobrecarga cognitiva elevada - reducir complejidad de tareas</li>
            <li>• Buena concentración mantenida - momento óptimo para tareas complejas</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}
