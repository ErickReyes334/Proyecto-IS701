import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, Users, Clock, Brain } from "lucide-react"

const stats = [
  {
    title: "Sesiones Hoy",
    value: "247",
    change: "+12%",
    icon: Users,
    color: "text-blue-600",
  },
  {
    title: "Tiempo Promedio",
    value: "23min",
    change: "-5%",
    icon: Clock,
    color: "text-green-600",
  },
  {
    title: "Precisión IA",
    value: "94.2%",
    change: "+2.1%",
    icon: Brain,
    color: "text-purple-600",
  },
  {
    title: "Alertas Generadas",
    value: "18",
    change: "+8%",
    icon: TrendingUp,
    color: "text-orange-600",
  },
]

const recentAlerts = [
  { time: "14:32", type: "Estrés Alto", user: "Usuario #1247", severity: "high" },
  { time: "14:28", type: "Fatiga Mental", user: "Usuario #1203", severity: "medium" },
  { time: "14:15", type: "Ansiedad Detectada", user: "Usuario #1189", severity: "medium" },
  { time: "14:02", type: "Sobrecarga Cognitiva", user: "Usuario #1156", severity: "high" },
]

export function StatsOverview() {
  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-2 gap-4">
        {stats.map((stat, index) => (
          <Card key={index}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-600 mb-1">{stat.title}</p>
                  <p className="text-lg font-bold text-gray-900">{stat.value}</p>
                  <p className={`text-xs ${stat.change.startsWith("+") ? "text-green-600" : "text-red-600"}`}>
                    {stat.change}
                  </p>
                </div>
                <stat.icon className={`w-8 h-8 ${stat.color}`} />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Recent Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Alertas Recientes</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentAlerts.map((alert, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-2 h-2 rounded-full ${alert.severity === "high" ? "bg-red-500" : "bg-yellow-500"}`}
                  ></div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{alert.type}</p>
                    <p className="text-xs text-gray-600">{alert.user}</p>
                  </div>
                </div>
                <span className="text-xs text-gray-500">{alert.time}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
