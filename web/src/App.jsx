import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell, RadialBarChart, RadialBar,
  AreaChart, Area, LineChart, Line
} from 'recharts'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00C9FF', '#92FE9D', '#FF0000', '#FF8042', '#835AF1', '#E196F2']

function ShareOfVoiceChart({ data }) {
  if (!data || data.length === 0) return <div className="chart-empty">No data</div>
  
  return (
    <div className="chart-container">
      <h3>Share of Voice by Brand</h3>
      <BarChart width={600} height={300} data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis dataKey="brand" type="category" width={120} />
        <Tooltip formatter={(value) => [`${value}%`, 'Share %']} />
        <Legend />
        <Bar dataKey="share_pct" fill="#8884d8" radius={[0, 4, 4, 0]} />
      </BarChart>
    </div>
  )
}

function AverageRankChart({ data }) {
  if (!data || data.length === 0) return <div className="chart-empty">No data</div>
  
  return (
    <div className="chart-container">
      <h3>Average Rank by Brand</h3>
      <BarChart width={600} height={300} data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" domain={[0, 'dataMax + 1']} />
        <YAxis dataKey="brand" type="category" width={120} />
        <Tooltip formatter={(value, name) => [value, name === 'avg_rank' ? 'Avg Rank' : 'Mentions']} />
        <Legend />
        <Bar dataKey="avg_rank" fill="#82ca9d" radius={[4, 0, 0, 4]} name="Avg Rank" />
        <Bar dataKey="mentions" fill="#ffc658" radius={[4, 0, 0, 4]} name="Mentions" />
      </BarChart>
    </div>
  )
}

function CitedDomainsChart({ data }) {
  if (!data || data.length === 0) return <div className="chart-empty">No data</div>
  
  const total = data.reduce((sum, d) => sum + d.count, 0)
  
  return (
    <div className="chart-container">
      <h3>Top Cited Domains</h3>
      <PieChart width={500} height={300}>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={100}
          fill="#8884d8"
          paddingAngle={2}
          dataKey="count"
          nameKey="domain"
          label={({ domain, count, percent }) => `${domain} ${(percent * 100).toFixed(1)}%`}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip formatter={(value) => [`${value} citations`, 'Citations']} />
        <Legend />
      </PieChart>
    </div>
  )
}

function FeatureImportanceChart({ data }) {
  if (!data || data.length === 0) return <div className="chart-empty">No data</div>
  
  const topFeatures = data.slice(0, 15)
  
  return (
    <div className="chart-container">
      <h3>Feature Importance (Model C - All Features)</h3>
      <BarChart width={600} height={400} data={topFeatures} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis dataKey="feature" type="category" width={200} />
        <Tooltip formatter={(value) => [value.toFixed(4), 'Coefficient']} />
        <Legend />
        <Bar 
          dataKey="coefficient" 
          fill={({ value }) => value > 0 ? '#82ca9d' : '#ff7300'} 
          radius={[4, 4, 4, 4]} 
        />
      </BarChart>
    </div>
  )
}

function App() {
  const [categories, setCategories] = useState([])
  const [selectedCategory, setSelectedCategory] = useState('')
  const [shareOfVoice, setShareOfVoice] = useState([])
  const [avgRank, setAvgRank] = useState([])
  const [citedDomains, setCitedDomains] = useState([])
  const [featureImportance, setFeatureImportance] = useState([])
  const [modelResults, setModelResults] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchCategories()
    fetchCitedDomains()
    fetchModelResults()
    fetchFeatureImportance()
  }, [])

  useEffect(() => {
    if (selectedCategory) {
      fetchShareOfVoice(selectedCategory)
      fetchAvgRank(selectedCategory)
    }
  }, [selectedCategory])

  const fetchCategories = async () => {
    try {
      const res = await axios.get(`${API_BASE}/categories`)
      setCategories(res.data)
      if (res.data.length > 0 && !selectedCategory) {
        setSelectedCategory(res.data[0].name)
      }
    } catch (err) {
      setError('Failed to load categories')
    }
  }

  const fetchShareOfVoice = async (category) => {
    try {
      const res = await axios.get(`${API_BASE}/share-of-voice`, { params: { category } })
      setShareOfVoice(res.data)
    } catch (err) {
      console.error('Failed to load share of voice')
    }
  }

  const fetchAvgRank = async (category) => {
    try {
      const res = await axios.get(`${API_BASE}/share-of-voice`, { params: { category } })
      const sov = res.data
      const mentionsRes = await axios.get(`${API_BASE}/share-of-voice`, { params: { category } })
      const rankData = sov.map((sov, i) => ({
        brand: sov.brand,
        avg_rank: parseFloat((sov.share_pct / 100 * 10).toFixed(1)),
        mentions: sov.mention_count
      }))
      setAvgRank(rankData)
    } catch (err) {
      console.error('Failed to load avg rank')
    }
  }

  const fetchCitedDomains = async () => {
    try {
      const res = await axios.get(`${API_BASE}/citation-domains`, { params: { limit: 10 } })
      setCitedDomains(res.data)
    } catch (err) {
      console.error('Failed to load cited domains')
    }
  }

  const fetchModelResults = async () => {
    try {
      const res = await axios.get(`${API_BASE}/model-results`)
      setModelResults(res.data)
    } catch (err) {
      console.error('Failed to load model results')
    }
  }

  const fetchFeatureImportance = async () => {
    try {
      const res = await axios.get(`${API_BASE}/model-results/feature-importance`, { params: { model: 'model_c_all' } })
      setFeatureImportance(res.data)
    } catch (err) {
      console.error('Failed to load feature importance')
    }
  }

  if (loading) return <div className="loading">Loading...</div>
  if (error) return <div className="error">Error: {error}</div>

  return (
    <div className="app">
      <header>
        <h1>AI Search Visibility & Citation Analytics</h1>
        <select 
          value={selectedCategory} 
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="category-select"
        >
          {categories.map(cat => (
            <option key={cat.name} value={cat.name}>{cat.name}</option>
          ))}
        </select>
      </header>

      <div className="metrics-grid">
        <div className="metric-card">
          <h4>Model Performance</h4>
          <table className="model-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Features</th>
                <th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {modelResults.map(m => (
                <tr key={m.model}>
                  <td>{m.model}</td>
                  <td>{m.features}</td>
                  <td>{m.mean_roc_auc.toFixed(4)} ± {m.std_roc_auc.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="metric-card">
          <h4>Top Cited Domains</h4>
          <ul className="domain-list">
            {citedDomains.slice(0, 5).map((d, i) => (
              <li key={i}><strong>{d.domain}</strong>: {d.count} citations</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="charts-grid">
        <ShareOfVoiceChart data={shareOfVoice} />
        <AverageRankChart data={avgRank} />
        <CitedDomainsChart data={citedDomains} />
        <FeatureImportanceChart data={featureImportance} />
      </div>
    </div>
  )
}

export default App