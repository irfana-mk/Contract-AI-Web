import { useState } from 'react'
import axios from 'axios'
import {
  UploadCloud,
  FileText,
  Loader2,
  AlertCircle,
  CheckCircle2,
  ShieldCheck,
  Database,
  ScrollText,
  FileCheck2,
  Plus,
  X,
  Tag,
} from 'lucide-react'
import './index.css'

const DEFAULT_FIELDS = [
  'vendor',
  'client',
  'contract_type',
  'start_date',
  'monthly_fee',
  'payment_terms',
  'service_tier',
  'minimum_uptime_percent',
  'penalty_percent',
  'evaluation_period',
]

function App() {
  const [policyFile, setPolicyFile] = useState(null)
  const [contractFile, setContractFile] = useState(null)
  const [fields, setFields] = useState([...DEFAULT_FIELDS])
  const [newField, setNewField] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [contractData, setContractData] = useState(null)
  const [step, setStep] = useState('')

  const handlePolicyChange = (e) => {
    if (e.target.files?.[0]) {
      setPolicyFile(e.target.files[0])
      setError(null)
    }
  }

  const handleContractChange = (e) => {
    if (e.target.files?.[0]) {
      setContractFile(e.target.files[0])
      setError(null)
    }
  }

  const addField = () => {
    const trimmed = newField.trim()
    if (trimmed && !fields.includes(trimmed)) {
      setFields([...fields, trimmed])
      setNewField('')
    }
  }

  const removeField = (index) => {
    setFields(fields.filter((_, i) => i !== index))
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') addField()
  }

  const handleAnalyze = async () => {
    if (!contractFile || !policyFile || fields.length === 0) return

    setLoading(true)
    setError(null)
    setContractData(null)
    setStep('Chunking & indexing policy into Qdrant...')

    const formData = new FormData()
    formData.append('contract_file', contractFile)
    formData.append('policy_file', policyFile)
    formData.append('fields', JSON.stringify(fields))

    try {
      setStep('Extracting data & enforcing policy rules...')
      const response = await axios.post('http://127.0.0.1:8000/analyze/', formData)
      setContractData(response.data)
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.response?.data?.error ||
          err.message ||
          'An error occurred during analysis.',
      )
    } finally {
      setLoading(false)
      setStep('')
    }
  }

  const formatLabel = (key) =>
    key
      .split(/[\s_]+/)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ')

  const getReasonType = (reasoning) => {
    if (!reasoning) return 'default'
    const r = reasoning.toLowerCase()
    if (r.startsWith('policy override')) return 'policy'
    if (r.startsWith('inferred from policy')) return 'inferred'
    if (r.startsWith('not found')) return 'notfound'
    if (r.startsWith('extracted from contract')) return 'extracted'
    return 'default'
  }

  const canAnalyze = !!contractFile && !!policyFile && fields.length > 0

  const resultEntries = contractData && !contractData.error
    ? Object.entries(contractData)
    : []

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <ShieldCheck className="sidebar-logo" size={24} />
          <h2>Contract AI</h2>
        </div>
        <div className="sidebar-nav">
          <div className="nav-item active">
            <FileText size={18} />
            <span>Extractor</span>
          </div>
          <div className="status-badge-sidebar">
            <Database size={14} className="status-icon" />
            <span>RAG Active</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-area">
        <div className="content-container">
          <header className="top-header">
            <h1>Contract Analysis</h1>
            <p className="top-subtitle">
              Upload a <strong>Policy PDF</strong> and an <strong>Agreement PDF</strong>, define the fields you want extracted, and get policy-enforced results with full reasoning.
            </p>
          </header>

          <div className="tab-content fade-in">
            {/* Upload + Field Config Card */}
            <div className="upload-card">

              {/* Dual Upload */}
              <div className="dual-upload-grid">
                {/* Policy Upload */}
                <div className={`upload-zone-box ${policyFile ? 'file-selected' : ''}`}>
                  <div className="upload-zone-label">
                    <ScrollText size={16} />
                    <span>Policy Document</span>
                    <span className="required-tag">Required</span>
                  </div>
                  <div className="upload-zone">
                    <input
                      type="file"
                      id="policyFile"
                      accept=".pdf"
                      onChange={handlePolicyChange}
                      className="file-input"
                    />
                    <label htmlFor="policyFile" className="upload-label">
                      <UploadCloud size={30} className="upload-icon" />
                      <span className="upload-text">
                        {policyFile ? policyFile.name : 'Select Policy PDF'}
                      </span>
                      <span className="upload-subtext">Chunked &amp; indexed into Qdrant on each run</span>
                    </label>
                  </div>
                  {policyFile && (
                    <div className="file-selected-badge">
                      <FileCheck2 size={14} />
                      <span>Ready</span>
                    </div>
                  )}
                </div>

                {/* Agreement Upload */}
                <div className={`upload-zone-box ${contractFile ? 'file-selected' : ''}`}>
                  <div className="upload-zone-label">
                    <FileText size={16} />
                    <span>Agreement Document</span>
                    <span className="required-tag">Required</span>
                  </div>
                  <div className="upload-zone">
                    <input
                      type="file"
                      id="contractFile"
                      accept=".pdf"
                      onChange={handleContractChange}
                      className="file-input"
                    />
                    <label htmlFor="contractFile" className="upload-label">
                      <UploadCloud size={30} className="upload-icon" />
                      <span className="upload-text">
                        {contractFile ? contractFile.name : 'Select Agreement PDF'}
                      </span>
                      <span className="upload-subtext">Service agreement to extract &amp; analyse</span>
                    </label>
                  </div>
                  {contractFile && (
                    <div className="file-selected-badge">
                      <FileCheck2 size={14} />
                      <span>Ready</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Dynamic Field Manager */}
              <div className="field-manager">
                <div className="field-manager-header">
                  <Tag size={14} />
                  <span>Fields to Extract</span>
                  <span className="field-count">{fields.length} field{fields.length !== 1 ? 's' : ''}</span>
                </div>

                <div className="field-chips">
                  {fields.map((field, i) => (
                    <div key={i} className="field-chip">
                      <span>{field}</span>
                      <button
                        className="field-chip-remove"
                        onClick={() => removeField(i)}
                        title={`Remove "${field}"`}
                      >
                        <X size={11} />
                      </button>
                    </div>
                  ))}
                  {fields.length === 0 && (
                    <span className="field-chips-empty">Add at least one field to extract</span>
                  )}
                </div>

                <div className="field-add-row">
                  <input
                    type="text"
                    className="field-input"
                    placeholder="Type a field name and press Enter (e.g. payment_terms)"
                    value={newField}
                    onChange={(e) => setNewField(e.target.value)}
                    onKeyDown={handleKeyDown}
                  />
                  <button
                    className="field-add-btn"
                    onClick={addField}
                    disabled={!newField.trim()}
                  >
                    <Plus size={15} />
                    Add
                  </button>
                </div>
              </div>

              {/* Analyse Button */}
              <button
                className="extract-btn"
                onClick={handleAnalyze}
                disabled={!canAnalyze || loading}
              >
                {loading ? (
                  <>
                    <Loader2 size={18} className="spinner" />
                    {step || 'Processing...'}
                  </>
                ) : (
                  'Analyse Document'
                )}
              </button>

              {!canAnalyze && !loading && (
                <p className="upload-hint">
                  {!policyFile && !contractFile
                    ? 'Upload both PDFs to begin'
                    : !policyFile
                    ? 'Policy PDF is required'
                    : !contractFile
                    ? 'Agreement PDF is required'
                    : 'Add at least one field to extract'}
                </p>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="error-message">
                <AlertCircle size={18} />
                <p>{error}</p>
              </div>
            )}

            {/* Error inside response */}
            {contractData?.error && (
              <div className="error-message">
                <AlertCircle size={18} />
                <p>{contractData.error}</p>
              </div>
            )}

            {/* Results */}
            {resultEntries.length > 0 && (
              <div className="result-container fade-in">
                <div className="result-header">
                  <CheckCircle2 size={20} className="success-icon" />
                  <h2>Analysis Results</h2>
                  <span className="result-field-count">
                    {resultEntries.length} field{resultEntries.length !== 1 ? 's' : ''} extracted
                  </span>
                </div>

                {/* Legend */}
                <div className="legend-row">
                  <span className="legend-item legend-item--extracted">Extracted from contract</span>
                  <span className="legend-item legend-item--policy">Policy override</span>
                  <span className="legend-item legend-item--inferred">Inferred from policy</span>
                  <span className="legend-item legend-item--notfound">Not found</span>
                </div>

                <div className="data-grid">
                  {resultEntries.map(([fieldName, fieldData]) => {
                    const value = fieldData?.value
                    const reasoning = fieldData?.reasoning || ''
                    const isNull = value === null || value === undefined || value === ''
                    const reasonType = getReasonType(reasoning)
                    const isOverridden = reasonType === 'policy' || reasonType === 'inferred'

                    return (
                      <div
                        key={fieldName}
                        className={`data-card ${isOverridden ? 'data-card--changed' : ''}`}
                      >
                        <div className="data-label">{formatLabel(fieldName)}</div>
                        <div className={`data-value ${isNull ? 'null-value' : ''}`}>
                          {isNull ? '—' : String(value)}
                        </div>
                        {reasoning && (
                          <div className={`reasoning-block reasoning-block--${reasonType}`}>
                            {reasoning}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
